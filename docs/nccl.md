# NCCL with memory sharing: an NCCL fork for pairs CUDA cannot connect

> **EXPERIMENTAL.** On a Tesla V100 (in a PCIe 3.0 x4 chipset slot) + RTX 3090 (PCIe 4.0 x16) this fork moves large
> all-reduces **44 % faster** than NCCL does through host RAM and makes plain llama.cpp **29 % faster** on prompts. Most
> of llama.cpp's gain on this pair comes from llama.cpp itself, on stock NCCL (the
> [P2P guide's llama.cpp section](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus#llamacpp-build-it-with-nccl)):
> on top of those changes the fork adds **5 %** (974 -> 1,017-1,028 tokens/s). Tested on one pair for one evening;
> results below. Also run on a GTX 750 Ti (Maxwell) + RTX 3090, where the default ordering let wrong data through:
> on that card use `NCCL_DRMSHARE_ORDER=readback` ([Maxwell](#maxwell-a-gtx-750-ti-with-an-rtx-3090)).

NVIDIA's NCCL moves data between GPUs for CUDA programs. Between two cards that CUDA refuses to pair, like a Tesla V100
and an RTX 3090 (two GPU generations: no peer access), NCCL has no direct route and goes through host RAM. This page
builds **a fork of NCCL 2.32.6 with one more transport**: it moves NCCL's data card to card through DRMShare's shared
memory, with the cards' copy engines (through Vulkan), while CUDA still sees no peer access. Programs need no change:
the fork is a drop-in `libnccl.so.2`.

## Results

Tesla V100 (PCIe 3.0 x4, a slot wired through the board's chipset) + RTX 3090 (PCIe 4.0 x16), NVIDIA's closed driver
580.178.04 with DRMShare's `nvidia-drm`, NCCL 2.32.6, 2026-11-26. Every NCCL result checked; every llama.cpp
comparison with identical perplexity, chunk for chunk.

NCCL's `all_reduce_perf`, two cards:

| | 8 B | 2 MiB | 8 MiB | 128 MiB |
|---|---|---|---|---|
| NCCL (shared memory in host RAM) | 13.6 µs | 1.75 GB/s | 1.80 GB/s | 1.82 GB/s |
| this fork, `NCCL_DRMSHARE_ORDER=readback` | 13.0 µs | 1.52 GB/s | 1.94 GB/s | 1.94 GB/s |
| **this fork (default ordering, `release`)** | 13.0 µs | **2.18 GB/s** | **2.60 GB/s** | **2.63 GB/s** |

The ceiling is the V100's PCIe 3.0 x4 slot with both directions busy, about 2.8 GB/s: these numbers measure the slot as
much as the method. Two cards in faster slots move several times more (13.1 GB/s between two RTX 3090s on PCIe 4.0 x8
lanes, [dmabuf-615.md D5](dmabuf-615.md#d5-three-rtx-3090s)). Small all-reduces go through NCCL's own
shared memory either way (the fork's two lanes, below).

llama.cpp, Qwen3.8-27B Q8_0, `-sm tensor`, prompt 2048 tokens per second, 2 rounds each:

| llama.cpp | stock NCCL | **fork (`release`)** | fork, `readback` | generation, all |
|---|---|---|---|---|
| plain (upstream as of 2026-11-24) | 720 | **932-933 (+29 %)** | 836 (+16 %) | 33.1-33.3 |
| with its tensor-parallel speed-ups, `-ts 3/1 -ub 2048` (best prompts) | 974-975 | **1,017-1,028 (+5 %)** | 931-937 (-3 %, older build) | 33.7-33.8 |
| the same at `-ts 3/2` (best generation), `readback` | 921-927 | | 909-910 (-1.6 %) | 37.4 |
| the same at `-ts 3/2`, 8k / 32k-token prompts, `readback` | 821 / 552 | | 818 / 550 | 36.0 at 8k, 34.1-34.6 at 32k |

With llama.cpp's speed-ups, moving data overlaps with computing, and what counts is how little the transfer disturbs
the GPUs as well as its speed. The `readback` ordering's fetch-and-read-back steps cost more there than they save;
`release` (every card writes its own data, no read across) is fast enough to come out ahead.

## What you need

- **DRMShare working on your pair**: its `nvidia-drm` loaded with `peer_vidmem_dmabuf=1`, and its copy test passing
  ([volta-dmabuf.md](volta-dmabuf.md) for a V100, Pascal or Maxwell card, [dmabuf-615.md](dmabuf-615.md) for Turing and newer).
  Check: `cat /sys/module/nvidia_drm/parameters/peer_vidmem_dmabuf` prints `Y`.
- **Vulkan on both cards**: the Vulkan loader and NVIDIA's Vulkan driver (`vulkaninfo --summary` lists both GPUs), and
  the headers to build: `sudo apt install libvulkan-dev`.
- **A CUDA toolkit that builds for both cards.** A Tesla V100 (sm_70), a Pascal card or a Maxwell card (a GTX 750 Ti is
  sm_50) needs CUDA 12.x: CUDA 13 dropped Maxwell, Pascal and Volta. Ubuntu's
  `nvidia-cuda-toolkit` (12.4 on 26.04) works.
- **One program that drives both GPUs from one process** (llama.cpp does). Programs with one process per GPU are not
  supported yet.

## 1. Build the NCCL fork

The fork is pinned here as `tools/nccl-drmshare` ([Cinderella-Man/nccl-drmshare](https://github.com/Cinderella-Man/nccl-drmshare),
branch `2.32.6-drmshare`: one commit on NVIDIA's `v2.32.6-1` tag).

```bash
git submodule update --init --depth 1 tools/nccl-drmshare
cd tools/nccl-drmshare
make -j8 src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_86,code=sm_86"
```

Put your cards' compute capabilities in `NVCC_GENCODE` (5.0 for a GTX 750 Ti, 7.0 for a V100, 7.5 Turing, 8.6 RTX 30,
8.9 RTX 40). With
Ubuntu's toolkit instead of NVIDIA's (`/usr/local/cuda`), add `CUDA_HOME=/usr CUDA_LIB=/usr/lib/x86_64-linux-gnu
CUDA_INC=/usr/include`, and `CUDARTLIB=cudart` if `/usr/lib/x86_64-linux-gnu/libcudart_static.a` is missing. To build
only what llama.cpp uses (much faster), add `ONLY_FUNCS="AllReduce Sum f32 * *|AllReduce Sum bf16 * *|SendRecv"`.
The library lands in `build/lib/libnccl.so.2.32.6`.

## 2. Build llama.cpp with NCCL

Any llama.cpp with the CUDA backend's NCCL support builds against the fork as against NCCL itself (same ABI), for
example the copy pinned here as `tools/llama.cpp` ([llama.cpp-drmshare](https://github.com/Cinderella-Man/llama.cpp-drmshare),
which also carries the tensor-parallel changes of [step 3's faster run](#faster-llamacpp-drmshares-tensor-parallel-changes)):

```bash
cmake -S tools/llama.cpp -B tools/llama.cpp/build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_NCCL=ON \
  -DNCCL_ROOT=$PWD/tools/nccl-drmshare/build -DCMAKE_CUDA_ARCHITECTURES="70;86"
cmake --build tools/llama.cpp/build-cuda -j8 --target llama-bench llama-perplexity llama-server
```

CMake's output must say `Found NCCL`: without it llama.cpp builds without NCCL, silently.

## 3. Run

```bash
export LD_LIBRARY_PATH=$PWD/tools/nccl-drmshare/build/lib
export NCCL_DRMSHARE_TRANSPORT=1
export NCCL_DRMSHARE_SPLIT_BYTES=2097152    # 4194304 with NCCL_DRMSHARE_ORDER=readback
tools/llama.cpp/build-cuda/bin/llama-bench -m model.gguf -ngl 99 -sm tensor -p 2048 -n 128
```

| Setting | What it does |
|---|---|
| `NCCL_DRMSHARE_TRANSPORT=1` | turns the transport on. With none of the `NCCL_DRMSHARE_` variables set, the fork behaves as NCCL 2.32.6 |
| `NCCL_DRMSHARE_SPLIT_BYTES=<bytes>` | **two lanes**: all-reduces below go through NCCL's shared memory (about 13 µs for a small one), larger ones card to card. llama.cpp's generation makes about 128 small all-reduces per token; without the cut each pays the copy engine's 28-45 µs. Set it only together with `NCCL_DRMSHARE_TRANSPORT=1`: alone, it sends large transfers through NCCL's socket network |
| `NCCL_DRMSHARE_ORDER` | `release` (the default): every card writes its own data into the other, each copy followed by Vulkan's release of that memory to the other device; reads nothing across; 98 % of the raw speed; passed the stale-step test below on the V100 + RTX 3090. `readback`: the first fixed version, slower (see the tables). **On a GTX 750 Ti (Maxwell) `release` let wrong data through, `readback` did not: use `readback` there** ([Maxwell](#maxwell-a-gtx-750-ti-with-an-rtx-3090)) |
| `NCCL_DRMSHARE_NOREAD=<bus id>` | `readback` only: a card whose memory must not be read by the other card fetches its data itself. Set it for a **Tesla V100 in a chipset slot**, which can freeze when another card reads its memory ([troubleshooting](troubleshooting.md#a-tesla-v100-in-a-chipset-slot-freezes-xid-62-while-another-card-reads-its-memory)); `release` never reads across anyway. The bus id from `nvidia-smi --query-gpu=name,pci.bus_id --format=csv`, any unique part of it (`04:00`) |

### Faster: llama.cpp-drmshare's tensor-parallel changes

The copy pinned as `tools/llama.cpp` carries four changes to llama.cpp's tensor parallelism for cards linked through
host memory or a slow slot; they change nothing unless their settings are on, and they help with stock NCCL too (the
P2P guide's [llama.cpp section](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus#more-speed-tensor-parallel-changes-in-a-llamacpp-copy-experimental) explains each). With the fork's
settings above, add:

```bash
export GGML_CUDA_ALLREDUCE=split   # small all-reduces by llama.cpp's own path, large ones by NCCL
export GGML_META_ASYNC_AR=1        # a layer's all-reduces overlap the next part of its work
tools/llama.cpp/build-cuda/bin/llama-bench -m model.gguf -ngl 99 -sm tensor -ts 3/1 -ub 2048 -p 2048 -n 128
```

This is the run behind the 1,017-1,028 tokens/s in [Results](#results) (974-975 with stock NCCL; Qwen3.8-27B Q8_0 on
the V100 + RTX 3090, 33.7 generating).

- `-ts 3/1` gives three quarters of each layer to CUDA device 0. CUDA numbers the fastest card first, here the RTX
  3090; llama.cpp lists the devices in that order when it starts. Split by your cards' speed: on this pair `-ts 3/1`
  is the best for prompts and `-ts 3/2` for generation (37.4 tokens/s, Results).
- The overlap needs batches of 1,024 tokens or more (`-ub 1024` or `-ub 2048`) and cuts each layer into two chunks of
  tokens, so far only for the Qwen3.5 / 3.8 family (`qwen35` in llama.cpp). Other models still get the two lanes.

For a server, the same settings with the split for chat (at `-ts 3/1` the RTX 3090 has no room left for a 32k context
with this model):

```bash
tools/llama.cpp/build-cuda/bin/llama-server -m model.gguf -ngl 99 -sm tensor -ts 3/2 -ub 2048 -b 2048 -c 32768
```

llama-server's speed was not measured: the numbers on this page are llama-bench's.

## 4. Check that it is used

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT tools/llama.cpp/build-cuda/bin/llama-bench -m model.gguf -ngl 99 -sm tensor -p 2048 -n 16 -r 1 2>&1 | grep -E 'DRMShare|via'
```

You should see one line per card (`DRMShare: CUDA device 0 = Vulkan NVIDIA GeForce RTX 3090, copy queues: ...`) and
connection lines `via DRM`, made at the first large all-reduce; the small lane's lines say `via SHM/direct`. No
`via DRM` at all: check `NCCL_DRMSHARE_TRANSPORT` and that DRMShare's module is loaded.

## 5. Check that the results are right

The transport has to get PCIe's ordering right: a copy engine reports "done" when its writes have left, and on a link
with relaxed ordering the "data ready" mark could be seen before the data. The first version had exactly that bug
(one step read early). Check on your own cards:

- **NCCL's own test**, every result compared: build [nccl-tests](https://github.com/NVIDIA/nccl-tests) against the fork
  (`make MPI=0 NCCL_HOME=$PWD/tools/nccl-drmshare/build`), then `./build/all_reduce_perf -g 2 -b 8K -e 256M -f 2 -n 50 -c 1`:
  the `#wrong` columns must be 0.
- **Back to back**: nccl-tests' check starts each checked all-reduce from idle and did not catch the first version's
  bug (0 wrong in 7000 checks). Thousands of all-reduces back to back did: the first version gave 79 424 wrong
  elements at 64-512 KiB, the `readback` and `release` orderings 0 in 42 000 all-reduces each (two runs, 64 KiB-4 MiB,
  3000 per size). [`tools/stale_test.cu`](../tools/stale_test.cu): `nvcc -O2 -arch=native stale_test.cu -o stale_test
  -I<nccl>/build/include -L<nccl>/build/lib -lnccl`, then `./stale_test 65536 4194304 3000` with the fork's settings
  (add `NCCL_SHM_DISABLE=1 NCCL_NET_DISABLE_INTRA=1 NCCL_PROTO=Simple` and leave out the split to send every size through the transport).
- **llama.cpp's perplexity** with and without the `NCCL_DRMSHARE_` variables: with two cards the sums come out the same
  in either order, so every chunk should be identical.

## Maxwell: a GTX 750 Ti with an RTX 3090

The fork on a GTX 750 Ti (GM107, Maxwell, 2 GB, 256 MB window) in a PCIe 3.0 x4 chipset slot and an RTX 3090 (PCIe 4.0
x16), NVIDIA's closed 580.178.04 with DRMShare's `nvidia-drm` ([volta-dmabuf.md](volta-dmabuf.md#gtx-750-ti--rtx-3090-maxwell--ampere-no-v100)),
2026-11-27. Built with Ubuntu's CUDA 12.4 for `compute_50` + `sm_86` (NCCL builds for sm_50; nccl-tests' all-to-all
test needs sm_60 and up, so build `all_reduce_perf` alone: `make -C src MPI=0 ... ../build/all_reduce_perf`).

- **The transport comes up on the Maxwell card** (`DRMShare: CUDA device 0 = Vulkan NVIDIA GeForce GTX 750 Ti`,
  connections `via DRM`). `all_reduce_perf`, every result checked, 0 wrong:

  | | 8 B | 2 MiB | 8 MiB | 128 MiB |
  |---|---|---|---|---|
  | NCCL (shared memory in host RAM) | 13.7 µs | 1.75 GB/s | 1.82 GB/s | 1.85 GB/s |
  | **this fork, two lanes** | 13.5 µs | **2.24 GB/s** | **2.62 GB/s** | **2.66 GB/s** |
  | this fork, every all-reduce card to card | 122.7 µs | 2.37 GB/s | 2.62 GB/s | 2.66 GB/s |

  The same as the V100 + RTX 3090 in that slot: the slot is the ceiling.
- **But the default ordering is not safe on this card.** The back-to-back test with every all-reduce card to card
  (`stale_test`, fresh process each run, 3000 per size): with `release`, **wrong values twice in about 50 runs**, 64
  and 1,152 elements, each time at 64 KiB, the first all-reduces of a fresh connection, in the 750 Ti's results; with
  **`readback`, 0 wrong in 48 runs** (over 480,000 all-reduces, 59 billion elements checked). The cause is not known
  (the 750 Ti's writes through its window seen out of order?); the V100 + RTX 3090 never showed it (above). On a
  Maxwell card, set `NCCL_DRMSHARE_ORDER=readback` (with `NCCL_DRMSHARE_SPLIT_BYTES=4194304`, step 3).
- **llama.cpp's CUDA backend runs across the pair** (llama.cpp-drmshare, `-sm tensor`): Qwen3-0.6B perplexity identical
  through the fork and through stock NCCL, chunk for chunk; Qwen3.8-27B UD-Q3_K_XL split 1:23 with MTP and DFlash2
  drafting, every answer identical through the fork and through stock NCCL. Use `-ub 512`: at 2048 the 750 Ti's 2 GB
  runs out. It is slow, though: the 750 Ti's compute holds every step back.

  | Qwen3.8-27B UD-Q3_K_XL, tokens/s | RTX 3090 alone | 750 Ti + 3090, NCCL | 750 Ti + 3090, this fork |
  |---|---|---|---|
  | llama-bench, prompt 512 / generation | 1,533 / 47.2 | 260 / 16.7 | 280 / 16.7 |
  | llama-server, MTP 3 | 79.3 | 19.0 | 19.0 |
  | llama-server, DFlash2 draft | 98.3 | 22.5 | 22.5 |

  A coverage result: a Maxwell card works with the fork, it does not make a 3090 faster.

## How it works

- **The transport** (`src/transport/drmshare.cc`) follows NCCL's P2P transport in its copy-engine mode. The receiving
  card's buffer is Vulkan memory exported twice: as a dma-buf, which the sending card's Vulkan imports through
  DRMShare, and as a file descriptor that the receiving card's CUDA maps for NCCL's kernel. The sending kernel writes
  each step into a local buffer; NCCL's proxy thread copies the step across with a copy engine, waits for its fence,
  and then raises the "data ready" mark that the receiving kernel polls.
- **Ordering.** `release` (default): every card pushes, and each copy ends with a Vulkan queue-family release to the
  other device. `readback`: after each copy, 4 bytes are read back from the destination (a PCIe read cannot pass the
  writes before it); a card in `NCCL_DRMSHARE_NOREAD` fetches the data itself instead, so it is never read across.
  Each card has two copy queues, so pushing and fetching run side by side.
- **Simple protocol only.** NCCL's LL and LL128 protocols put their flags inside the data, which needs writes to land
  in order; a copy engine does not promise that. The small lane, which uses LL, stays on NCCL's shared memory.
- **Two lanes** (`NCCL_DRMSHARE_SPLIT_BYTES`): each communicator gets a companion over the same GPUs that may use only
  NCCL's P2P and shared-memory transports; `ncclAllReduce` sends a collective below the cut to the companion.
- **No CUDA call in the proxy's copy loop**: a CUDA call there can wait on a driver lock held by the program's own
  thread while it loads a CUDA module, while NCCL's kernel waits for the proxy: a hang.
- **Why not real P2P?** CUDA cannot be given a pointer into the other card's memory on this pair: no peer access
  between two generations, CUDA's dma-buf import exists only on Tegra Thor, and NVIDIA's closed 580 driver refuses to
  register another card's memory window as I/O memory. Vulkan, through DRMShare, can reach it: hence copy engines.

## Limits

- **One process only.** Programs with one process per GPU (PyTorch's DDP, vLLM, most NCCL launchers) get no `via DRM`
  connection.
- **Two pairs on NVIDIA's closed 580**: a Tesla V100 + RTX 3090 (one evening) and a GTX 750 Ti + RTX 3090 (one morning,
  where `release` failed and `readback` held: [Maxwell](#maxwell-a-gtx-750-ti-with-an-rtx-3090)). Turing and newer pairs on 615
  should work the same way (same Vulkan calls) but have not been run; more than two cards have not been run.
- **Outside the Vulkan specification**: NVIDIA's Vulkan driver lists a dma-buf and an opaque file descriptor as not
  compatible for one allocation; the transport exports one allocation as both, which works on 580 (every result
  checked) but could stop working in another driver version.
- **Copy-engine latency**: a copy across takes 28-45 µs to start and finish, so transfers below about 2 MiB are faster
  through host RAM; that is why the two lanes exist.
- The rest of DRMShare's limits apply: the board must carry peer traffic, and a V100 in a chipset slot must not be read
  across.
