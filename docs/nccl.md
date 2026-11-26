# NCCL with memory sharing: an NCCL fork for pairs CUDA cannot connect

> **EXPERIMENTAL.** On a Tesla V100 + RTX 3090 this fork moves large
> all-reduces **44 % faster** than NCCL does through host RAM and makes plain llama.cpp **29 % faster** on prompts. Most
> of llama.cpp's gain on this pair comes from llama.cpp itself, on stock NCCL (the
> [P2P guide's llama.cpp section](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus#llamacpp-build-it-with-nccl)):
> on top of those changes the fork adds **5 %** (974 -> 1,017-1,028 tokens/s). Tested on one pair for one evening;
> results below.

NVIDIA's NCCL moves data between GPUs for CUDA programs. Between two cards that CUDA refuses to pair, like a Tesla V100
and an RTX 3090 (two GPU generations: no peer access), NCCL has no direct route and goes through host RAM. This page
builds **a fork of NCCL 2.32.6 with one more transport**: it moves NCCL's data card to card through DRMShare's shared
memory, with the cards' copy engines (through Vulkan), while CUDA still sees no peer access. Programs need no change:
the fork is a drop-in `libnccl.so.2`.

## Results

Tesla V100 (PCIe 3.0 x4, a slot wired through the board's chipset) + RTX 3090 (x16), NVIDIA's closed driver
580.178.04 with DRMShare's `nvidia-drm`, NCCL 2.32.6, 2026-11-26. Every NCCL result checked; every llama.cpp
comparison with identical perplexity, chunk for chunk.

NCCL's `all_reduce_perf`, two cards:

| | 8 B | 2 MiB | 8 MiB | 128 MiB |
|---|---|---|---|---|
| NCCL (shared memory in host RAM) | 13.6 µs | 1.75 GB/s | 1.80 GB/s | 1.82 GB/s |
| this fork, `NCCL_DRMSHARE_ORDER=readback` | 13.0 µs | 1.52 GB/s | 1.94 GB/s | 1.94 GB/s |
| **this fork (default ordering, `release`)** | 13.0 µs | **2.18 GB/s** | **2.60 GB/s** | **2.63 GB/s** |

The ceiling is the V100's x4 slot with both directions busy, about 2.8 GB/s. Small all-reduces go through NCCL's own
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
  ([volta-dmabuf.md](volta-dmabuf.md) for a V100 or Pascal card, [dmabuf-615.md](dmabuf-615.md) for Turing and newer).
  Check: `cat /sys/module/nvidia_drm/parameters/peer_vidmem_dmabuf` prints `Y`.
- **Vulkan on both cards**: the Vulkan loader and NVIDIA's Vulkan driver (`vulkaninfo --summary` lists both GPUs), and
  the headers to build: `sudo apt install libvulkan-dev`.
- **A CUDA toolkit that builds for both cards.** A Tesla V100 (sm_70) needs CUDA 12.x: CUDA 13 dropped Volta. Ubuntu's
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

Put your cards' compute capabilities in `NVCC_GENCODE` (7.0 for a V100, 7.5 Turing, 8.6 RTX 30, 8.9 RTX 40). With
Ubuntu's toolkit instead of NVIDIA's (`/usr/local/cuda`), add `CUDA_HOME=/usr CUDA_LIB=/usr/lib/x86_64-linux-gnu
CUDA_INC=/usr/include`, and `CUDARTLIB=cudart` if `/usr/lib/x86_64-linux-gnu/libcudart_static.a` is missing. To build
only what llama.cpp uses (much faster), add `ONLY_FUNCS="AllReduce Sum f32 * *|AllReduce Sum bf16 * *|SendRecv"`.
The library lands in `build/lib/libnccl.so.2.32.6`.

## 2. Build llama.cpp with NCCL

Any llama.cpp with the CUDA backend's NCCL support builds against the fork as against NCCL itself (same ABI), for
example the copy pinned here as `tools/llama.cpp`:

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
| `NCCL_DRMSHARE_ORDER` | `release` (the default): every card writes its own data into the other, each copy followed by Vulkan's release of that memory to the other device; reads nothing across; 98 % of the raw speed; passed the stale-step test below. `readback`: the first fixed version, slower (see the tables) |
| `NCCL_DRMSHARE_NOREAD=<bus id>` | `readback` only: a card whose memory must not be read by the other card fetches its data itself. Set it for a **Tesla V100 in a chipset slot**, which can freeze when another card reads its memory ([troubleshooting](troubleshooting.md#a-tesla-v100-in-a-chipset-slot-freezes-xid-62-while-another-card-reads-its-memory)); `release` never reads across anyway. The bus id from `nvidia-smi --query-gpu=name,pci.bus_id --format=csv`, any unique part of it (`04:00`) |

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
- **One pair, one evening of tests**: a Tesla V100 + RTX 3090 on NVIDIA's closed 580. Turing and newer pairs on 615
  should work the same way (same Vulkan calls) but have not been run; more than two cards have not been run.
- **Outside the Vulkan specification**: NVIDIA's Vulkan driver lists a dma-buf and an opaque file descriptor as not
  compatible for one allocation; the transport exports one allocation as both, which works on 580 (every result
  checked) but could stop working in another driver version.
- **Copy-engine latency**: a copy across takes 28-45 µs to start and finish, so transfers below about 2 MiB are faster
  through host RAM; that is why the two lanes exist.
- The rest of DRMShare's limits apply: the board must carry peer traffic, and a V100 in a chipset slot must not be read
  across.
