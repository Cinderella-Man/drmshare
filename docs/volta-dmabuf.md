# Volta + Ampere: memory sharing between a Tesla V100 and an RTX 3090, experimental

A Tesla V100 cannot run on this guide's patched driver, and NVIDIA's own driver refuses P2P between a V100 and a GeForce
card (`nvidia-smi topo -p2p` says `NS`). This page shares video memory between the two cards instead: one GPU's memory
is imported into the other through a Linux dma-buf, and each card copies straight into the other's memory. It runs on
NVIDIA's own closed 580 driver without touching any NVIDIA binary: one opt-in option in `nvidia-drm` (which NVIDIA
ships as source) does it. This is not CUDA P2P; [the next section](#what-this-is-and-what-it-is-not) spells out the
difference.

| | Through host RAM (CUDA today) | **dma-buf route** | Real CUDA P2P, RTX 3090 pair ([benchmarks.md](benchmarks.md)) |
|---|---|---|---|
| Copy one way, 3090 -> V100 / V100 -> 3090 | 2.50 / 2.79 GB/s | **2.96 / 3.26 GB/s** | 2.96 / 3.27 GB/s |
| Both ways at once (writes) | | **5.62 GB/s** | 5.52 GB/s |
| A small (4-byte) copy | 17.0 / 19.5 µs | **1.0-2.4 / 2.1-3.1 µs** | 1.06 / 1.90 µs |
| Byte-exact checks | | 20 runs x 1 GiB, 0 wrong words | |

Same two slots in all three columns, so the same PCIe ceiling: the V100's chipset slot (PCIe 3.0 x4). The route is
reachable from **Vulkan** only. CUDA programs (NCCL, PyTorch, vLLM) cannot use it. llama.cpp can: with a second small
change its tensor-parallel mode exchanges each layer's results straight between the two cards
([V6](#v6-llamacpp-tensor-parallel-with-an-all-reduce-through-shared-video-memory)).

**Experimental.** Verified on 2026-11-24 on two machines, one driver version (580.178.04). Run the checks on your own
cards. The changed module is loaded by hand and gone after a reboot.

## What this is, and what it is not

**It is memory sharing.** Vulkan on one GPU allocates video memory and exports it as a dma-buf, Linux's standard way
to hand a buffer from one device to another. Vulkan on the other GPU imports it, and that GPU's copy engine then
writes into (or reads from) the first card's memory over PCIe, without a stop in system RAM.

**It is not CUDA P2P**, and nothing on this page turns CUDA P2P on:

- `nvidia-smi topo -p2p r` still says `NS` for the pair, and `cudaDeviceCanAccessPeer` still says 0.
- CUDA, NCCL, PyTorch and vLLM do not see the shared memory and keep going through system RAM. Only Vulkan programs
  written for it use it: this page's test program and the changed llama.cpp in [V6](#v6-llamacpp-tensor-parallel-with-an-all-reduce-through-shared-video-memory).
- It is not the GeForce P2P patch from the rest of this guide either: different driver (NVIDIA's closed 580), different
  mechanism.

**Why the numbers sit next to CUDA P2P's.** Underneath, the transfer is the same kind: one GPU's DMA engine writing
into the other card's memory window (BAR1) over PCIe. So the speeds are comparable, and the tables compare them. That
comparison is all it is.

**Why some names still say P2P.** Code and log names use the word because of that hardware path: the llama.cpp
variable `GGML_VK_ALLREDUCE_P2P`, its log lines, and the module option's description ("P2P through BAR1"). They are
quoted exactly as they are, so you can match them; the text around them says memory sharing.

## Why not the guide's driver

The guide's driver is built on NVIDIA's open kernel modules, which need the GSP, a processor that came with Turing.
NVIDIA's own README: the open modules "can be used on any Turing or later GPU". A V100 (Volta) has none, and the open
source has no Volta chip table at all (`halmgrGetHalForGpu()` finds nothing, the probe prints "does not include the
required GPU System Processor"). The part of the driver that drives the hardware runs as closed firmware on the GSP;
the open half calls it through 222 different remote procedure calls. Adding Volta would mean rewriting that firmware's
work for Volta on the CPU, and NVIDIA never published the Volta power-management firmware, so such a port would likely
stay at the card's start-up clocks. NVIDIA's CUDA library also dropped Volta after branch 580. Not realistic.

NVIDIA's closed 580 driver runs both cards. Its core is a binary, but two parts of it are source: the interface layer
of `nvidia.ko` and the whole of `nvidia-drm`. That is enough.

## What the closed 580 driver does with the pair

| Path | Stock 580 |
|---|---|
| `nvidia-smi topo -p2p r` / `w` | `NS` both |
| [Route 2](mixed-generations.md#route-2-io-memory-without-cuda-peer-access) (`p2p_iomem`: register the other GPU's memory window as I/O memory) | refused, `invalid argument`. The check sits in the binary core, with no switch |
| CUDA: export GPU memory as a dma-buf | not supported (`CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` is 0; the closed package's `nv-dmabuf.c` is a stub) |
| Vulkan: export video memory as a dma-buf | works (`VK_EXT_external_memory_dma_buf`) |
| Vulkan: import that dma-buf on the other NVIDIA GPU | fails: no common memory type. The kernel log shows `__nv_drm_gem_nvkms_prime_dup ... Failed to import NVKMS memory to GEM object` |

The last row is the whole problem, and it is in `nvidia-drm`:

- For a buffer from another NVIDIA GPU, `nvidia-drm` insists on an internal duplicate step (NVKMS `dupMemory()`)
  that fails across GPUs, and never tries Linux's standard dma-buf attach.
- Its standard path could not have served it either: it describes a buffer as a list of system-RAM pages, and video
  memory has none.

On the receiving side the closed core needs no change. `nvidia-drm` hands it the attached buffer as a list of device
addresses (`getSystemMemoryHandleFromSgt`), and its open interface code already takes a list without pages as it is:
`nv_register_sgt()` "Populate[s] phys addrs with DMA addrs from SGT", and `nv_dma_map_sgt()` "use[s] that directly".
The I/O-memory check that stops route 2 is not on this path.

## The change: one opt-in option in nvidia-drm

One commit, [`9904ca0`](https://github.com/Cinderella-Man/open-gpu-kernel-modules/commit/9904ca03caa2d1e133715fd902e503961d000eaa), on top of NVIDIA's own `580.178.04` tag in a fork of NVIDIA's open driver
(branch `580-peer-vidmem-dmabuf`, pinned in this repository as `tools/open-gpu-kernel-modules-580`). It changes three
files of `nvidia-drm` (127 lines) and adds the module option `peer_vidmem_dmabuf` (default off):

- **Export:** a video-memory buffer gets its own dma-buf handlers. Each importer gets a one-entry address list without
  pages, pointing at the buffer's place in this GPU's memory window (BAR1; NVKMS already maps it there for CPU
  access), mapped for the importing device with `dma_map_resource()`.
- **Import:** such a buffer from another GPU goes through the standard dma-buf attach instead of `dupMemory()`.

With the option off, the module behaves exactly like the stock one (checked: same refusal, same log line).

The technique is not new on Linux. AMD's `amdgpu` has exported video memory to peers this way since about 2020
(Christian König's dma-buf series, which the kernel names peer-to-peer), and the kernel has a `peer2peer` attach flag
for importers that can handle
memory without pages. This change applies it to `nvidia-drm`. A cleaner version should honour that flag, and hand the
address list only to importers that set it.

## Test rigs

| | Two V100s | V100 + RTX 3090 (this guide's test rig) |
|---|---|---|
| Board, CPU | ASUS TUF GAMING B450-PLUS II, Ryzen 7 5700X | B550 board, Ryzen 9 3900X |
| GPU in the CPU slot | Tesla V100-PCIE-32GB, PCIe 3.0 x16 | RTX 3090 (GA102), PCIe 4.0 x16, 32 GB window (Resizable BAR) |
| GPU in the chipset slot | Tesla V100-PCIE-32GB, **PCIe 2.0 x4** | Tesla V100-PCIE-32GB, PCIe 3.0 x4 |
| Memory windows | 32 GB each, from the factory | 32 GB each |
| Software | Ubuntu 24.04.5, kernel 7.0.0-43, `nvidia-driver-580` 580.178.04 (Ubuntu's prebuilt modules), CUDA 12.9 | Ubuntu 26.04.1, kernel 7.0.0-44, `nvidia-driver-580` 580.178.04 (DKMS), Ubuntu's CUDA 12.4 |

Both run `iommu=pt` ([procedure.md step 4](procedure.md#4-iommu-in-passthrough-mode)). A V100 PCIe has no fan: give
it airflow straight through its fins. It takes an 8-pin CPU (EPS) power plug, not a PCIe one.

## V1. Switch to NVIDIA's closed 580 driver

Coming from this guide's driver 615: note what is installed, release the holds, remove the 615 packages, install 580
with its kernel source, hold it, reboot. The same switch as for Pascal
([pascal-and-volta.md](pascal-and-volta.md#the-software-path-on-ubuntu-2604-verified)). There is also a
`ForceP2P` section on that page; this route does not need it.

```bash
dpkg -l | awk '/^(ii|hi)/ && /nvidia/ {print $1, $2, $3}' > ~/nvidia-packages-before-580.txt
apt-mark showhold > ~/holds-before-580.txt
sudo apt-mark unhold $(cat ~/holds-before-580.txt)
sudo apt purge $(dpkg -l | awk '/^(ii|hi)/ && /-615/ {print $2}')
sudo apt install nvidia-driver-580 nvidia-kernel-source-580
sudo apt-mark hold $(dpkg -l | awk '/^(ii|hi)/ && /-580/ {print $2}')
sudo reboot
```

Check: `nvidia-smi` lists both cards with driver 580.178.04, `nvidia-smi topo -p2p r` says `NS` for the pair, and
`/usr/src/nvidia-580.178.04` exists.

## V2. Build nvidia-drm with the option

Take a copy of the installed driver's source and put the three changed files of the pinned commit into it. NVIDIA's
open `580.178.04` `nvidia-drm` is the closed package's, file for file, except one line (`MODULE_LICENSE`, "Dual
MIT/GPL" against "MIT"), and the three files are not that one. The build links NVIDIA's binary parts exactly as
shipped, as a normal driver install does; only `nvidia-drm.ko` is used.

```bash
cd ~/how-to-enable-p2p-on-nvidia-gpus
git submodule update --init --depth 1 tools/open-gpu-kernel-modules-580
cp -a /usr/src/nvidia-580.178.04 ~/nv580-src
cp tools/open-gpu-kernel-modules-580/kernel-open/nvidia-drm/{nvidia-drm-gem.c,nvidia-drm-gem-nvkms-memory.c,nvidia-drm-gem-nvkms-memory.h} ~/nv580-src/nvidia-drm/
make -C ~/nv580-src -j"$(nproc)" SYSSRC=/lib/modules/$(uname -r)/build modules
modinfo -F parm ~/nv580-src/nvidia-drm.ko | grep peer
```

Expected: about 30 seconds of build, and
`peer_vidmem_dmabuf:Export video memory as dma-bufs another GPU can import (P2P through BAR1) (bool)`.

## V3. Load it (until the next reboot)

The display manager holds `nvidia-drm`, so stop it first. While experimenting, let the machine reboot itself after a
kernel crash instead of hanging (also only until the next reboot).

```bash
sudo sysctl kernel.panic=10 kernel.panic_on_oops=1
sudo systemctl stop display-manager
sudo rmmod nvidia_drm
sudo insmod ~/nv580-src/nvidia-drm.ko modeset=1 fbdev=1 peer_vidmem_dmabuf=1
cat /sys/module/nvidia_drm/parameters/peer_vidmem_dmabuf
```

Expected: `Y`. Undo: reboot (the stock module loads again).

## V4. Test: each card writes and reads the other's memory

[`tools/p2p_dmabuf.cpp`](../tools/p2p_dmabuf.cpp): Vulkan on GPU A allocates video memory and exports it as a dma-buf,
Vulkan on GPU B imports it, B's copy engine writes into it and reads it back, and every word is checked on the other
card. GPU numbers follow the PCI bus order, as `nvidia-smi` counts them; the first number owns the memory.

```bash
sudo apt install -y libvulkan-dev
cd ~/how-to-enable-p2p-on-nvidia-gpus/tools
g++ -O2 -std=c++17 p2p_dmabuf.cpp -o p2p_dmabuf -lvulkan
./p2p_dmabuf 0 1 64
./p2p_dmabuf 1 0 64
```

On the test rig (run as root), the V100 as GPU 0:

```
A = Tesla V100-PCIE-32GB (04:00), B = NVIDIA GeForce RTX 3090 (2b:00)
A: 64 MiB in memory type 7 (DEVICE_LOCAL ), heap 0
dma-buf in the kernel: 67108864  ...  nvidia_drm  ...
A fills its exported buffer:     823.3 GB/s
B: imported A's memory
B copies into A:     2.96 GB/s, checked on A: OK (0 wrong words)
B reads from A:      1.14 GB/s, checked on B: OK (0 wrong words)
4-byte copy B -> A:   1.04 us
ALL OK
```

The last line must be `ALL OK`. The exporter must say `nvidia_drm`: plain `drm` means the option is off, and the run
stops at `B cannot import A's dma-buf: no common memory type`. "A fills its exported buffer" at hundreds of GB/s shows
the buffer is video memory, not system RAM.

Both directions at once, writes only, from a shared start:

```bash
B=$(mktemp -d)
NO_READ=1 REPS=20 BARRIER=$B ./p2p_dmabuf 0 1 1024 > $B/a.log &
NO_READ=1 REPS=20 BARRIER=$B ./p2p_dmabuf 1 0 1024 > $B/b.log &
until [ -e $B/ready.01 ] && [ -e $B/ready.10 ]; do sleep 0.05; done; touch $B/go; wait
grep -h "copies into" $B/a.log $B/b.log
```

Leave heavy reads in both directions at once alone: on the Turing rig that reset the machine
([troubleshooting](troubleshooting.md#the-machine-resets-data-fabric-sync-flood)).

For the "through host RAM" column, [`tools/staged_baseline.cu`](../tools/staged_baseline.cu) times CUDA's own
`cudaMemcpyPeer`, which goes through host RAM for this pair. CUDA 13 cannot build for Volta, so use Ubuntu's
`nvidia-cuda-toolkit` (12.4):

```bash
/usr/bin/nvcc -O2 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_86,code=sm_86 staged_baseline.cu -o staged_baseline
CUDA_DEVICE_ORDER=PCI_BUS_ID ./staged_baseline 0 1 64
```

## V5. Results

V100 + RTX 3090, 64 MiB, every word checked:

| | Copies into the other card | Reads from it | 4-byte copy |
|---|---|---|---|
| V100's memory, 3090's copy engine | 2.96 GB/s | 1.14 GB/s | 1.04 µs |
| 3090's memory, V100's copy engine | 3.27 GB/s | 2.93 GB/s | 2.08 µs |
| Both ways at once (1 GiB, writes) | 2.76 + 2.86 = **5.62 GB/s** | | |

Soak, 20 runs at 1 GiB, one direction at a time: 20/20 `ALL OK`, 0 wrong words, nothing in the kernel log. Medians:
writes 2.96 / 3.26 GB/s, reads 1.14 / 1.38 GB/s, 4-byte copy 2.39 / 3.11 µs. Writes hold their speed at 1 GiB, reads of
the 3090's memory drop (2.93 GB/s at 64 MiB): write across, do not read across.

Two V100s on the second machine: 1.67 / 1.63 GB/s writes, 0.91 / 1.12 GB/s reads, 20/20 soak runs byte-exact. NVIDIA's
native CUDA P2P between the same two V100s: 1.62 / 1.67 GB/s. Both are capped by that board's PCIe 2.0 x4 chipset slot.

## V6. llama.cpp: tensor-parallel with an all-reduce through shared video memory

llama.cpp splits a model across cards by layers (`-sm layer`) or, since recently, tensor-parallel (`-sm tensor`):
every layer is split across both cards, which add up their partial results (an all-reduce) after each one. Its Vulkan
backend does that all-reduce through host RAM: each card writes its part into a buffer in system memory and the other
reads it back over PCIe.

One commit, [`ab88444`](https://github.com/Cinderella-Man/miner-llama/commit/ab884449db7e416625f001d901155aff13d14ff9), on top of upstream's `0ce7f213` (branch `vk-allreduce-shared-vram`, pinned in this
repository as `tools/llama.cpp`; 233 lines in five files of `ggml/src/ggml-vulkan`), adds an opt-in
`GGML_VK_ALLREDUCE_P2P=1`: each card's receive buffer lives in its own video memory, shared with the other card through
the dma-buf route, and each card writes its part straight into the other's memory (one PCIe trip instead of two), then
adds locally. Same synchronisation and slot-reuse rules as upstream, so the arithmetic is identical. It needs V3's
option.

```bash
sudo apt install -y glslc spirv-headers glslang-tools libvulkan-dev cmake build-essential
cd ~/how-to-enable-p2p-on-nvidia-gpus
git submodule update --init --depth 1 tools/llama.cpp
cd tools/llama.cpp
cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build-vk -j6 --target llama-bench llama-perplexity llama-server llama-cli
```

Expected: a build of about 4 minutes (`-j6` because the test rig has 8 GB of RAM). Then:

```bash
GGML_VK_ALLREDUCE_P2P=1 ./build-vk/bin/llama-bench -m model.gguf -ngl 99 -sm tensor -p 512,2048 -n 128 -r 5
```

To see which all-reduce ran, add `-v` to `llama-perplexity`: the shared-memory run logs
`ggml_vulkan: all-reduce writes straight into peer VRAM (dma-buf P2P)` and `P2P all-reduce: ... receive buffer in each
device's VRAM`. Without the variable it is upstream's host-RAM all-reduce. Note: on this pair llama.cpp numbers the
RTX 3090 as `Vulkan0` and the V100 as `Vulkan1`, and the V100 has no matrix cores under Vulkan
(`matrix cores: none`), so it is the slow card in this backend.

Correctness, Qwen3-0.6B Q8_0, perplexity over 16 chunks of 512 tokens of this guide's own text:

| | Perplexity |
|---|---|
| layer split | 18.3425 +/- 0.95615 (different arithmetic) |
| tensor-parallel, upstream all-reduce | **18.3614 +/- 0.95733** |
| tensor-parallel, patched build, variable unset | **18.3614 +/- 0.95733** |
| tensor-parallel, all-reduce through shared video memory | **18.3614 +/- 0.95733** |

Speed, `llama-bench`, 5 repetitions, tokens per second:

| Qwen3-0.6B Q8_0 | Prompt, 512 | Prompt, 2048 | Generation, 128 |
|---|---|---|---|
| layer split | 12216 | 12547 | 318 |
| tensor-parallel, upstream all-reduce | 4115 | 3787 | 168 |
| tensor-parallel, all-reduce through shared video memory | 5048 (**+23 %**) | 4562 (**+20 %**) | 175 (**+4 %**) |

The all-reduce through shared video memory makes tensor-parallel 20-23 % faster on prompts. On a model this small, splitting by layer is still
faster than either: each layer is so little work that exchanging results after every layer dominates.

**Qwen3.8-27B Q8_0** ([unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q8_0.gguf),
29 GB, sha256 `a680f44a...`), split across the V100 and the RTX 3090. Perplexity over 8 chunks of 512 tokens: upstream
all-reduce and all-reduce through shared video memory both give **5.9896 +/- 0.32753**; the log shows a 40 MiB receive
buffer in each card. Speed, `llama-bench`, 3 repetitions, tokens per second:

| Qwen3.8-27B Q8_0 | Prompt, 512 | Prompt, 2048 | Generation, 128 |
|---|---|---|---|
| layer split | 433 +/- 48 | **504** | 23.3 |
| tensor-parallel, upstream all-reduce | 347 | 347 | 27.9 |
| tensor-parallel, all-reduce through shared video memory | 410 (**+18 %**) | 412 (**+18 %**) | **28.9** (+3.5 %) |

On the big model tensor-parallel pays off for generation: 27.9 tokens per second against 23.3 split by layer, which is
upstream's tensor parallelism at work. The all-reduce through shared video memory adds 3.5 % on top (28.9, the fastest
of the three for generation) and 18 % on prompts. Split by layer still reads a long prompt fastest.

## On a board that does not carry peer traffic

The change removes NVIDIA's software refusals, not the hardware's: the data still travels from one card to the other
over PCIe, and the board has to route it. Tested on the P106-090 mining board from
[pascal-and-volta.md](pascal-and-volta.md#the-test-on-a-mining-box-the-driver-said-yes-the-hardware-said-no): Celeron
3865U (Intel Sunrise Point-LP chipset), nine P106-090 behind two ASMedia ASM1184e switches, IOMMU translated, the same
driver 580.178.04, with `nvidia-drm` built from [V2](#v2-build-nvidia-drm-with-the-option)'s steps. Writes only
(`NO_READ=1`), 64 MiB, both directions:

| Path between the two cards | Import | Words that arrived |
|---|---|---|
| Same ASM1184e switch | works | **0 of 16,777,216** |
| Two different switches | works | **0 of 16,777,216** |
| Chipset root port and a card behind a switch | works | **0 of 16,777,216** |

The software side works on Pascal too: the stock module refused the import, the changed one accepted it. Then the copy
engine "finished" (0.21 GB/s) and nothing arrived, on every path, with nothing in the kernel log: no IOMMU fault, no
PCIe error, no GPU error. So run [V4](#v4-test-each-card-writes-and-reads-the-others-memory)'s test, which checks every
word, on every new board before trusting it. What is known per board: [boards.md](boards.md).

## Limits

- **Needs a board that carries peer traffic**, like CUDA P2P. On a mining board with PCIe switches nothing arrived,
  and nothing said so ([above](#on-a-board-that-does-not-carry-peer-traffic)).
- **Vulkan only.** CUDA on the closed 580 driver cannot import a dma-buf, and memory Vulkan imported this way cannot
  be handed on to CUDA. NCCL, PyTorch and vLLM do not see this route.
- **Loaded by hand.** `nvidia-drm` with the option is not installed; a reboot returns to the stock module. A boot install
  was not tested.
- **The display manager has to be stopped** to swap the module.
- **Write across, do not read across** in bulk: reads of the other card's memory are slower and drop at large sizes,
  and heavy reads in both directions at once reset the Turing rig.
- **Opt-in at two levels:** the module option and, for llama.cpp, the environment variable. Both default to off.

## Undo

- Reboot: the stock `nvidia-drm` loads again, and the crash settings of V3 are gone.
- Back to this guide's driver 615: remove the 580 packages (`apt-mark unhold`, `apt purge`) and install the 615
  packages again as in [procedure.md step 3](procedure.md#3-nvidia-driver-6157109-open-kernel-modules-from-the-ubuntu-archive);
  `~/nvidia-packages-before-580.txt` lists what was there.

## NVIDIA's ForceP2P key: the kernel says yes, CUDA still says no

NVIDIA's driver reads a documented registry key, `ForceP2P` ([pascal-and-volta.md](pascal-and-volta.md#forcep2p-overriding-the-chipset-check)).
In 580's source (`p2p_caps.c`) it is checked before the chipset, topology and same-architecture tests. Loaded with
`NVreg_RegistryDwords="ForceP2P=0x11"` on the V100 + RTX 3090 (read-only check, no data moved):

| | Stock 580 | `ForceP2P=0x11` |
|---|---|---|
| `nvidia-smi topo -p2p r` / `w` | `NS` / `NS` | **`OK` / `OK`** |
| `nvidia-smi topo -p2p a` (atomics) | `NS` | `NS` |
| `cudaDeviceCanAccessPeer` ([`tools/p2p_access.cu`](../tools/p2p_access.cu)), both ways | NO | **NO** |

The kernel driver accepts the pair, CUDA does not: `libcuda` has its own architecture check, the one that also
stops the RTX 2060 + RTX 3060 ([mixed-generations.md](mixed-generations.md#why-cuda-says-no)). CUDA P2P for this
pair would also need that check patched (as route 1 there does, on driver 615), and a test that forced P2P between a
data-center card and a GeForce card actually moves data correctly. Neither was tried. The memory sharing on this page
needs neither.

## Not tested yet

- A boot install of the changed `nvidia-drm`.
- Other GPU pairs on a board that carries peer traffic: any two NVIDIA GPUs with Vulkan on driver 580 should behave the
  same. Data crossed between two V100s and between a V100 and an RTX 3090; two P106-090s (Pascal) imported each
  other's memory, but on a board that carried nothing.

## Credits

- **NVIDIA:** `nvidia-drm` and the interface layer of `nvidia.ko` ship as source, MIT-licensed; the interface code
  already handled imported address lists without pages.
- **Christian König and the amdgpu developers:** the dma-buf technique for sharing video memory between devices.
- **ggml-org/llama.cpp:** tensor parallelism, and the Vulkan all-reduce this builds on
  ([PR #25051](https://github.com/ggml-org/llama.cpp/pull/25051) by pwilkin, merged 2026-11-02, host-memory transport).
- **TheBlueMatt:** the first direct GPU-to-GPU Vulkan all-reduce for llama.cpp, on Intel Xe, in a fork
  ([commit 5eb904a](https://github.com/TheBlueMatt/llama.cpp/commit/5eb904aaad6cc865bed2b30aad2559ef90558563),
  [issue #22648](https://github.com/ggml-org/llama.cpp/issues/22648)); not merged. It imported memory with Vulkan's
  external-memory fds across devices, which he later flagged against a Vulkan validity rule; this page's version imports
  through the dma-buf handle type, which Vulkan defines for sharing between devices.
- [volta-forever](https://github.com/nbald/volta-forever) (nbald) was checked: a user-space layer that runs CUDA 13
  programs on a V100 under driver 580. It does not share memory between GPUs.
