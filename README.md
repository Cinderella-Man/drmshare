# Hidden GEM: share video memory between NVIDIA GPUs of different generations

One opt-in option in `nvidia-drm`, the part of NVIDIA's Linux driver that ships as source, lets **Vulkan** on one GPU
import another GPU's video memory through a Linux **dma-buf**. Each card then copies straight into the other's memory
over PCIe, instead of through host RAM. It works on pairs CUDA refuses, because they are of different generations: a
Tesla V100 with an RTX 3090, an RTX 2060 or a GTX 1070 Ti, and an RTX 2060 with an RTX 3060. It runs on NVIDIA's own
drivers 580, 595 and 615, without a patched driver and without a patched `libcuda`.

The name: the buffers it shares are `nvidia-drm`'s **GEM** objects, and the ability was already there, hidden in
NVIDIA's own open-source code. The change adds 110 lines (580) or 144 lines (615) to three `nvidia-drm` files, and the
option is off by default.

**This is memory sharing, not CUDA P2P.** Only Vulkan programs can use it; CUDA, NCCL, PyTorch and vLLM see no peer
access, as before. llama.cpp can use it: with a second small change, its tensor-parallel mode adds up each layer's
results through the shared memory ([what this is, and what it is not](docs/volta-dmabuf.md#what-this-is-and-what-it-is-not)).

**Use CUDA P2P where you can.** If your pair has a P2P route in the companion guide,
[how-to-enable-p2p-on-nvidia-gpus](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus), use that: it works for every CUDA program (NCCL, PyTorch, vLLM).
Hidden GEM is for the cases where P2P is not possible or does not fit:

- a **Tesla V100 or a Pascal card with a card of another generation**: these cards run only on NVIDIA's closed driver
  580 (the guide's patched driver cannot run them), and CUDA refuses peer access between two architectures;
- **two generations of Turing or newer on NVIDIA's stock driver** (the guide's routes for them need its patched driver,
  plus a patched copy of `libcuda` or root);
- **programs that use Vulkan rather than CUDA**, like llama.cpp's Vulkan backend.

The guide keeps the same list: [When P2P is not possible](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus#when-p2p-is-not-possible-hidden-gem).

**Experimental.** Tested on two machines, three driver versions and five pairs, every word of every copy checked. The
changed module is loaded by hand and gone after a reboot.

## Which page is yours

| Your cards | Driver | Follow | Tested |
|---|---|---|---|
| Tesla V100 + a GeForce card (Volta + Ampere, Turing or Pascal), or two V100s | NVIDIA's driver 580 (the closed package: the open modules do not run Volta) | [docs/volta-dmabuf.md](docs/volta-dmabuf.md) | **works**: V100 + RTX 3090, V100 + RTX 2060, V100 + GTX 1070 Ti; two V100s |
| Two cards of Turing or newer, any generations | Ubuntu's driver 615 (or 595) | [docs/dmabuf-615.md](docs/dmabuf-615.md) | **works**: RTX 2060 + RTX 3060 |
| Another driver version (580 open modules, 590, 610, 620) | | [docs/dmabuf-drivers.md](docs/dmabuf-drivers.md) | 580 open modules **works**; 590, 610, 620: the change applies, not run |
| One model across both cards with llama.cpp | any of the above | [benchmarks/llama](benchmarks/llama/README.md), [volta-dmabuf.md V6](docs/volta-dmabuf.md#v6-llamacpp-tensor-parallel-with-an-all-reduce-through-shared-video-memory) | **works**: Qwen3.8-27B on V100 + RTX 3090 and on RTX 2060 + RTX 3060 |

Three things decide whether it works on your machine:

- **The board must carry peer traffic**, as for CUDA P2P. On a mining board with PCIe switches nothing arrived, and
  nothing said so ([volta-dmabuf.md](docs/volta-dmabuf.md#on-a-board-that-does-not-carry-peer-traffic); what is known
  per board: the P2P guide's [boards page](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/blob/main/docs/boards.md)).
- **A card shares only what fits its memory window** (BAR1). RTX 30 cards and newer show all their memory with Resizable
  BAR on; a Turing card's stock 256 MB window shares buffers up to 224 MiB, and the P2P guide's
  [turing.md](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/blob/main/docs/turing.md) T1-T3
  enlarge it.
- **Programs that map a lot of video memory** (llama.cpp with Resizable BAR) need the driver's static BAR1 off,
  `RMForceStaticBar1=0`, for every card ([troubleshooting](docs/troubleshooting.md)).

## Results

Copies of 1 GiB into the first card's memory / into the second card's, every word checked, against CUDA's
`cudaMemcpyPeer`, which goes through host RAM for every one of these pairs:

| Driver | Cards | Memory sharing | Both ways at once | 4-byte copy | Through host RAM |
|---|---|---|---|---|---|
| 580, closed | Tesla V100 + RTX 3090 | **2.96 / 3.26 GB/s** | **5.62 GB/s** | **2.4 / 3.1 µs** | 2.50 / 2.79 GB/s, 17.0 / 19.5 µs |
| 580, closed | Tesla V100 + RTX 2060 | **2.91 / 3.28 GB/s** | **5.62 GB/s** | **2.2 / 3.2 µs** | 2.27 / 2.51 GB/s, 19.9 / 21.5 µs |
| 580, closed | Tesla V100 + GTX 1070 Ti | **2.96 / 3.28 GB/s** | **5.63 GB/s** | **1.5 / 2.2 µs** | 2.82 / 3.22 GB/s, 18.4 / 20.4 µs |
| 580 open, 595, 615 | RTX 2060 + RTX 3060 | **2.93-2.97 / 3.18-3.28 GB/s** | **5.61-5.64 GB/s** | **1.6-2.2 / 2.7-3.1 µs** | 2.47 / 2.78 GB/s, 13.5 / 17.8 µs |

Both ways at once is 5.6 GB/s for every pair on that board: its x4 chipset slot is the ceiling. Two V100s on another
board reach 1.67 / 1.63 GB/s in a PCIe 2.0 x4 slot, the same as CUDA P2P there.

llama.cpp, tensor parallel, with the all-reduce through shared video memory instead of host RAM:

| Cards, model | Results | Prompt | Generation |
|---|---|---|---|
| V100 + RTX 3090, Qwen3.8-27B Q8_0 | identical (perplexity 5.9896 both ways) | **+18 %** (410 vs 347 tokens/s) | +4 % (28.9 vs 27.9) |
| RTX 2060 + RTX 3060, Qwen3.8-27B UD-Q3_K_XL, drivers 580 / 595 / 615 | identical on each driver | -1.5 to -3.5 % | +0.7 to +1.1 % (19.3 vs 19.2) |

On the RTX 2060 + RTX 3060 the shared all-reduce gives no meaningful gain; tensor parallel itself generates 42 % faster
than layer split there ([benchmarks/llama](benchmarks/llama/README.md)).

## Contents

| | |
|---|---|
| [docs/volta-dmabuf.md](docs/volta-dmabuf.md) | Where it started: a Tesla V100 with an RTX 3090 (also an RTX 2060 and a GTX 1070 Ti) on NVIDIA's driver 580. Why nothing else works for a V100, the change, how to build and load it, tests, the llama.cpp all-reduce (V6), a board that does not carry peer traffic. |
| [docs/dmabuf-615.md](docs/dmabuf-615.md) | The same on Ubuntu's driver 615 for Turing and newer, tested on an RTX 2060 + RTX 3060; llama.cpp on that pair (D4). |
| [docs/dmabuf-drivers.md](docs/dmabuf-drivers.md) | Every NVIDIA branch from 580 to 620: which version of the change it takes, what was tested, all results in one table, llama.cpp on each driver, how to build for another version. |
| [docs/troubleshooting.md](docs/troubleshooting.md) | The kernel log filling with `dmaAllocMapping_GM107` lines (static BAR1), and driver 595's per-card option that misses one card. |
| [benchmarks/llama](benchmarks/llama/README.md) | llama.cpp tensor parallel on the RTX 2060 + RTX 3060: scripts, results on three drivers, and a crash at exit on driver 615. |
| [tools/](tools) | `p2p_dmabuf.cpp` (the Vulkan test: share, copy both ways, check every word), `staged_baseline.cu` (CUDA through host RAM, the baseline), `keep_busy.cu` (keeps a receiving card out of its idle state), `bar1_probe.cpp` (how much host-visible video memory Vulkan can map), `p2p_iomem.cu` (the P2P guide's I/O-memory route, compared in volta-dmabuf.md); the changed `nvidia-drm` for 580 and 615 and the changed llama.cpp as pinned submodules. |
| [docs/assets](docs/assets) | Infographics: V100 + RTX 3090, and four generations on driver 580 (`make_infographic.py`). |

The code lives in two forks, pinned here as submodules:
[Cinderella-Man/open-gpu-kernel-modules](https://github.com/Cinderella-Man/open-gpu-kernel-modules), branches
`580-peer-vidmem-dmabuf` and `615-peer-vidmem-dmabuf` (the `nvidia-drm` option `peer_vidmem_dmabuf`), and
[Cinderella-Man/miner-llama](https://github.com/Cinderella-Man/miner-llama), branch `vk-allreduce-shared-vram` (llama.cpp
with `GGML_VK_ALLREDUCE_P2P=1`). Code, option and log names still say P2P after the hardware path; the text says memory
sharing.

These pages started in the P2P guide and moved here on 2026-11-25, with their history.

## Credits

- Christian König and the [amdgpu](https://sources.debian.org/src/linux/6.19.6-2/drivers/gpu/drm/amd/amdgpu/amdgpu_dma_buf.c)
  developers: the dma-buf technique for sharing video memory between devices.
- [NVIDIA open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules): `nvidia-drm` and the driver's
  kernel interface as source, dual MIT/GPLv2.
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp): tensor parallelism, and the Vulkan all-reduce by pwilkin
  ([PR #25051](https://github.com/ggml-org/llama.cpp/pull/25051)) that the shared-memory all-reduce builds on.
- [TheBlueMatt](https://github.com/TheBlueMatt/llama.cpp/commit/5eb904aaad6cc865bed2b30aad2559ef90558563): the first
  direct GPU-to-GPU Vulkan all-reduce for llama.cpp, on Intel Xe.
- [noon-at-cgn/geforce-bar1-resize](https://github.com/noon-at-cgn/geforce-bar1-resize): the larger memory window of the
  Turing cards.
