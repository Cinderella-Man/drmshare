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
[how-to-enable-p2p-on-nvidia-gpus](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus), use that: it works for every CUDA program (NCCL, PyTorch, vLLM). For llama.cpp on cards that CUDA runs, its CUDA backend built with NCCL
was faster than the Vulkan backend with memory sharing on three RTX 3090s even without P2P
([D5](docs/dmabuf-615.md#d5-three-rtx-3090s)).
Hidden GEM is for the cases where P2P is not possible or does not fit:

- a **Tesla V100 or a Pascal card with a card of another generation**: these cards run only on NVIDIA's closed driver
  580 (the guide's patched driver cannot run them), and CUDA refuses peer access between two architectures;
- **two generations of Turing or newer on NVIDIA's stock driver** (the guide's routes for them need its patched driver,
  plus a patched copy of `libcuda` or root);
- **programs that use Vulkan rather than CUDA**, like llama.cpp's Vulkan backend.

The guide keeps the same list: [When P2P is not possible](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus#when-p2p-is-not-possible-hidden-gem).

**Experimental.** Tested on three machines, three driver versions and six setups (five pairs, and three RTX 3090s
together), every word of every copy checked: about 20 hours of tests so far, and a longer stress run is in progress.
One known limit: a Tesla V100 in a chipset slot can freeze when another card reads its memory while a program starts or
stops on it ([troubleshooting](docs/troubleshooting.md#a-tesla-v100-in-a-chipset-slot-freezes-xid-62-while-another-card-reads-its-memory)). The changed module is loaded by hand and gone after a reboot.

## Which page is yours

| Your cards | Driver | Follow | Tested |
|---|---|---|---|
| Tesla V100 + a GeForce card (Volta + Ampere, Turing or Pascal), or two V100s | NVIDIA's driver 580 (the closed package: the open modules do not run Volta) | [docs/volta-dmabuf.md](docs/volta-dmabuf.md) | **works**: V100 + RTX 3090, V100 + RTX 2060, V100 + GTX 1070 Ti; two V100s |
| Two or more cards of Turing or newer, any generations | Ubuntu's driver 615 (or 595) | [docs/dmabuf-615.md](docs/dmabuf-615.md) | **works**: RTX 2060 + RTX 3060; three RTX 3090s ([D5](docs/dmabuf-615.md#d5-three-rtx-3090s)) |
| Another driver version (580 open modules, 590, 610, 620) | | [docs/dmabuf-drivers.md](docs/dmabuf-drivers.md) | 580 open modules **works**; 590, 610, 620: the change applies, not run |
| One model across both cards with llama.cpp | any of the above | [benchmarks/llama](benchmarks/llama/README.md), [volta-dmabuf.md V6](docs/volta-dmabuf.md#v6-llamacpp-tensor-parallel-with-an-all-reduce-through-shared-video-memory) | **works**: Qwen3.8-27B on V100 + RTX 3090, on RTX 2060 + RTX 3060 and on three RTX 3090s |

Four things decide whether it works on your machine:

- **The board must carry peer traffic**, as for CUDA P2P. On a mining board with PCIe switches nothing arrived, and
  nothing said so ([volta-dmabuf.md](docs/volta-dmabuf.md#on-a-board-that-does-not-carry-peer-traffic); what is known
  per board: the P2P guide's [boards page](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/blob/main/docs/boards.md)).
- **A card shares only what fits its memory window** (BAR1). RTX 30 cards and newer show all their memory with Resizable
  BAR on; a Turing card's stock 256 MB window shares buffers up to 224 MiB, and the P2P guide's
  [turing.md](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/blob/main/docs/turing.md) T1-T3
  enlarge it.
- **Programs that map a lot of video memory** (llama.cpp with Resizable BAR) need the driver's static BAR1 off,
  `RMForceStaticBar1=0`, for every card ([troubleshooting](docs/troubleshooting.md)).
- **Where a Tesla V100 sits, if other cards read its memory.** In a slot wired through the board's chipset, a V100
  whose memory another card reads in large pieces can freeze when a program starts or stops on it. In a CPU slot it
  passed every run, and writing into it is safe in either slot: that is all llama.cpp does
  ([troubleshooting](docs/troubleshooting.md#a-tesla-v100-in-a-chipset-slot-freezes-xid-62-while-another-card-reads-its-memory)).

## Results

Copies of 1 GiB into the first card's memory / into the second card's, every word checked, against CUDA's
`cudaMemcpyPeer`, which goes through host RAM for every one of these pairs:

| Driver | Cards | Memory sharing | Both ways at once | 4-byte copy | Through host RAM |
|---|---|---|---|---|---|
| 580, closed | Tesla V100 + RTX 3090 | **2.96 / 3.26 GB/s** | **5.62 GB/s** | **2.4 / 3.1 µs** | 2.50 / 2.79 GB/s, 17.0 / 19.5 µs |
| 580, closed | Tesla V100 + RTX 2060 | **2.91 / 3.28 GB/s** | **5.62 GB/s** | **2.2 / 3.2 µs** | 2.27 / 2.51 GB/s, 19.9 / 21.5 µs |
| 580, closed | Tesla V100 + GTX 1070 Ti | **2.96 / 3.28 GB/s** | **5.63 GB/s** | **1.5 / 2.2 µs** | 2.82 / 3.22 GB/s, 18.4 / 20.4 µs |
| 580 open, 595, 615 | RTX 2060 + RTX 3060 | **2.93-2.97 / 3.18-3.28 GB/s** | **5.61-5.64 GB/s** | **1.6-2.2 / 2.7-3.1 µs** | 2.47 / 2.78 GB/s, 13.5 / 17.8 µs |
| 615 open | RTX 3090 (PCIe 4.0 x4 chipset slot) + RTX 3090 (x8) | **6.60 / 6.60 GB/s** | **12.84 GB/s** | **3.2 / 2.6 µs** | 4.11 / 4.02 GB/s, 18.8 / 15.5 µs |

Both ways at once is 5.6 GB/s for every pair on the first board: its x4 chipset slot is the ceiling. Two V100s on another
board reach 1.67 / 1.63 GB/s in a PCIe 2.0 x4 slot, the same as CUDA P2P there. On a third board, the two RTX 3090s on
x8 CPU lanes copy at 13.1-13.2 GB/s ([D5](docs/dmabuf-615.md#d5-three-rtx-3090s)).

llama.cpp, tensor parallel, with the all-reduce through shared video memory instead of host RAM:

| Cards, model | Results | Prompt | Generation |
|---|---|---|---|
| V100 (x4 chipset slot) + RTX 3090 (x16), Qwen3.8-27B Q8_0 | identical (perplexity equal on every chunk) | **+19 %** (413 vs 347 tokens/s) | +4 % (28.9 vs 27.8) |
| the same cards with the slots swapped: V100 (x16) + RTX 3090 (x4 chipset slot) | identical | **-18 %** (388 vs 474) | 0 % (30.2 vs 30.1) |
| three RTX 3090s, Qwen3.8-27B Q8_0, driver 615 | identical (perplexity equal on every chunk) | **+45 %** (1213 vs 835) | the same (37.0 vs 37.2) |
| RTX 2060 + RTX 3060, Qwen3.8-27B UD-Q3_K_XL, drivers 580 / 595 / 615 | identical on each driver | -1.5 to -3.5 % | +0.7 to +1.1 % (19.3 vs 19.2) |

Prompts of 2048 tokens; the two routes alternated, same binary. **On the V100 + RTX 3090 the slot layout decides.** The
V100 is the slower card and sets the pace. With it on the narrow x4 link, the shared all-reduce saves it from reading
host RAM over that link; with it on x16, host RAM is the better route and stock llama.cpp is faster
([volta-dmabuf.md V6](docs/volta-dmabuf.md#v6-llamacpp-tensor-parallel-with-an-all-reduce-through-shared-video-memory)).
On three RTX 3090s the gain shrinks with longer prompts: +40 % at 8,000 tokens, a few percent from 32,000 to 262,000
(single runs), with the generated text identical at every length ([D5](docs/dmabuf-615.md#d5-three-rtx-3090s)).

On the RTX 2060 + RTX 3060 the shared all-reduce gives no meaningful gain; tensor parallel itself generates 42 % faster
than layer split there ([benchmarks/llama](benchmarks/llama/README.md)).

## Every pair so far

The goal is for Hidden GEM to work on every pair, including those where CUDA P2P already works, so this table lists
every pair from this repository and from the P2P guide, side by side. The `not tested` cells are the to-do list.

| Cards | Generations | CUDA P2P | Hidden GEM |
|---|---|---|---|
| Tesla V100 + RTX 3090 | Volta + Ampere | **not possible**: the V100 needs the closed driver 580, and CUDA refuses peer access between two architectures (NVIDIA's `ForceP2P` key makes the kernel agree, CUDA still says no; a patched `libcuda` for 580 was not tried) | **works** on 580: 2.96 / 3.26 GB/s, 5.62 GB/s both ways, soak 20/20; llama.cpp Qwen3.8-27B identical, prompts +19 % with the V100 in the x4 chipset slot (with it in x16, host RAM is faster) ([volta-dmabuf.md](docs/volta-dmabuf.md)); a V100 in a chipset slot can freeze when read while a program starts or stops on it ([troubleshooting](docs/troubleshooting.md#a-tesla-v100-in-a-chipset-slot-freezes-xid-62-while-another-card-reads-its-memory)) |
| Tesla V100 + RTX 2060 | Volta + Turing | **not possible**, as above | **works** on 580: 2.91 / 3.28 GB/s, soak 20/20 (with an 8 GB window on the RTX 2060; with its stock 256 MB window, buffers up to 224 MiB) |
| Tesla V100 + GTX 1070 Ti | Volta + Pascal | **not possible**, as above | **works** on 580: 2.96 / 3.28 GB/s, soak 20/20; the 1070 Ti shares up to 224 MiB (256 MB window) |
| Two Tesla V100s | Volta | **works** on the stock 580 with the IOMMU in passthrough ([P2P guide](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/blob/main/docs/pascal-and-volta.md)) | **works** on 580: 1.67 / 1.63 GB/s, the same as CUDA P2P in that PCIe 2.0 x4 slot |
| RTX 2060 + RTX 3060 | Turing + Ampere | **works with workarounds**: the P2P guide's patched driver plus a patched copy of `libcuda`, or writes through the memory window as root ([P2P guide](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/blob/main/docs/mixed-generations.md)) | **works** on 580 (open modules), 595 and 615: 2.93-2.97 / 3.18-3.28 GB/s, 5.6 GB/s both ways, soaks 10/10 and 20/20; llama.cpp Qwen3.8-27B identical, same speed ([benchmarks/llama](benchmarks/llama/README.md)) |
| Three RTX 3090s | Ampere | **works** with the patched driver 615 | **works** on 615 (open modules): 6.60 GB/s through a x4 chipset link, 13.1-13.2 GB/s between two x8 cards, all six directions at once for over an hour, 56 TB checked, 0 wrong; llama.cpp Qwen3.8-27B identical, prompts +45 % ([dmabuf-615.md D5](docs/dmabuf-615.md#d5-three-rtx-3090s)) |
| RTX 3080 12 GB + RTX 3080 10 GB | Ampere (one chip) | **works** with the patched driver 615 | not tested |
| RTX 2060 + GTX 1650 SUPER | Turing (two chips) | **works** with the patched driver and a chip-ID shim for CUDA | not tested |
| Two P106-090s, in the test rig | Pascal | **works** on the stock 580 | not tested |
| GTX 1070 Ti + RTX 2060 | Pascal + Turing | **not possible**: two architectures, and Pascal needs 580 | not tested (the V100 works with each of them) |
| P106-090s behind the PCIe switches of a mining board | Pascal | **does not work**: the board drops the traffic | **does not work**: nothing arrives, no error ([volta-dmabuf.md](docs/volta-dmabuf.md#on-a-board-that-does-not-carry-peer-traffic)) |
| Maxwell, Ada, Hopper, Blackwell cards | | not tested | not tested (595 and 615 run Ada, Hopper and Blackwell on the same driver code as the cards above) |

## Contents

| | |
|---|---|
| [docs/volta-dmabuf.md](docs/volta-dmabuf.md) | Where it started: a Tesla V100 with an RTX 3090 (also an RTX 2060 and a GTX 1070 Ti) on NVIDIA's driver 580. Why nothing else works for a V100, the change, how to build and load it, tests, the llama.cpp all-reduce (V6), a board that does not carry peer traffic. |
| [docs/dmabuf-615.md](docs/dmabuf-615.md) | The same on Ubuntu's driver 615 for Turing and newer, tested on an RTX 2060 + RTX 3060; llama.cpp on that pair (D4); three RTX 3090s at once (D5). |
| [docs/dmabuf-drivers.md](docs/dmabuf-drivers.md) | Every NVIDIA branch from 580 to 620: which version of the change it takes, what was tested, all results in one table, llama.cpp on each driver, how to build for another version. |
| [docs/troubleshooting.md](docs/troubleshooting.md) | A Tesla V100 in a chipset slot that freezes while another card reads its memory; the kernel log filling with `dmaAllocMapping_GM107` lines (static BAR1); driver 595's per-card option that misses one card, with a fix. |
| [docs/test-rig-log.md](docs/test-rig-log.md) | What was run on the test rig on 2026-11-25, in order: what worked, what did not, the dead ends and false alarms, and what the tests do not show. |
| [benchmarks/llama](benchmarks/llama/README.md) | llama.cpp tensor parallel on the RTX 2060 + RTX 3060: scripts, results on three drivers, and a crash at exit on driver 615. |
| [tools/](tools) | `p2p_dmabuf.cpp` (the Vulkan test: share, copy both ways, check every word), `staged_baseline.cu` (CUDA through host RAM, the baseline), `keep_busy.cu` (keeps a receiving card out of its idle state), `bar1_probe.cpp` (how much host-visible video memory Vulkan can map), `p2p_iomem.cu` (the P2P guide's I/O-memory route, compared in volta-dmabuf.md); the changed `nvidia-drm` for 580 and 615 and the changed llama.cpp as pinned submodules. |
| [scripts/](scripts) | The steps as scripts, for a test machine: `drv-test.sh` (build Hidden GEM's `nvidia-drm` for the running driver and run every test: D1-D3 in one go), `switch-driver.sh` (another Ubuntu driver branch), `run-driver.sh` (NVIDIA's `.run` installer for branches Ubuntu does not package), `reload-nvidia.sh` (reload with driver keys, without a reboot). |
| [patches/](patches) | `595-per-device-registry-keys.diff`: 615's per-card option lookup for driver 595 ([troubleshooting](docs/troubleshooting.md)). |
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
