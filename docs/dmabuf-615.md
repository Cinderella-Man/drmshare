# Memory sharing on driver 615: an RTX 2060 and an RTX 3060, experimental

The memory sharing of [volta-dmabuf.md](volta-dmabuf.md) also works on this guide's driver 615, for the cards that driver
supports (Turing and newer). One opt-in option in `nvidia-drm`, built from Ubuntu's own copy of the driver source, lets
Vulkan on one card import the other card's video memory through a Linux dma-buf, and each card copies straight into the
other's memory. Tested on an RTX 2060 + RTX 3060, a pair CUDA refuses because the two cards are of different
generations. No patched driver and no patched `libcuda` are involved.

This is **not CUDA P2P**: only Vulkan programs can use it, and CUDA, NCCL, PyTorch and vLLM still see no peer access
([what this is, and what it is not](volta-dmabuf.md#what-this-is-and-what-it-is-not)).

| RTX 2060 + RTX 3060, driver 615 | Through system memory (CUDA's `cudaMemcpyPeer`) | **Memory sharing** |
|---|---|---|
| Copy 3060 -> 2060 / 2060 -> 3060 | 2.48 / 2.77 GB/s | **2.93 / 3.28 GB/s** |
| Both ways at once (writes) | | **5.62 GB/s** |
| A small (4-byte) copy, 3060 -> 2060 / 2060 -> 3060 | 13.5 / 17.8 µs | **1.59 / 2.68 µs** |
| Byte-exact checks | | 20 runs x 1 GiB, 0 wrong words |

The same speeds as [route 2 of mixed-generations.md](mixed-generations.md) (writes through the other card's memory
window with this guide's patched driver, 2.93 / 3.28 GB/s): the same PCIe path, here on NVIDIA's stock driver.

**Experimental.** Verified on 2026-11-24 on one pair and one driver version (Ubuntu's 615.71.09 open packages). The
changed module is loaded by hand and gone after a reboot.

## What changed from the 580 version

The same three `nvidia-drm` files as the 580 version, ported to NVIDIA's 615.71.09:
[`9616483`](https://github.com/Cinderella-Man/open-gpu-kernel-modules/commit/9616483fb515ed5ec93c7499fdda9d5e882c747f)
on branch `615-peer-vidmem-dmabuf` of a fork of NVIDIA's open kernel modules, pinned in this repository as
`tools/open-gpu-kernel-modules-615`. The 580 change applies to 615 as it is; two things are different:

- **The driver calls happen when the buffer is exported.** On 615, `nvidia.ko` first tries to attach and map an
  imported dma-buf itself, while it holds its own API lock. A first port called back into the driver at that moment,
  as the 580 version does (to map the memory window, and to keep the card awake), and the driver refused the nested
  call: the kernel log showed `Assertion failed: !rmapiLockIsOwner()` and `Error (89) while trying to import dma_buf!`,
  and the import fell back to a second path, which worked. Now both happen when the buffer is exported, and the map
  step only asks the kernel for a DMA address: no assertions, no failed first attempt.
- **The card stays awake while its memory is shared.** 615 can put an idle GPU into a deep power state (GC6) and keeps
  it out of it while the CPU maps its memory. The port does the same for a buffer another GPU uses.

## Test rig

| | |
|---|---|
| Board, CPU | B550 board, Ryzen 9 3900X |
| GPU in the CPU slot | RTX 3060, PCIe 4.0 x16, 16 GB window (Resizable BAR) |
| GPU in the chipset slot | RTX 2060, PCIe 3.0 x4, 8 GB window ([turing.md](turing.md) T1-T3) |
| Software | Ubuntu 26.04.1, kernel 7.0.0-44, `nvidia-driver-615-open` 615.71.09 (Ubuntu's packages, DKMS), `iommu=pt`, Ubuntu's CUDA 12.4 for the baseline |

## D1. Build nvidia-drm with the option

Needs Ubuntu's 615 open packages from [procedure.md step 3](procedure.md#3-nvidia-driver-6157109-open-kernel-modules-from-the-ubuntu-archive),
which put the driver source in `/usr/src/nvidia-615.71.09`. This guide's patched driver is not needed.

Take a copy of that source, apply the build fixes Ubuntu's DKMS applies for your kernel, and put the three changed
files in. DKMS lists the fixes in `dkms.conf` (`PATCH[n]`, each with the kernels it is for in `PATCH_MATCH[n]`); a copy
built by hand does not get them. On kernel 7.0 these are two:

```bash
cd ~/how-to-enable-p2p-on-nvidia-gpus
git submodule update --init --depth 1 tools/open-gpu-kernel-modules-615
grep -E '^PATCH(_MATCH)?\[' /usr/src/nvidia-615.71.09/dkms.conf
cp -a /usr/src/nvidia-615.71.09 ~/nv615-src
cd ~/nv615-src
for p in buildfix_kernel_7.0-dma_buf_invalidate_mappings.patch buildfix_kernel_7.0-cxl_structs.patch; do patch -p1 < patches/$p; done
cp ~/how-to-enable-p2p-on-nvidia-gpus/tools/open-gpu-kernel-modules-615/kernel-open/nvidia-drm/{nvidia-drm-gem.c,nvidia-drm-gem-nvkms-memory.c,nvidia-drm-gem-nvkms-memory.h} nvidia-drm/
make -j"$(nproc)" SYSSRC=/lib/modules/$(uname -r)/build modules
modinfo -F parm nvidia-drm.ko | grep peer
```

Expected: a 147 MB checkout, about 30 seconds of build, and
`peer_vidmem_dmabuf:Export video memory as dma-bufs another GPU can import (memory sharing through BAR1, not CUDA P2P) (bool)`.
Without the kernel 7.0 fixes the build stops at `nv-linux.h: error: field 'cxlds' has incomplete type`. Ubuntu's fixes
do not touch the three files. Only `nvidia-drm.ko` is used; the other modules of this build stay unused.

## D2. Load it (until the next reboot)

As on 580 ([volta-dmabuf.md V3](volta-dmabuf.md#v3-load-it-until-the-next-reboot)): stop the display manager, which
holds `nvidia-drm`, and let the machine reboot itself after a kernel crash instead of hanging.

```bash
sudo sysctl kernel.panic=10 kernel.panic_on_oops=1
sudo systemctl stop display-manager
sudo rmmod nvidia_drm
sudo insmod ~/nv615-src/nvidia-drm.ko modeset=1 fbdev=1 peer_vidmem_dmabuf=1
cat /sys/module/nvidia_drm/parameters/peer_vidmem_dmabuf
```

Expected: `Y`. Undo: reboot (the stock module loads again).

## D3. Test: each card writes and reads the other's memory

The same Vulkan test as [volta-dmabuf.md V4](volta-dmabuf.md#v4-test-each-card-writes-and-reads-the-others-memory).
GPU numbers follow the PCI bus order, as `nvidia-smi` counts them; the first number owns the memory, the second copies
into it.

```bash
sudo apt install -y libvulkan-dev
cd ~/how-to-enable-p2p-on-nvidia-gpus/tools
g++ -O2 -std=c++17 p2p_dmabuf.cpp -o p2p_dmabuf -lvulkan
./p2p_dmabuf 0 1 64
./p2p_dmabuf 1 0 64
```

On the test rig (run as root), the RTX 2060 as GPU 0:

```
A = NVIDIA GeForce RTX 2060 (04:00), B = NVIDIA GeForce RTX 3060 (2b:00)
A: 64 MiB in memory type 1 (DEVICE_LOCAL ), heap 0
dma-buf in the kernel: 67108864  ...  nvidia_drm  ...
A fills its exported buffer:     287.2 GB/s
B: imported A's memory
B copies into A:     2.93 GB/s, checked on A: OK (0 wrong words)
B reads from A:      1.14 GB/s, checked on B: OK (0 wrong words)
4-byte copy B -> A:   2.14 us
ALL OK
```

The last line must be `ALL OK`, and the exporter must say `nvidia_drm`: with the stock module it says `drm`, and the
run stops at `B cannot import A's dma-buf: no common memory type`. Both directions at once, and CUDA's copy through
system memory for comparison, as in V4 (build [`staged_baseline.cu`](../tools/staged_baseline.cu) for both cards:
`-gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_86,code=sm_86`).

**Time a copy into a card that is busy, as it is in real use.** An idle GeForce card drops to P8 and its PCIe link to
Gen1, and writes arriving from the other GPU do not wake it. For the RTX 2060 in the x4 chipset slot that capped a long
copy into it at 1.08 GB/s (link `Gen1 x4`); the RTX 3060 at `Gen1 x16` still received at full speed. When both cards
work, as in the both-ways test or a model split across them, the links stay at Gen3. Locking the clocks
(`nvidia-smi -lgc`) only reached P5 and Gen2 (1.85 GB/s). To time one direction, keep the receiving card busy with
[`tools/keep_busy.cu`](../tools/keep_busy.cu) (one block spinning, no memory traffic), on the receiving card only:
on the sending card it would take turns with the copy and slow it down.

```bash
nvcc -O2 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_86,code=sm_86 keep_busy.cu -o keep_busy
CUDA_DEVICE_ORDER=PCI_BUS_ID ./keep_busy 0 120 &
NO_READ=1 REPS=40 ./p2p_dmabuf 0 1 1024
nvidia-smi --query-gpu=index,pstate,pcie.link.gen.current,pcie.link.width.current --format=csv
```

Expected: GPU 0 at `P2` and link generation 3 while the copy runs, and about 2.9 GB/s.

## Results

1 GiB in each direction, every word checked:

| | Through system memory (CUDA's `cudaMemcpyPeer`) | Memory sharing |
|---|---|---|
| Copy 3060 -> 2060, the 2060 kept busy | 2.48 GB/s | **2.93 GB/s** |
| Copy 3060 -> 2060, the 2060 idle (link `Gen1 x4`) | | 1.08 GB/s |
| Copy 2060 -> 3060 | 2.77 GB/s | **3.28 GB/s** |
| Both ways at once (writes, 1 GiB x 20 each, four rounds) | | **5.62 GB/s** (2.89 + 2.73) |
| A 4-byte copy, 3060 -> 2060 / 2060 -> 3060 | 13.5 / 17.8 µs | **1.59 / 2.68 µs** |
| Reads of the other card's memory, 3060 reads the 2060's / 2060 reads the 3060's | | 0.56 / 1.42 GB/s |

Soak, 20 runs at 1 GiB, writes and reads, 10 in each direction: 20/20 `ALL OK`, 0 wrong words, no line from the driver in
the kernel log. Both cards ran `keep_busy` through the soak, so its speeds are not comparable and not shown.

Both ways at once gives 5.62 GB/s, the same as every V100 pair on 580: the x4 chipset slot is the ceiling.

## Limits

- **Needs a board that carries peer traffic**, like CUDA P2P. On a mining board with PCIe switches nothing arrived,
  and nothing said so ([volta-dmabuf.md](volta-dmabuf.md#on-a-board-that-does-not-carry-peer-traffic)).
- **Vulkan only.** CUDA, NCCL, PyTorch and vLLM do not see this route.
- **A card shares only what fits its memory window.** A Turing card's stock 256 MB window shares buffers up to
  224 MiB (measured on 580 with an RTX 2060); [turing.md](turing.md) T1-T3 enlarge it. RTX 30 cards and newer usually
  show all their memory with Resizable BAR on.
- **An idle receiving card in a narrow slot is slow** (above). Under a real load it is not.
- **Write across, do not read across** in bulk: reads of the other card's memory are slower, and heavy reads in both
  directions at once reset the Turing rig ([troubleshooting](troubleshooting.md#the-machine-resets-data-fabric-sync-flood)).
- **Loaded by hand,** with the display manager stopped. A boot install was not tested.
- **Not on Grace systems** (GH200, GB200): 615 itself refuses to map video memory there unless the buffer is
  physically contiguous.

## Not tested yet

- Ada, Hopper and Blackwell cards. 615 runs them on the same driver code, so they should behave the same.
- This module together with this guide's patched driver (it was built from and loaded with Ubuntu's stock modules).
- llama.cpp's all-reduce through shared video memory ([volta-dmabuf.md V6](volta-dmabuf.md#v6-llamacpp-tensor-parallel-with-an-all-reduce-through-shared-video-memory))
  on this pair; it needs only this module, so it should run unchanged.
- A boot install.

## Undo

Reboot: the stock `nvidia-drm` loads again, and the crash settings of D2 are gone. Nothing was installed.

## Credits

As for [volta-dmabuf.md](volta-dmabuf.md#credits): Christian König and the amdgpu developers for the dma-buf technique,
and NVIDIA for shipping `nvidia-drm` and the driver's kernel interface as source. The RTX 2060's 8 GB window comes from
[noon-at-cgn/geforce-bar1-resize](https://github.com/noon-at-cgn/geforce-bar1-resize) ([turing.md](turing.md)).
