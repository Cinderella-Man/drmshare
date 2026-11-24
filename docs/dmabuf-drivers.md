# Hidden GEM on each NVIDIA driver version

Where the memory sharing of [volta-dmabuf.md](volta-dmabuf.md) and [dmabuf-615.md](dmabuf-615.md) was tested, branch by
branch, and which version of the `nvidia-drm` change (**Hidden GEM**) each driver takes. The same idea everywhere: one opt-in option in
`nvidia-drm` lets Vulkan on one card import another card's video memory through a dma-buf. Not CUDA P2P: Vulkan
programs only ([what this is, and what it is not](volta-dmabuf.md#what-this-is-and-what-it-is-not)).

## The branches

| Branch | On Ubuntu 26.04 | Change to use | Tested here |
|---|---|---|---|
| 580 | `nvidia-driver-580` (closed) and `nvidia-driver-580-open`, 580.178.04 | 580 version: branch `580-peer-vidmem-dmabuf` ([`9904ca0`](https://github.com/Cinderella-Man/open-gpu-kernel-modules/commit/9904ca03caa2d1e133715fd902e503961d000eaa)) | **works** (closed package): Tesla V100 with an RTX 3090, an RTX 2060 and a GTX 1070 Ti; two V100s. Open modules: not run yet |
| 590 | none: Ubuntu's 590 packages install 595; NVIDIA's `.run` 590.48.01 | 580 version (590 has no GC6 guard either) | source only: the change applies |
| 595 | `nvidia-driver-595-open`, 595.99.02 | 615 version: branch `615-peer-vidmem-dmabuf` ([`9616483`](https://github.com/Cinderella-Man/open-gpu-kernel-modules/commit/9616483fb515ed5ec93c7499fdda9d5e882c747f)) | **works**: RTX 2060 + RTX 3060 |
| 610 | none: Ubuntu's 610 packages install 615; NVIDIA's `.run` and apt repository, 610.57.04 | 615 version | source only: the change applies |
| 615 | `nvidia-driver-615-open`, 615.71.09 (this guide's driver) | 615 version | **works**: RTX 2060 + RTX 3060 |
| 620 | none; NVIDIA's apt repository (620.70.10) and `.run` (620.73.04) | 615 version | source only: the change applies |

Older branch names on 26.04 (575, 570, 560, 550, 535) are transitional packages that install 580.

The code the change touches barely moves from 580 to 620: the three `nvidia-drm` files and the path by which
`nvidia.ko` imports a dma-buf change little between branches, and not in the parts the change uses. Two differences
decide which version to use:

- **The GC6 guard.** From 595 on, the driver keeps a GPU out of its deep power state (GC6) while its video memory is
  mapped, and the 615 version takes that guard while another GPU uses a buffer. 580 and 590 have no such guard, so
  the 615 version does not build there; they take the 580 version.
- **When the change calls into the driver.** The 580 version maps a buffer's memory window when the importing GPU
  first maps it. On 615 that moment can fall inside the driver's own import, under its API lock, and the nested call
  was refused (lock assertions in the kernel log, then a working fallback). The 615 version does it when the buffer is
  exported. On 580's closed package no test logged a lock assertion or a failed import.

## Results

Copies into the first card's memory / into the second card's memory, 1 GiB, every word checked. "Through system memory"
is CUDA's own `cudaMemcpyPeer`, which goes through host RAM for every pair here.

| Driver | Cards (slots) | Copy, memory sharing | Both ways at once | 4-byte copy | Through system memory | Soak |
|---|---|---|---|---|---|---|
| 580.178.04, closed | Tesla V100 (chipset, PCIe 3.0 x4) + RTX 3090 (CPU slot) | 2.96 / 3.26 GB/s | 5.62 GB/s | 2.39 / 3.11 µs | 2.50 / 2.79 GB/s, 17.0 / 19.5 µs | 20/20 |
| 580.178.04, closed | Tesla V100 (chipset) + RTX 2060 (CPU slot) | 2.91 / 3.28 GB/s | 5.62 GB/s | 2.15 / 3.18 µs | 2.27 / 2.51 GB/s, 19.9 / 21.5 µs | 20/20 |
| 580.178.04, closed | Tesla V100 (chipset) + GTX 1070 Ti (CPU slot) | 2.96 / 3.28 GB/s | 5.63 GB/s | 1.51 / 2.17 µs | 2.82 / 3.22 GB/s, 18.4 / 20.4 µs | 20/20 |
| 580.178.04, closed | two Tesla V100s (another board, PCIe 2.0 x4 chipset slot) | 1.67 / 1.63 GB/s | | | (CUDA P2P works here: 1.62 / 1.67 GB/s) | 20/20 |
| 595.99.02, open | RTX 2060 (chipset) + RTX 3060 (CPU slot) | 2.97 / 3.28 GB/s | 5.64 GB/s | 2.19 / 3.05 µs | 2.47 / 2.78 GB/s, 13.6 / 17.5 µs | 10/10 |
| 615.71.09, open | RTX 2060 (chipset) + RTX 3060 (CPU slot) | 2.93 / 3.28 GB/s | 5.62 GB/s | 1.59 / 2.68 µs | 2.48 / 2.77 GB/s, 13.5 / 17.8 µs | 20/20 |

Every run: the stock `nvidia-drm` refuses the import (`no common memory type`), the changed one allows it, and the
kernel log has no line from the driver (apart from the expected refusal of a buffer larger than a card's window). Both ways at once is the same 5.6 GB/s for every pair on this board: its x4
chipset slot is the ceiling. The 4-byte copy varies between runs and buffer sizes (1.5-3.2 µs); through system memory
it takes 13.5-21.5 µs. On the RTX 2060 + RTX 3060 one-way copies were timed with the receiving card kept busy
([why](dmabuf-615.md#d3-test-each-card-writes-and-reads-the-others-memory)). The memory window limits what a card can
share: a 256 MB window (GTX 1070 Ti; RTX 2060 before [turing.md](turing.md) T1-T3) shares buffers up to 224 MiB.

Software: Ubuntu 26.04.1, kernel 7.0.0-44, Ubuntu's driver packages (DKMS), `iommu=pt`, B550 board with a Ryzen 9
3900X; the two V100s on Ubuntu 24.04.5, kernel 7.0.0-43, ASUS TUF GAMING B450-PLUS II with a Ryzen 7 5700X.

## Building for another driver version

[dmabuf-615.md D1](dmabuf-615.md#d1-build-nvidia-drm-with-the-option) copies three whole files, which fits 615 only.
For any other version, apply the change as a diff to that driver's own source:

```bash
cd ~/how-to-enable-p2p-on-nvidia-gpus
git submodule update --init --depth 2 tools/open-gpu-kernel-modules-615
git -C tools/open-gpu-kernel-modules-615 diff HEAD~1 HEAD -- kernel-open/nvidia-drm > ~/peer-vidmem.diff
V=$(modinfo -F version nvidia)
cp -a /usr/src/nvidia-$V ~/nv-src
cd ~/nv-src
grep -E '^PATCH(_MATCH)?\[' dkms.conf
patch -p2 < ~/peer-vidmem.diff
make -j"$(nproc)" SYSSRC=/lib/modules/$(uname -r)/build modules
modinfo -F parm nvidia-drm.ko | grep peer
```

For 580 and 590 take `tools/open-gpu-kernel-modules-580` instead. `--depth 2` fetches the commit's parent, which the
diff needs. Before `patch`, apply the files in `patches/` that `dkms.conf` lists for your kernel (`PATCH[n]`, matched
against `uname -r` by `PATCH_MATCH[n]`), as DKMS does: on kernel 7.0, Ubuntu's 615 lists two (see dmabuf-615.md D1),
Ubuntu's 595 none. Then load and test as in [dmabuf-615.md](dmabuf-615.md) D2 and D3.

## Not tested yet

- The 580 version on 580's open modules (Turing and newer cards), and on 590.
- 610 and 620: NVIDIA's own installers, which do not carry Ubuntu's kernel 7.0 fixes.
- Maxwell, Ada, Hopper and Blackwell cards.
