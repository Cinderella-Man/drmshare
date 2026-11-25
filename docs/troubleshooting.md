# Troubleshooting

Problems that came up with Hidden GEM and the programs that use it. The hardware-side problems (a board that does not
carry peer traffic, a chipset that drops peer reads, the machine reset under heavy reads in both directions) are the
same as for CUDA P2P and are described in the P2P guide's
[troubleshooting](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/blob/main/docs/troubleshooting.md)
and [boards](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/blob/main/docs/boards.md) pages.

## Vulkan: the kernel log fills with `dmaAllocMapping_GM107: can't alloc VA space`

Seen with llama.cpp's Vulkan backend on an RTX 3060 12 GB with Resizable BAR (16 GB window) on driver 580 and 595, with
or without memory sharing; measured on 615 too, which logs fewer lines (25 per failed allocation instead of 73).

**Symptom:** while a model of more than about 4 GB loads onto that card, the kernel log gets hundreds of `NVRM` lines
(1182 for one load of Qwen3.8-27B split across two cards):

```
NVRM: dmaAllocMapping_GM107: can't alloc VA space for mapping.
NVRM: mmuWalkMap: Failed to map VA Range 0x1fd40000 to 0x1fd7ffff. Status = 0x00000040
NVRM: dmaAllocMapping_GM107: can't update VA space for mapping @vaddr=0x1fda0000
NVRM: nvAssertOkFailedNoLog: Assertion failed: Out of memory [NV_ERR_NO_MEMORY] (0x00000051) returned from reusemappingdbMap(&pBar1VaInfo->reuseDb, ...
```

The model still runs and gives the same results. A Vulkan program that allocates host-visible video memory
(`DEVICE_LOCAL | HOST_VISIBLE`) gets `VK_ERROR_OUT_OF_DEVICE_MEMORY` after 4 GiB, although the card has 12 GB:

```bash
g++ -O2 -std=c++17 tools/bar1_probe.cpp -o tools/bar1_probe -lvulkan
sudo tools/bar1_probe 1 1024 11
```

prints `block 4: vkAllocateMemory VkResult -2, stop` and `done: 4 blocks, 4.00 GiB mapped, NVRM lines +73` (the
first argument is the card, in `nvidia-smi` order).

**Cause:** the driver's **static BAR1**. When a card's memory window can hold all of its memory (16 GB window, 12 GB
memory), the driver maps all the memory into the window once, at load, with 2 MB pages. Not every allocation may use
that mapping (`kbusIncreaseStaticBar1Refcount_TU102` in NVIDIA's source: on 580 and 595 only memory with 2 MB pages;
615 relaxes the rule, and Vulkan's memory still does not qualify), so the driver maps Vulkan's host-visible memory into
what is left of the window: 16 - 12 = 4 GB. llama.cpp puts every buffer in
host-visible video memory when Resizable BAR is on. Once those 4 GB are used, every further attempt fails inside the
driver (the log lines) and llama.cpp quietly falls back to plain video memory. A card whose window the driver does not
map statically (here the RTX 2060 with its 8 GB window for 6 GB) maps all its memory without a line.

**Fix:** turn static BAR1 off for that card with `RMForceStaticBar1=0`, the per-card option of
[procedure step 11b](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/blob/main/docs/procedure.md#11b-driver-dynamic-bar1-for-each-card-rmforcestaticbar10), for **every** card, and
reboot. If the geforce-bar1-resize tool writes that file, its default `DRIVER_OPTION=auto` covers only the cards it
enlarged itself; a card with a large window of its own (most RTX 30 and newer with Resizable BAR on) needs
`DRIVER_OPTION=on` in `/etc/geforce-bar1-resize.conf`, then:

```bash
sudo geforce-bar1-resize driver-option --write
sudo modprobe -c | grep NVreg_RegistryDwordsPerDevice
```

Expected: one `pci=...;RMForceStaticBar1=0;` group per card. After the reboot `tools/bar1_probe 1 1024 11` maps
`11.00 GiB` with `NVRM lines +0`, and the model loads without a line. Speeds in llama.cpp do not change (same numbers
within 1 % on 580 and 595).

**On driver 595 the per-card option misses one card.** 595 reads `NVreg_RegistryDwordsPerDevice` while it sets up each
card, but looks the card up in its device list before the card is added to it. Each card gets its keys only when a
later card is set up, so the last card the driver finds (here the RTX 3060 at `2b:00.0`) never gets them, and on a
machine with one GPU no card does. 580 and 615 do not have this problem: 580 adds the card first, and 615 matches the
keys against the card being set up (the fix NVIDIA shipped). On 595, give the key to all cards at once instead:

```bash
echo 'options nvidia NVreg_RegistryDwords="RMForceStaticBar1=0"' | sudo tee /etc/modprobe.d/zz-static-bar1-off.conf
```

or skip it for one program: `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` makes llama.cpp use plain video memory (no lines
in the log; its speed was not timed here).

