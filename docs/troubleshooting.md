# Troubleshooting

Problems that came up with DRMShare and the programs that use it. The hardware-side problems (a board that does not
carry peer traffic, a chipset that drops peer reads, the machine reset under heavy reads in both directions) are the
same as for CUDA P2P and are described in the P2P guide's
[troubleshooting](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/blob/main/docs/troubleshooting.md)
and [boards](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/blob/main/docs/boards.md) pages.

## A Tesla V100 in a chipset slot freezes (Xid 62) while another card reads its memory

Seen on driver 580 (closed) with a Tesla V100 in the chipset slot of two AMD desktop boards (B550 and B450), with an
RTX 3090 or another V100 reading its memory through memory sharing.

**Symptom:** the V100 stops answering. The kernel log shows `NVRM: Xid (PCI:...): 62` (an internal micro-controller
halt), `nvidia-smi` hangs, and programs on the V100 hang inside the driver. The milder form: the reading card gets
wrong data for a moment (every wrong word `0xffffffff`, or data from other places in the same buffer), and the V100
may log `Xid 32`. The V100's memory itself stays correct: a second read straight after is right.

**When:** it takes three things at once.

1. The V100 sits in a slot wired through the board's chipset, not in one wired to the CPU.
2. Another card reads the V100's memory in large pieces (1 MiB per copy or more).
3. A program starts or stops on the V100 at that moment: a Vulkan device or CUDA context created or destroyed, or a
   process ending, whether normally, by a crash or killed.

A loop that keeps starting programs on the V100 during such reads freezes it within seconds. A single program ending
at the wrong moment mostly gives the milder form, sometimes the freeze (7 of 20 killed programs did one or the other in
one test).

**Not affected**, tested:

- writes into the V100's memory: more than 300 runs, every write checked on the V100 during program starts, ends and
  kills. llama.cpp's all-reduce only writes into the other card, so it is not affected;
- reads in pieces of 64 KiB or less, each copy on its own (10 runs each);
- the same V100 in a CPU slot (160 runs on two boards);
- RTX 3090s being read, in a CPU slot or behind a chipset (about 380 runs);
- CUDA P2P between two V100s with one of them in the chipset slot (60 runs).

**What to do:** put a V100 whose memory other cards read in a CPU slot, or only write across. After a freeze a normal
reboot can hang in the driver's shutdown; a power cycle brings the card back whole, and so does Linux's emergency
reboot, which skips that shutdown (`sync`, then `echo b | sudo tee /proc/sysrq-trigger`; it restarts at once, so save
your work first).

**Cause:** not proven. In the test where it was measured, the frozen V100 logged a PCIe Completion Timeout of its own:
while it sets up or tears down a device it needs answers from host memory, and the likely explanation is that these
queue behind the other card's read requests in the chipset's PCIe switch until the V100's micro-controller gives up.
It happens inside NVIDIA's closed driver and the card's firmware, so the `nvidia-drm` change cannot fix it. Without
memory sharing nothing reads the V100's memory this way: CUDA P2P on these cards did not trigger it.

## Two cards behind one AMD chipset: the copy finishes, nothing arrives

Seen on the test rig's B550 board (driver 580, closed) with an RTX 3090 in the chipset's x16 slot (PCIe 3.0 x4) and a
GTX 1650 SUPER in one of its x1 slots, a Tesla V100 in the CPU slot. [V4](volta-dmabuf.md#v4-test-each-card-writes-and-reads-the-others-memory)'s
test, 1 MiB, every word checked:

| Pair | Path between the cards | Writes | Reads |
|---|---|---|---|
| V100 and GTX 1650 SUPER, both ways | through the CPU | 0 wrong (0.21 GB/s, the x1 link's limit) | 0 wrong |
| V100 and RTX 3090, both ways | through the CPU | 0 wrong (2.9-3.4 GB/s) | 0 wrong |
| RTX 3090 and GTX 1650 SUPER, both ways | **both behind the chipset** | **every word wrong** | **every word wrong** (0.00 GB/s) |

**Symptom:** the import works and the copy engine reports the copy done, but nothing lands, with nothing in the kernel
log. With the driver's GSP firmware on (the default for Turing and newer), the card waiting for the reads can stop
answering instead: `Xid 119` (`Timeout after 45s of waiting for RPC response from GPU GSP`) three times, then `Xid 154`
(`GPU Reset Required`); it needs a reboot. On this AMD board the card that does the copying logs a PCIe Completion
Timeout and a Malformed TLP in its error registers (`lspci -vvv`, `UESta`). On an Intel Z390 (outside report, below)
nothing is recorded at all: the firmware keeps AER from Linux, and even the raw registers of the chipset's ports and
the cards latch nothing. Only a check of the data shows it.

**Cause:** the chipset's own ports refuse requests from one of its slots to another. The header of the logged packet
(`ECAP_AER+0x1c`) is a completion with status Unsupported Request, sent by the copying card's own chipset port. Reads
get that answer; writes, which get no answer, are dropped the same way. Requests that leave the chipset toward the CPU
are passed on, which is why every pair through the CPU works. The chipset's ports have no ACS, so there is no switch to
turn this on. With the IOMMU translating instead of in passthrough mode (`iommu=pt` dropped) the request does leave the
chipset, and the CPU sends it back down: the chipset then jammed for 2.5 minutes (network included) until the RTX 3090
halted (`Xid 62`), and in the other direction the machine hung until a power cycle. Not a route either.

**What to do:** at most one card behind an AMD chipset; the others on CPU lanes (for more than one CPU slot's worth, a
bifurcation splitter in the CPU slot, with the board's lane setting to match). Test every new pair with V4's test,
which checks every word: a program that does not check would compute with wrong data. CUDA P2P fails the same way:
an outside report on an Intel Z390 with three RTX A2000s (CUDA P2P on the stock driver) lost every write between the
two cards on the chipset's root ports, while `nvidia-smi topo -p2p r` said `OK` and every pair with the CPU-port card
was byte-exact ([the P2P guide's issue #5](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/issues/5)).

## A card on a PCIe x1 riser falls off the bus (Xid 79) when the driver starts it

Seen with a GTX 1650 SUPER on a USB-cable x1 riser (the mining kind) in a chipset x1 slot of the test rig's B550 board,
driver 580.

**Symptom:** the card shows up at boot, linked at 2.5 GT/s x1; about 4 seconds after the driver loads, `NVRM: Xid
(PCI:...): 79, GPU has fallen off the bus`, then `RmInitAdapter failed` over and over. Every access to the dead card
stalls a CPU core (`pci_pme_list_scan hogged CPU`, clock-watchdog warnings), `nvidia-smi` and `lspci` hang, and about
90 seconds after boot the whole machine stops answering. The card's port cannot be read either once the card is gone,
so the link cannot be fixed from Linux after the fact.

**Cause:** cards idle at 2.5 GT/s and the driver raises the link speed when it starts the card; the riser does not carry
8 GT/s.

**What to do:** keep the link at 2.5 GT/s from the start. Some boards have a per-slot speed setting; the test rig's
B550 only has one for the CPU slot and a "Chipset Gen Switch", which set the CPU-to-chipset link to 2.5 GT/s instead
(and the board then did not set up its chipset slots at all: the cards behind it disappeared). What worked is a
one-shot service that, before the driver loads, sets the Target Link Speed of the port above every NVIDIA card on an
x1 link to 2.5 GT/s and retrains the link: [`scripts/x1-riser-gen1.sh`](../scripts/x1-riser-gen1.sh) with
[`scripts/x1-riser-gen1.service`](../scripts/x1-riser-gen1.service). With it the card started cleanly at every boot
(2.5 GT/s x1, about 0.25 GB/s each way). The same riser at each speed (the cap set before the driver starts the card:
raised later, it is not used):

| Cap | Driver start | Copies, each way | Every word checked | The riser link's correctable errors |
|---|---|---|---|---|
| none (8 GT/s) | the card falls off the bus (`Xid 79`) | | | |
| 5 GT/s (`X1_RISER_SPEED=2`) | fine | 0.42 / 0.38 GB/s | 0 wrong, 32 GB in 60 s both ways | **many** (`RxErr`, `BadTLP`, `BadDLLP`, replay timeouts), from the driver start on |
| 2.5 GT/s (the default) | fine | 0.21 / 0.19 GB/s | 0 wrong, about an hour of traffic | **none** |

At 5 GT/s the link keeps resending damaged packets, which PCIe does by itself, so the data still arrives right, but
there is no margin left; 2.5 GT/s is the safe setting for this riser. Install:

```bash
sudo install -m 0755 scripts/x1-riser-gen1.sh /usr/local/sbin/x1-riser-gen1
sudo install -m 0644 scripts/x1-riser-gen1.service /etc/systemd/system/
sudo systemctl enable x1-riser-gen1.service
```

It only touches cards on an x1 link; undo with `sudo systemctl disable x1-riser-gen1.service`. The NVIDIA module must
not be in the initramfs (it is not on Ubuntu), or it loads before the service runs.

**Is a riser close to its limit?** Look at the riser link's correctable errors under load: `lspci -vvv` (`CESta` on the
card and on the port above it) or `/sys/bus/pci/devices/<card>/aer_dev_correctable`. An outside report's USB-cable
riser ran an RTX A2000 at 8 GT/s for a whole run, but that link logged `RxErr` / `BadTLP` while every other link was
clean: it works with no margin. The test rig's riser looked the same at 5 GT/s and logged none at 2.5 GT/s (table
above).

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

**Or fix 595's per-card option** (tested on the RTX 2060 + RTX 3060): [`patches/595-per-device-registry-keys.diff`](../patches/595-per-device-registry-keys.diff)
puts 615's version of that lookup into 595's `nvidia.ko` (three files of the open part, `common/inc/nv-proto.h`,
`nvidia/nv-pci.c`, `nvidia/os-registry.c`; 38 changed lines). Build it from Ubuntu's copy of the source and load it by
hand; a reboot returns to Ubuntu's module:

```bash
cp -a /usr/src/nvidia-595.99.02 ~/nv595-reg
cd ~/nv595-reg
patch -p1 < ~/drmshare/patches/595-per-device-registry-keys.diff
make -j"$(nproc)" SYSSRC=/lib/modules/$(uname -r)/build modules
sudo systemctl stop gdm nvidia-persistenced
sudo rmmod nvidia_drm nvidia_uvm nvidia_modeset nvidia
sudo insmod nvidia.ko NVreg_RegistryDwordsPerDevice="pci=0000:04:00.0;RMForceStaticBar1=0;pci=0000:2b:00.0;RMForceStaticBar1=0;"
sudo insmod nvidia-modeset.ko
sudo insmod nvidia-uvm.ko
```

Put your own cards' addresses in the `pci=` groups (`lspci -D -d 10de::0300`). `insmod` does not read
`/etc/modprobe.d`, so the keys go on its command line. Then load DRMShare's `nvidia-drm` as usual (if the stock one
came back with `nvidia-modeset`, `rmmod nvidia_drm` first). Expected: `sudo tools/bar1_probe 1 1024 11` maps
`11.00 GiB` on the RTX 3060 with `NVRM lines +0`, with only the per-card keys; giving only the 3060 a key works too,
and a per-card `RMForceStaticBar1=1` for the first card still reaches it (2 GiB on the RTX 2060). Not published as a
driver branch.

