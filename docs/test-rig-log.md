# Test-rig log

What was actually run on the test rig, in order, with the dead ends. The pages in [docs/](.) give the procedures and
results; this page is the record behind them. The earlier work (2026-11-24: the Tesla V100 pairs on driver 580, the
first port to 615, every driver branch) is told in [volta-dmabuf.md](volta-dmabuf.md),
[dmabuf-615.md](dmabuf-615.md) and [dmabuf-drivers.md](dmabuf-drivers.md).

## 2026-11-25: Qwen across an RTX 2060 + RTX 3060, on drivers 580, 595 and 615

| | |
|---|---|
| Board, CPU, RAM | B550, Ryzen 9 3900X, 8 GB |
| GPU 0 | RTX 2060 6 GB (TU104), chipset slot, PCIe 3.0 x4, 8 GB window (the P2P guide's Turing steps T1-T3) |
| GPU 1 | RTX 3060 12 GB (GA106), CPU slot, PCIe 4.0 x16, 16 GB window (Resizable BAR) |
| Software | Ubuntu 26.04.1, kernel 7.0.0-44, Ubuntu's `nvidia-driver-*-open` packages, llama.cpp from `tools/llama.cpp` (`ab88444`), Vulkan build |
| Model | Qwen3.8-27B UD-Q3_K_XL, 13.1 GB, sha256 checked after the copy |

### 1. Driver 580, open modules: DRMShare

The rig was on Ubuntu's `nvidia-driver-580-open` from the day before, never tested with DRMShare.
`scripts/drv-test.sh` with the published 580 version (`9904ca0`): the stock `nvidia-drm` refuses the import, the
changed one shares 1 GiB each way byte-exact (3.18 / 2.95 GB/s), 5.63 GB/s both ways, soak 10/10, and the kernel log
has no driver line, no lock assertion and no failed import. So the export-time variant that 615 needed is not needed
on 580; the published 580 version works on the open modules as it is.

### 2. Choosing the model

With tensor parallel each card holds its share of every layer, so the 6 GB card sets the limit. The Q8_0 of
Qwen3.8-27B (29 GB) does not fit the pair at all; the UD-Q3_K_XL (13.1 GB) split 1:2 puts about 4.4 GB on the RTX 2060
and leaves room for the KV cache. Perplexity of Q3_K_XL over the same text: 6.13, against 5.99 for Q8_0.

### 3. First Qwen run (580 open): correct, but the kernel log fills up

`benchmarks/llama/run.sh`: perplexity 6.1281 with the all-reduce through host RAM and through shared video memory
alike, the log line `all-reduce writes straight into peer VRAM` and a 40 MiB receive buffer in each card. But each model
load added 1182 `NVRM` lines to the kernel log (`dmaAllocMapping_GM107: can't alloc VA space`, `mmuWalkMap: Failed to
map VA Range`, `NV_ERR_NO_MEMORY` from `reusemappingdbMap`).

**Dead end: `llama-bench -ts 1,2` aborted** every run. In `llama-bench` a comma separates configurations, so `1,2` ran
a split of `1` and one of `2`, each putting everything on one card; one split's values are separated with `/`:
`-ts 1/2`. (`llama-perplexity` and `llama-server` take `-ts 1,2`.)

### 4. Where the driver lines come from

Counted per step:

| Run | Driver lines |
|---|---|
| Perplexity with the all-reduce through host RAM (no dma-buf at all) | 50 to 1182 |
| Qwen3-0.6B, any card or both, any setting | 0 |
| Qwen3.8-27B, RTX 2060 alone, 8 / 16 / 24 layers | 0 / 0 / 0 |
| Qwen3.8-27B, RTX 3060 alone, 8 / 16 / 32 / 48 layers | 0 / 0 / 624 / 1203 |
| Qwen3.8-27B, both cards, `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` | 0 |

**Dead end:** the first single-card runs exited at once. `llama-bench` appends repeated options: `-ngl 99 ... -ngl 8`
ran 99 layers on one card first, which does not fit.

So: the RTX 3060, more than about 3 GB, and only with host-visible video memory (llama.cpp puts every buffer there when
Resizable BAR is on). [`tools/bar1_probe.cpp`](../tools/bar1_probe.cpp) maps 1 GiB blocks of it: the RTX 3060 stops at
4 GiB (`VK_ERROR_OUT_OF_DEVICE_MEMORY`, 73 driver lines for the failed fifth block), the RTX 2060 maps 5 GiB without a
line, and every page that did map reads back right. NVIDIA's source explains it: when the window can hold all of the
memory (16 GB for 12 GB), the driver maps the memory into it statically at load with 2 MB pages, only memory with
2 MB pages may use that mapping on 580 and 595, and Vulkan's host-visible memory goes into the rest, 16 - 12 = 4 GB.
llama.cpp then falls back to plain video memory, silently. Details: [troubleshooting](troubleshooting.md).

**Fix:** reloading the driver with `NVreg_RegistryDwords="RMForceStaticBar1=0"` (`scripts/reload-nvidia.sh`): the RTX
3060 maps 11 GiB without a line. The full run again: 0 driver lines in every step, perplexity and speeds the same.

The rig's `/etc/modprobe.d` had `RMForceStaticBar1=0` for the RTX 2060 only: `geforce-bar1-resize driver-option` in its
default `DRIVER_OPTION=auto` covers only the cards it enlarged itself, and the RTX 3060's window comes from its own
firmware. Set to `DRIVER_OPTION=on` and rewritten (`geforce-bar1-resize driver-option --write`): one group per card.

### 5. Driver 595: DRMShare, Qwen, and a per-card option that misses one card

`scripts/switch-driver.sh 595`, reboot, `scripts/drv-test.sh` with the 615 version: passed (3.27 / 2.94 GB/s,
5.61 GB/s both ways, soak 10/10, no driver line). The Qwen run: correct (perplexity 6.1309 both ways), but 1182 driver
lines again, although both cards had `RMForceStaticBar1=0` in `NVreg_RegistryDwordsPerDevice`. Reloads with other
strings, measured with `bar1_probe`:

| Keys | Result |
|---|---|
| per card, both cards | RTX 3060 stops at 4 GiB: the key did not reach it |
| per card, the RTX 3060 only | 4 GiB |
| per card, both, the RTX 3060 first | 4 GiB |
| for all cards (`NVreg_RegistryDwords`) | 11 GiB, no line |
| per card, `RMForceStaticBar1=1` for the RTX 2060 | RTX 2060 stops at 2 GiB (8 - 6): the key did reach it |

In 595's source the per-card string is parsed at the end of each card's setup, but the card is looked up in a list it
joins only later in that setup: each card gets its keys when a later card is set up, and the last one never does.
580 adds the card first; 615 matches the keys against the card being set up. With 615's version of the lookup in 595's
`nvidia.ko` ([`patches/595-per-device-registry-keys.diff`](../patches/595-per-device-registry-keys.diff)) the per-card
key reaches the RTX 3060 (11 GiB, alone or with both cards listed), and the RTX 2060 still gets its own (2 GiB with
`RMForceStaticBar1=1`). The full Qwen run on that `nvidia.ko`: 0 driver lines.

**False alarm: a slow chat.** The first `llama-server` answer on 595 generated at 9.7 tokens/s with the prompt at
1.0 token/s, about half the speed of 580's run without static BAR1, so static BAR1 looked like it cost speed. A second
server in the same state answered at 18.9 tokens/s: the first request after a driver change compiles the Vulkan
pipelines the server needs. The run script now sends two requests.

### 6. Driver 615: DRMShare, Qwen, and a crash at exit

`scripts/switch-driver.sh` back to the P2P guide's 615 set, reboot. 615 applies the per-card key to the RTX 3060
(11 GiB with the per-card string alone). `scripts/drv-test.sh` with the 615 version: passed (3.28 / 2.96 GB/s,
5.63 GB/s both ways, soak 10/10, no driver line). Qwen: perplexity 6.1305 both ways, 0 driver lines, the chat at
19.0 tokens/s on the second request.

But `llama-bench` ended with exit status 139 in 3 of the 4 tensor-parallel runs, after printing its results. Then:

| Test | Crashes |
|---|---|
| Short settings (`-p 64 -n 16 -r 1`): Qwen3-0.6B on each card and on both, Qwen3.8-27B on both | 0 of 30 |
| Full settings under gdb | 2 of 7 |
| Full settings, a fix that only waits for the devices (v1) | 0 of 8, but see below |
| Unfixed, right after (cache warm from all the runs before) | 0 of 3 |
| Unfixed, fresh empty shader cache per run (`__GL_SHADER_DISK_CACHE_PATH`) | 3 of 4 |
| The fix that destroys the devices (v2), fresh cache per run, alternating with the unfixed runs | 0 of 4 |

gdb: thread 3 crashes in NVIDIA's shader compiler (`libnvidia-glvkspirv.so.615.71.09`, started by
`libnvidia-eglcore`) while thread 1 runs `__run_exit_handlers` -> an exit handler of `libEGL_nvidia.so.0` -> `dlclose`.
The driver unloads itself while its own compile thread still runs, which happens when the cache was cold and the
driver is still compiling at exit. ggml-vulkan never destroyed its devices (each device's buffer type holds a reference
back to it: 2 references at exit), so nothing stopped that work in time. v1 only waited for the devices; it looked
perfect, but the unfixed build stopped crashing too once the cache was warm, so it proved nothing. v2 also drops the
references and destroys the devices in an exit handler that runs before NVIDIA's (1 reference at release): 0 of 4 with
a cold cache. It is commit [`4edab03`](https://github.com/Cinderella-Man/miner-llama/commit/4edab03eff0937fc5d3923b8b63ec52d0baa1d24)
on the llama.cpp fork, pinned as `tools/llama.cpp`.

615 without any key also stops the RTX 3060 at 4 GiB (25 driver lines per failed block instead of 73): 615 relaxes the
static BAR1 rule, and Vulkan's memory still does not qualify.

### 7. Results

| Driver | Generation, host RAM / shared | Prompt 2048, host RAM / shared | Perplexity, both the same |
|---|---|---|---|
| 580, open modules | 19.16 / 19.37 tokens/s | 475 / 467 | 6.1281 |
| 595 | 19.19 / 19.32 | 474 / 461 | 6.1309 |
| 615 | 18.88 / 19.01 | 483 / 466 | 6.1305 |

Layer split: 13.5 tokens/s generating, about 615 tokens/s reading 2048-token prompts. Every number and file:
[benchmarks/llama](../benchmarks/llama/README.md).

### What this does not show

- A long run: each setting ran for minutes (perplexity over 8 chunks, `llama-bench` with 3 repetitions, two chat
  answers). No hours-long server soak, no long contexts, no several users at once, one quantization of one model.
- A speedup on this pair: the shared all-reduce gives the same speed as host RAM within 1-4 %.
- DRMShare together with the P2P guide's patched driver: every run here used Ubuntu's stock modules.
- A boot install: the changed `nvidia-drm` was loaded by hand each time.

Rig state at the end: Ubuntu's 615 open packages, held; `RMForceStaticBar1=0` for both cards through
`geforce-bar1-resize` (`DRIVER_OPTION=on`); DRMShare's `nvidia-drm` loaded by hand.
