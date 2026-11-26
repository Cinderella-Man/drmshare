# llama.cpp on two GPUs: the all-reduce through host RAM vs through shared video memory

Does memory sharing ([DRMShare](../../docs/dmabuf-615.md)) make a real language model faster? This folder runs
**llama.cpp**'s Vulkan backend with **tensor parallel** (`-sm tensor`: every layer is split across both cards, which add
up their partial results, an all-reduce, after each one) and compares the all-reduce through host RAM (upstream) with
the all-reduce through shared video memory (`GGML_VK_ALLREDUCE_P2P=1`, the changed llama.cpp of
[volta-dmabuf.md V6](../../docs/volta-dmabuf.md#v6-llamacpp-tensor-parallel-with-an-all-reduce-through-shared-video-memory)).
Layer split (`-sm layer`) runs as a reference: each card holds whole layers, and only one small vector crosses per
token.

Everything here was run on the test rig's **RTX 2060 + RTX 3060** on 2026-11-25, on three driver versions. This is
memory sharing, not CUDA P2P: CUDA refuses this pair of two generations.

## Short answer

On this pair: **no meaningful speedup from the shared all-reduce, and identical results.** Shared video memory vs host
RAM, mean of two rounds of three repetitions each:

| | 580.178.04 open | 595.99.02 | 615.71.09 |
|---|---|---|---|
| Generating tokens (128) | **+1.1 %** | **+0.7 %** | **+0.7 %** |
| Reading a 2048-token prompt | -1.5 % | -2.8 % | -3.5 % |
| Perplexity, host RAM / shared | 6.1281 / 6.1281 | 6.1309 / 6.1309 | 6.1305 / 6.1305 |
| Driver lines in the kernel log | 0 | 0 | 0 |

Not the +18 % on prompts the same change gave on a Tesla V100 + RTX 3090
([volta-dmabuf.md V6](../../docs/volta-dmabuf.md#v6-llamacpp-tensor-parallel-with-an-all-reduce-through-shared-video-memory));
why the gain does not show on this pair was not investigated. Perplexity is identical between the two routes on each
driver and differs in the fourth decimal between drivers.

Tensor parallel itself is worth it here for chat: **19.2 tokens/s against 13.5** with layer split (+42 %). Layer split
reads long prompts faster (about 615 against 475 tokens/s for 2048 tokens).

Two problems turned up on the way, neither caused by memory sharing:

- **Hundreds of driver lines in the kernel log** while the model loads: the driver's static BAR1 leaves only 4 GB of the
  RTX 3060's 16 GB window for Vulkan. Fix: `RMForceStaticBar1=0` for every card, and on driver 595 not per card
  ([troubleshooting](../../docs/troubleshooting.md#vulkan-the-kernel-log-fills-with-dmaallocmapping_gm107-cant-alloc-va-space)).
  Speeds do not change with the fix.
- **llama-bench crashes at exit on driver 615** (exit status 139, after printing its results) when the shader cache is
  cold, in NVIDIA's Vulkan driver; a 30-line change to llama.cpp avoids it ([below](#llama-bench-crashes-at-exit-on-driver-615)).

## Test setup

| | |
|---|---|
| GPU 0 | RTX 2060 6 GB (TU104), B550 chipset slot, PCIe 3.0 x4, 8 GB window ([turing.md](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/blob/main/docs/turing.md) T1-T3) |
| GPU 1 | RTX 3060 12 GB (GA106), CPU slot, PCIe 4.0 x16, 16 GB window (Resizable BAR) |
| Memory sharing | DRMShare `nvidia-drm` loaded by hand: the 580 version on 580, the 615 version on 595 and 615 ([dmabuf-drivers.md](../../docs/dmabuf-drivers.md)); before each Qwen run the driver test of [dmabuf-615.md](../../docs/dmabuf-615.md) D3 passed (1 GiB each way byte-exact, both ways 5.61-5.63 GB/s, 10 x 1 GiB soak, no driver line in the kernel log) |
| Static BAR1 | off for both cards (`RMForceStaticBar1=0`): on 580 for all cards at once (`NVreg_RegistryDwords`), on 615 per card, on 595 per card with `nvidia.ko` rebuilt with 615's version of the per-card lookup ([why](../../docs/dmabuf-drivers.md#llamacpp-on-each-driver)). The `-static-bar1-on` folders are the runs before that, with static BAR1 still on for the RTX 3060 |
| Host | Ryzen 9 3900X, 8 GB RAM, Ubuntu 26.04.1, kernel 7.0.0-44, Ubuntu's `nvidia-driver-*-open` packages |
| llama.cpp | `tools/llama.cpp` at `ab88444` (upstream `0ce7f213` + the shared-memory all-reduce), Vulkan build; the exit fix `4edab03` came after these runs |
| Model | [Qwen3.8-27B UD-Q3_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) (12.2 GiB, sha256 `7b20f067...`), split 1:2 (`-ts 1,2`: the 2060 holds a third, the 3060 two thirds) |

Why this model: with tensor parallel both cards hold their share of every layer, so the 6 GB card sets the limit. A
third of a 13 GB file fits the RTX 2060 with room for the KV cache; the Q8_0 of the same model (29 GB) does not fit the
pair at all.

## Results

`summary.md` in the results folder, made by `summarize.py`; tokens per second, mean over the rounds:

| Driver | Setting | Prompt, 512 | Prompt, 2048 | Generation, 128 |
|---|---|---|---|---|
| 580.178.04 open | tensor parallel, host RAM | 473 | 475 | 19.16 |
| | tensor parallel, shared video memory | 467 | 467 | 19.37 |
| | layer split | 391 | 615 | 13.51 |
| 595.99.02 | tensor parallel, host RAM | 475 | 474 | 19.19 |
| | tensor parallel, shared video memory | 461 | 461 | 19.32 |
| | layer split | 398 | 611 | 13.66 |
| 615.71.09 | tensor parallel, host RAM | 464 | 483 | 18.88 |
| | tensor parallel, shared video memory | 467 | 466 | 19.01 |
| | layer split | 405 | 618 | 13.49 |

The 512-token prompt has a warm-up spread in the first round after a model load (for example `445.64 ± 57.80` on 615);
the 2048-token prompt and generation repeat within 1 % between rounds.

**Chat through llama-server** (tensor parallel, shared video memory, `-c 4096`): 19.0 tokens/s on every driver. The
server log shows `all-reduce writes straight into peer VRAM` with a 2 MiB receive buffer in each card. The **first**
request after a driver change is slow (prompt about 1 token/s, 9.5-9.8 tokens/s generating): the driver compiles the
Vulkan pipelines the server needs and caches them on disk. That is why the two `-static-bar1-on` folders show 9.5
and 9.7; a second server on 595 with static BAR1 still on gave 18.9 (`chat-second-server-*.json`).

**Static BAR1 on vs off** (580 and 595, same numbers within 1 %): the fix removes the kernel log lines and changes no
speed.

## llama-bench crashes at exit on driver 615

On 615, tensor-parallel `llama-bench` with the full settings (`-p 512,2048 -n 128 -r 3`) ended with exit status 139
(a segfault) in 5 of 11 runs, always after it had printed all its results. It never happened on 580 or 595, nor with
short settings (`-p 64 -n 16 -r 1`: 30 runs, the small Qwen3-0.6B and this model, one card or both), and
`llama-perplexity` exited normally every time. The numbers above are complete. Under gdb:

```
Thread 3 "llama-bench" received signal SIGSEGV, Segmentation fault.
#0  ?? () from /usr/lib/x86_64-linux-gnu/libnvidia-glvkspirv.so.615.71.09
#1  ?? () from /usr/lib/x86_64-linux-gnu/libnvidia-eglcore.so.615.71.09
...
Thread 1:
#6  __dlclose (handle=<optimized out>) at ./dlfcn/dlclose.c:31
#7  ?? () from /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0
#9  __run_exit_handlers (status=0, ...) at ./stdlib/exit.c:118
```

At `exit()`, an exit handler of NVIDIA's `libEGL_nvidia.so.0` unloads the driver's libraries while one of the
driver's own threads is still running in its shader compiler (`libnvidia-glvkspirv`). Both sides are NVIDIA's closed
Vulkan driver.

**It needs a cold shader cache.** With a fresh, empty NVIDIA shader cache for every run
(`__GL_SHADER_DISK_CACHE_PATH=<empty directory>`), each run compiles its pipelines (14 MB) as the first runs after a
driver change do, and the unfixed `llama-bench` crashed in 3 of 4 runs. With the cache warm from earlier runs it
did not crash (0 of 3): the driver's compiler is then idle at exit. That is why it showed up in the first runs on 615.

**A workaround in llama.cpp:** [`ggml-vulkan-release-devices-at-exit.diff`](ggml-vulkan-release-devices-at-exit.diff),
30 lines on `tools/llama.cpp`. ggml-vulkan never destroys its Vulkan devices: each device's buffer type holds a
reference back to the device, and ggml's globals would be torn down only after NVIDIA's handler anyway
(`libggml-vulkan` is loaded first, and exit handlers run in reverse order). The change registers an exit handler once
the first device exists, so after the driver is loaded and before the driver's own handler runs. It waits for each
device, drops ggml's references back to it, and so destroys the pipelines and the device while the driver is still
there (`GGML_VK_EXIT_DEBUG=1` prints each device it releases, with 1 reference left). Same test, cold cache: crashed
in 0 of 4 runs (the two builds alternated), same speed (18.9 tokens/s generating, both).

It is in the llama.cpp fork as [`4edab03`](https://github.com/Cinderella-Man/miner-llama/commit/4edab03eff0937fc5d3923b8b63ec52d0baa1d24), the second commit on `vk-allreduce-shared-vram`, and `tools/llama.cpp`
is pinned to it, so `git submodule update --init --depth 1 tools/llama.cpp` brings it. The diff here is the same
change, for another llama.cpp tree.

## Run it

```bash
cd ~/how-to-enable-p2p-on-nvidia-gpus
git submodule update --init --depth 1 tools/llama.cpp
cmake -S tools/llama.cpp -B tools/llama.cpp/build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build tools/llama.cpp/build-vk -j6 --target llama-bench llama-perplexity llama-server
cat README.md docs/*.md > /tmp/guide-text.txt
sudo MODEL=/path/to/Qwen3.8-27B-UD-Q3_K_XL.gguf TEXT=/tmp/guide-text.txt benchmarks/llama/run.sh
benchmarks/llama/summarize.py benchmarks/llama/results/local/<date>-<model>
```

Load the DRMShare `nvidia-drm` first ([dmabuf-615.md](../../docs/dmabuf-615.md) D1-D2), with the display manager
stopped. Without it the shared-memory runs fall back to host RAM, and their log has no `all-reduce writes straight
into peer VRAM` line. `run.sh` takes the split from each card's memory, counts the driver's kernel log lines per step
when it runs as root, and takes about 15 minutes for this model (the first run after a driver change compiles shaders
for a few minutes more). The perplexity text here was the P2P guide's own text (its README and docs), 72 KB; any text works, as long as both
settings read the same file.

| File | What |
|---|---|
| `run.sh` | perplexity (both all-reduce routes), llama-bench (two rounds of each route, then layer split), two chat requests through llama-server |
| `summarize.py` | one Markdown table from one or more result folders |
| `results/local/` | where `run.sh` writes, ignored by git: the raw logs hold file paths and the host name |
| `results/2026-11-25-rtx2060-rtx3060/` | one folder per driver, picked from such runs: `bench-*.md` (llama-bench), `ppl-*.txt` (the perplexity log's key lines), `chat*.json` (answer and timings), `devices.txt` |
