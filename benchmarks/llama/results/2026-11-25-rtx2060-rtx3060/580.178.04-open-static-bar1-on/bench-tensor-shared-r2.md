ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = NVIDIA GeForce RTX 2060 (NVIDIA) | uma: 0 | rebar: 1 | fp16: 1 | bf16: 1 | fp4: 0 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
ggml_vulkan: 1 = NVIDIA GeForce RTX 3060 (NVIDIA) | uma: 0 | rebar: 1 | fp16: 1 | bf16: 1 | fp4: 0 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
| model                          |       size |     params | backend    | ngl |     sm | ts           |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | ------------ | --------------: | -------------------: |
| qwen35 27B Q3_K - Large        |  12.23 GiB |    27.32 B | Vulkan     |  99 | tensor | 1.00/2.00    |           pp512 |        465.44 ± 2.10 |
| qwen35 27B Q3_K - Large        |  12.23 GiB |    27.32 B | Vulkan     |  99 | tensor | 1.00/2.00    |          pp2048 |        466.44 ± 0.53 |
| qwen35 27B Q3_K - Large        |  12.23 GiB |    27.32 B | Vulkan     |  99 | tensor | 1.00/2.00    |           tg128 |         19.29 ± 0.03 |

build: 0ce7f213b (12714)
