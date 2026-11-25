| Driver | Setting | Prompt, 512 | Prompt, 2048 | Generation, 128 | Rounds | Perplexity |
|---|---|---|---|---|---|---|
| 580.178.04-open | tensor-parallel, host RAM | 473 | 475 | 19.16 | 2 | 6.1281 ± 0.33467 |
| 580.178.04-open | tensor-parallel, shared video memory | 467 | 467 | 19.37 | 2 | 6.1281 ± 0.33467 |
| 580.178.04-open | layer split | 391 | 615 | 13.51 | 1 |  |
| 580.178.04-open | shared vs host RAM | -1.2 % | -1.5 % | +1.1 % | | |
| 580.178.04-open | chat through llama-server, generation | | | 19.0 | | |
| 595.99.02 | tensor-parallel, host RAM | 475 | 474 | 19.19 | 2 | 6.1309 ± 0.33472 |
| 595.99.02 | tensor-parallel, shared video memory | 461 | 461 | 19.32 | 2 | 6.1309 ± 0.33472 |
| 595.99.02 | layer split | 398 | 611 | 13.66 | 1 |  |
| 595.99.02 | shared vs host RAM | -2.9 % | -2.8 % | +0.7 % | | |
| 595.99.02 | chat through llama-server, generation | | | 19.0 | | |
| 615.71.09 | tensor-parallel, host RAM | 464 | 483 | 18.88 | 2 | 6.1305 ± 0.33486 |
| 615.71.09 | tensor-parallel, shared video memory | 467 | 466 | 19.01 | 2 | 6.1305 ± 0.33486 |
| 615.71.09 | layer split | 405 | 618 | 13.49 | 1 |  |
| 615.71.09 | shared vs host RAM | +0.8 % | -3.5 % | +0.7 % | | |
| 615.71.09 | chat through llama-server, generation | | | 19.0 | | |
| 580.178.04-open-static-bar1-on | tensor-parallel, host RAM | 458 | 477 | 19.16 | 2 | 6.1281 ± 0.33467 |
| 580.178.04-open-static-bar1-on | tensor-parallel, shared video memory | 466 | 467 | 19.31 | 2 | 6.1281 ± 0.33467 |
| 580.178.04-open-static-bar1-on | layer split | 406 | 622 | 13.47 | 1 |  |
| 580.178.04-open-static-bar1-on | shared vs host RAM | +1.9 % | -2.0 % | +0.8 % | | |
| 580.178.04-open-static-bar1-on | chat through llama-server, generation | | | 9.5 | | |
| 595.99.02-static-bar1-on | tensor-parallel, host RAM | 476 | 475 | 19.12 | 2 | 6.1309 ± 0.33472 |
| 595.99.02-static-bar1-on | tensor-parallel, shared video memory | 461 | 460 | 19.24 | 2 | 6.1309 ± 0.33472 |
| 595.99.02-static-bar1-on | layer split | 396 | 611 | 13.53 | 1 |  |
| 595.99.02-static-bar1-on | shared vs host RAM | -3.3 % | -3.1 % | +0.7 % | | |
| 595.99.02-static-bar1-on | chat through llama-server, generation | | | 9.7 | | |
