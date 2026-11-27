# Three GPU generations in one machine

DRMShare lets every card of a mixed set share memory with every other, on NVIDIA's closed driver 580, with no CUDA P2P
and no patched driver. This page shows it on one machine with three cards of three different generations at once, two
different sets, every copy checked word for word; then the NCCL fork's ring over all three, and llama.cpp across them.
Tested 2026-11-27.

## The machine

| | |
|---|---|
| Board, CPU | MSI MPG B550 GAMING PLUS, Ryzen 9 3900X, 8 GB RAM, 850 W supply |
| Slots | a passive **x16 -> 4 x x4 bifurcation splitter** (Gen 3, cables to four riser boards) in the CPU's x16 slot; the board's settings **PCI_E1 Lanes Configuration x4x4x4x4** and **PCI_E1 Gen Switch Gen3**. Each card sits on its own CPU port at PCIe 3.0 x4 (about 3.3 GB/s each way); nothing is behind the chipset |
| Driver | NVIDIA's closed 580.178.04 with DRMShare's `nvidia-drm` ([volta-dmabuf.md](volta-dmabuf.md)), the IOMMU in passthrough mode |
| CUDA | Ubuntu's CUDA 12.4 (CUDA 13 dropped Maxwell, Pascal and Volta), NCCL 2.32.6, the NCCL fork of [nccl.md](nccl.md), llama.cpp-drmshare |

Why the splitter: on the same board without it, two of the cards sat behind the B550 chipset, and **two cards behind
one chipset cannot reach each other at all**: every word is lost and nothing says so
([troubleshooting](troubleshooting.md#two-cards-behind-one-amd-chipset-the-copy-finishes-nothing-arrives)). With the
board's default settings the splitter's cards were not found at all (no picture, the VGA debug light on): set the two
options first with one card straight in the slot.

## Memory sharing: every pair, every word checked

[V4](volta-dmabuf.md#v4-test-each-card-writes-and-reads-the-others-memory)'s test on all six directions (64 MiB, each card
writing into and reading the other's memory), then all six directions at once for 60 seconds with random pauses:

| Set | Generations | Each direction, writes / reads | All six at once, 60 s | Errors |
|---|---|---|---|---|
| GTX 1650 SUPER + RTX 3090 + Tesla V100 | **Turing + Ampere + Volta** | 3.30 / 3.28-3.31 GB/s, 0 wrong | **196 GB checked, 0 wrong** | no Xid, no PCIe error |
| GTX 1070 Ti + RTX 3090 + Tesla V100 | **Pascal + Ampere + Volta** | 3.30 / 3.29-3.31 GB/s, 0 wrong | **203 GB checked, 0 wrong** | no Xid, no PCIe error |

Every direction runs at the x4 link's limit, reads as fast as writes. The GTX 1070 Ti shares buffers of up to about
224 MiB (its fixed 256 MB window); the others share everything.

## NCCL: one ring over three generations

The NCCL fork ([nccl.md](nccl.md)) builds one ring over the three cards, each card's CUDA device matched to its Vulkan
device (`DRMShare: CUDA device 0 = Vulkan NVIDIA GeForce GTX 1650 SUPER`, `1 = ... RTX 3090`, `2 = Tesla V100`), six
connections card to card (the same with `GTX 1070 Ti` as device 0 in the second set). Every result checked:

| Check | Turing + Ampere + Volta | Pascal + Ampere + Volta |
|---|---|---|
| Back-to-back all-reduces, card to card, 64 KiB-4 MiB x 3,000, f32 and bf16 (`readback`) | 0 wrong | 0 wrong |
| Through the two lanes, 4-16 MiB x 300, f32 and bf16 | 0 wrong | 0 wrong |
| The same with `release` ordering, f32 | 0 wrong | 0 wrong |
| `all_reduce_perf -g 3`, every result checked | 0 wrong; 2.38 GB/s stock, 1.53 the fork | 0 wrong; 2.34 GB/s stock, 1.52 the fork |

## llama.cpp: Qwen3.8-27B Q8_0 across all three

llama.cpp-drmshare, CUDA, tensor split (`-sm tensor`), the RTX 3090 capped at 250 W, tokens/s, 2048-token prompt /
generation; perplexity over 8 chunks of 512; `llama-server` with MTP 3 and with a DFlash2 draft, 5 requests at
temperature 0, answers compared byte for byte:

| | Turing + Ampere + Volta | Pascal + Ampere + Volta |
|---|---|---|
| stock NCCL (through host memory) | 460 / 30.8 | **756** / 31.4 |
| the NCCL fork (card to card) | 405 / 30.8 | 611 / 31.5 |
| perplexity, stock / fork | 5.9977 / 5.9977, every chunk identical | 5.9934 / 5.9934, every chunk identical |
| MTP 3, stock / fork | 49.9 / 49.9, answers 5 of 5 identical | 50.2 / 50.4, answers 5 of 5 identical |
| DFlash2 draft, stock / fork | 51.8 / 51.8, answers 5 of 5 identical | 51.0 / 51.0, answers 5 of 5 identical |
| for comparison: RTX 3090 + V100 alone, stock / fork | 838 / 32.2, 857 / 32.1 | 826 / 32.2, 881 / 32.2 |

## What it shows, and what it does not

- **It works**: three generations share memory pair by pair and in one NCCL ring, with not one wrong word or element
  in hundreds of gigabytes, and llama.cpp gives the same answers through the fork as through stock NCCL.
- **It is not faster.** A third, smaller card slows the split model down (each layer waits for its share: much less with
  the GTX 1070 Ti, 756 tokens/s, than with the GTX 1650 SUPER, 460), and with
  every card on its own CPU link NCCL's route through host memory beats the fork's card-to-card route: in a ring every
  card sends and receives at once, and all of that turns at the CPU. The fork pays off where a card's link to host
  memory is the narrow one ([nccl.md](nccl.md#three-generations-a-v100-a-gtx-1650-super-and-an-rtx-3090)).
- **Minutes, not hours**: each set ran for about half an hour of tests; the four-hour stress ran on three RTX 3090s
  ([nccl.md](nccl.md#three-rtx-3090s-on-driver-615-a-four-hour-stress)).
- **The board decides**: one card at most behind an AMD or Intel chipset; cards on a USB x1 riser may need their link
  slowed ([troubleshooting](troubleshooting.md#a-card-on-a-pcie-x1-riser-falls-off-the-bus-xid-79-when-the-driver-starts-it)).

The scripts behind these runs are DRMShare's tools (`p2p_dmabuf`, `stale_test`) and nccl-tests' `all_reduce_perf`; the
same set of cards on a board with two cards behind the chipset is described in
[volta-dmabuf.md](volta-dmabuf.md#v100--gtx-1650-super-volta--turing-tu116-three-cards-in-one-machine) and
[nccl.md](nccl.md#three-generations-a-v100-a-gtx-1650-super-and-an-rtx-3090).
