#!/usr/bin/env python3
"""Build docs/assets/infographic*.html (1600x900: RTX 30, Turing, Pascal, two generations, V100 + 3090) from the numbers below.

Render to PNG (Firefox is a snap on Ubuntu, so keep the files under $HOME):
  python3 make_infographic.py
  firefox --headless --no-remote --profile <empty dir> --window-size=1600,900 \
          --screenshot infographic.png "file://$PWD/infographic.html"
  (the same for infographic-turing, -pascal, -mixed and -volta)
"""
from pathlib import Path

# Numbers from docs/benchmarks.md.
# Latency: mean of both directions, GPU-timed (p2pBandwidthLatencyTest), microseconds.
# NCCL: bus bandwidth at 1 GiB, GB/s.
DATA = {
    "latency": [
        {"title": "Copying data from one GPU to the other",
         "desc": "time until a small copy arrives, both directions averaged (p2pBandwidthLatencyTest)",
         "max": 20, "unit": "µs", "rows": [
            ("RTX 3090 pair", 18.9, 1.5),
            ("RTX 3080 pair", 19.0, 1.4),
        ]},
        {"title": "One sync step with a small message",
         "desc": "an 8 KB add-up step, as on the right (NCCL all-reduce); software overhead dominates",
         "max": 20, "unit": "µs", "rows": [
            ("RTX 3090 pair", 19.5, 14.6),
            ("RTX 3080 pair", 19.2, 13.5),
        ]},
    ],
    "nccl": [
        {"title": "Syncing a model split across both GPUs",
         "desc": "both GPUs add up their partial results, in every layer (NCCL all-reduce)",
         "max": 3, "unit": "GB/s", "rows": [
            ("RTX 3090 pair", 1.88, 2.75),
            ("RTX 3080 pair", 1.85, 2.72),
        ]},
        {"title": "Collecting each other's data",
         "desc": "each GPU ends up with a full copy of the other's data (NCCL all-gather)",
         "max": 3, "unit": "GB/s", "rows": [
            ("RTX 3090 pair", 1.61, 2.74),
            ("RTX 3080 pair", 1.57, 2.71),
        ]},
    ],
    "kpis": [
        ("13×", "less delay between the GPUs"),
        ("+46–74%", "faster data exchange between the GPUs"),
        ("0 errors", "in 1.5 TB of GPU-to-GPU test copies, on each pair"),
    ],
    "page_title": "P2P results infographic",
    "eyebrow": "UBUNTU 26.04 · PATCHED OPEN DRIVER 615.71.09",
    "h1": 'GPU-to-GPU <span class="a">P2P</span> on GeForce RTX 30 cards',
    "sub": "Tested on 2× RTX 3090 and an RTX 3080 10 GB + 12 GB, with Resizable BAR windows of 32 GB and 16 GB. Same results on both.",
    "nccl_psub": "GB/s, higher is better · NVIDIA NCCL, 1 GB at a time",
    "foot": "One card sits in a PCIe x4 chipset slot, which caps absolute speeds. Faster data exchange is not the same as a faster model.",
}

# Turing pair, docs/benchmarks.md#turing-pair-rtx-2060--gtx-1650-super. NCCL at 512 MiB (the largest size a 4 GB card
# runs); the RTX 3080 pair at 512 MiB too, from the same runs as above.
TURING = {
    "latency": [
        {"title": "Copying data from one GPU to the other",
         "desc": "time until a small copy arrives, both directions averaged (p2pBandwidthLatencyTest)",
         "max": 30, "unit": "µs", "rows": [
            ("Turing pair", 29.3, 1.45),
            ("RTX 3080 pair", 19.0, 1.4),
        ]},
        {"title": "One sync step with a small message",
         "desc": "an 8 KB add-up step, as on the right (NCCL all-reduce); software overhead dominates",
         "max": 30, "unit": "µs", "rows": [
            ("Turing pair", 20.4, 13.6),
            ("RTX 3080 pair", 19.2, 13.5),
        ]},
    ],
    "nccl": [
        {"title": "Syncing a model split across both GPUs",
         "desc": "both GPUs add up their partial results, in every layer (NCCL all-reduce)",
         "max": 3, "unit": "GB/s", "rows": [
            ("Turing pair", 1.89, 2.74),
            ("RTX 3080 pair", 1.85, 2.72),
        ]},
        {"title": "Collecting each other's data",
         "desc": "each GPU ends up with a full copy of the other's data (NCCL all-gather)",
         "max": 3, "unit": "GB/s", "rows": [
            ("Turing pair", 1.53, 2.73),
            ("RTX 3080 pair", 1.56, 2.69),
        ]},
    ],
    "kpis": [
        ("16–32×", "bigger window, covers all VRAM"),
        ("20×", "less delay between the GPUs"),
        ("+34–78%", "faster data exchange"),
        ("0 errors", "in 1.3 TB of test copies"),
    ],
    "page_title": "Turing P2P infographic",
    "eyebrow": "UBUNTU 26.04 · PATCHED OPEN DRIVER 615.71.09 · EXPERIMENTAL",
    "h1": 'GPU-to-GPU <span class="a">P2P</span> on an RTX 2060 + GTX 1650 SUPER',
    "sub": "Two different Turing chips (TU104 + TU116), in the same slots as the RTX 3080 pair, and just as fast.",
    "nccl_psub": "GB/s, higher is better · NVIDIA NCCL, 512 MB at a time",
    "foot": "Experimental: CUDA needs a small shim to accept two different chips. One card sits in a PCIe x4 chipset slot, which caps absolute speeds.",
}

GRAY, GREEN = "#626c67", "#58ad2a"   # validated pair (dataviz validator, dark surface #0c100e)

# Page themes. "bars" is (without P2P, with P2P). Light: bars validated against the white panel (the gray is a deliberate
# neutral, so it reads as gray; every bar carries its value as text); text colors at least 4.5:1 on the background.
THEMES = {
    "dark": {"vars": "--bg:#07090a; --panel:#0c100e; --line:#1c2621; --base:#2a332e; --text:#eef3ef; --muted:#9aa69f; "
                     "--dim:#6b7670; --accent:#76d63a;",
             "glow": ("rgba(118,214,58,.12)", "rgba(118,214,58,.06)"), "bars": (GRAY, GREEN)},
    "light": {"vars": "--bg:#f3f4ef; --panel:#ffffff; --line:#dfe3da; --base:#c4cbc1; --text:#0f1411; --muted:#4b544e; "
                      "--dim:#5f6862; --accent:#367a10;",
              "glow": ("rgba(74,148,32,.16)", "rgba(74,148,32,.08)"), "bars": ("#8f9893", "#4a9420")},
    # Navy: bars validated against the navy panel (both >= 3:1; the gray is the deliberate neutral).
    "navy": {"vars": "--bg:#080d18; --panel:#0f1626; --line:#1e2a42; --base:#2c3a58; --text:#eef2f8; --muted:#9ba7bf; "
                     "--dim:#7a87a1; --accent:#76d63a;",
             "glow": ("rgba(88,173,42,.13)", "rgba(70,110,210,.14)"), "bars": ("#5d6880", "#58ad2a")},
    # Slate: a mid tone, neither dark nor white. Bars validated against the slate panel (both >= 3:1, CVD-separated;
    # the gray is the deliberate neutral); text >= 4.5:1 on panel and background.
    "slate": {"vars": "--bg:#3c4960; --panel:#46546c; --line:#5a6983; --base:#75839b; --text:#f5f7fb; --muted:#d3dae5; "
                      "--dim:#bcc5d3; --accent:#8fe356;",
              "glow": ("rgba(143,227,86,.10)", "rgba(150,185,240,.16)"), "bars": ("#a3adbf", "#7fdc44")},
}


def fmt(v, unit):
    if unit == "t/s":
        return f"{v:,.0f} {unit}"
    return f"{v:.1f} {unit}" if unit == "µs" else f"{v:.2f} {unit}"


def groups(gs, bars):
    gray, green = bars
    out = []
    for g in gs:
        head = ""
        if g["title"]:
            head = f'<div class="gtitle">{g["title"]}</div><div class="gdesc">{g["desc"]}</div>'
        rows = []
        for label, off, on in g["rows"]:
            if off is None or on is None:
                raise SystemExit(f"missing data for {label}")
            w_off, w_on = off / g["max"] * 100, on / g["max"] * 100
            rows.append(f'''
          <div class="row"><div class="rlabel">{label}</div><div class="tracks">
            <div class="track"><div class="bar" style="width:{w_off:.2f}%;background:{gray}"></div><span class="val">{fmt(off, g["unit"])}</span></div>
            <div class="track"><div class="bar" style="width:{w_on:.2f}%;background:{green}"></div><span class="val strong">{fmt(on, g["unit"])}</span></div>
          </div></div>''')
        out.append(f'<div class="group">{head}{"".join(rows)}</div>')
    return "".join(out)


def kpis(ks):
    for n, l in ks:
        if n is None or l is None:
            raise SystemExit(f"missing KPI: {n!r} {l!r}")
    return "".join(f'\n    <div class="kpi"><div class="n">{n}</div><div class="l">{l}</div></div>' for n, l in ks)


def credits(d):
    """Optional strip right under the subtitle: whose work this stands on, then what this guide added."""
    if "credits" not in d:
        return ""
    items = "".join(f'<span><b>{who}</b> {what}</span>' for who, what in d["credits"])
    return (f'\n  <div class="credits"><span class="ck">BUILT ON</span><div class="ci">{items}</div>'
            f'<span class="ck">ADDED HERE</span><div class="ci"><span>{d["added"]}</span></div></div>')


def families(d):
    """Optional strip: the GPU families a driver supports, each verified (with its card) or not tested."""
    if "families" not in d:
        return ""
    chips = "".join(f'<div class="fam {"ok" if card else "no"}"><b>{name}</b><span>{card or "not tested"}</span></div>'
                    for name, card in d["families"])
    return f'\n  <div class="fams"><span class="ck">{d.get("families_label", "FAMILIES")}</span><div class="fl">{chips}</div></div>'


FAMILY_CSS = '''
  .fams { margin-top:14px; display:grid; grid-template-columns:124px 1fr; align-items:center; }
  .fl { display:flex; gap:8px; }
  .fam { flex:1; min-width:0; border:1px solid var(--line); border-radius:10px; padding:6px 10px; background:var(--panel); }
  .fam b { display:block; font-size:16px; font-weight:700; color:var(--text); }
  .fam span { display:block; font-size:13px; color:var(--dim); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .fam.ok { border-color:var(--accent); }
  .fam.ok b::before { content:"✓ "; color:var(--accent); }
  .fam.no b { color:var(--dim); }'''


def legend(t, d):
    off, on = d.get("legend", ("Without P2P: through the computer's main memory", "With P2P: straight to the other GPU"))
    return f'''  <div class="legend">
    <span><span class="sw" style="background:{t["bars"][0]}"></span>{off}</span>
    <span><span class="sw" style="background:{t["bars"][1]}"></span>{on}</span>
  </div>'''


def charts(d, t):
    """Default body: four KPI tiles, the legend, then a delay panel and a bandwidth panel."""
    return f'''  <div class="kpis">{kpis(d["kpis"])}
  </div>
{legend(t, d)}
  <div class="charts">
    <div class="panel">
      <div class="ptitle">{d.get("latency_ptitle", "Delay between the GPUs")}</div>
      <div class="psub">{d.get("latency_psub", "microseconds, lower is better · NVIDIA tools")}</div>
      <div class="plot">{groups(d["latency"], t["bars"])}</div>
    </div>
    <div class="panel">
      <div class="ptitle">{d.get("nccl_ptitle", "How fast the GPUs exchange data")}</div>
      <div class="psub">{d["nccl_psub"]}</div>
      <div class="plot">{groups(d["nccl"], t["bars"])}</div>
    </div>
  </div>'''


def routes(d, t):
    """Split body: the legend, then one half per route, each led by the technique it uses."""
    halves = []
    for r in d["routes"]:
        tiles = "".join(f'<div class="kpi"><div class="n">{n}</div><div class="l">{l}</div></div>' for n, l in r["kpis"])
        halves.append(f'''
    <div class="route">
      <div class="rtag">{r["tag"]}</div>
      <div class="rtech">{r["tech"]}</div>
      <div class="rhow">{r["how"]}</div>
      <div class="rkpis">{tiles}</div>
      <div class="plot">{groups(r["groups"], t["bars"])}</div>
    </div>''')
    return f'''{legend(t, d)}
  <div class="routes">{"".join(halves)}
  </div>'''


ROUTES_CSS = '''
  .routes { display:flex; gap:16px; margin-top:12px; flex:1; min-height:0; }
  .route { flex:1; background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:16px 22px 14px;
           display:flex; flex-direction:column; }
  .rtag { font-family:"Ubuntu Sans Mono","Noto Sans Mono",monospace; font-size:14px; letter-spacing:.08em; color:var(--accent); }
  .rtech { margin-top:4px; font-size:25px; font-weight:800; letter-spacing:-.01em; line-height:1.15; }
  .rhow { margin-top:6px; font-size:16px; color:var(--muted); line-height:1.35; }
  .rkpis { display:flex; gap:12px; margin-top:10px; }
  .rkpis .kpi { background:var(--bg); padding:8px 16px; }
  .rkpis .kpi .n { font-size:30px; }
  .rkpis .kpi .l { font-size:16px; margin-top:0; }
  .legend { margin-top:16px; }
  .route .plot { margin-top:6px; gap:14px; }
  .route .group { gap:8px; }
  .route .gdesc { margin-top:-6px; }'''


CREDITS_CSS = '''
  .credits { margin-top:18px; display:grid; grid-template-columns:124px 1fr; gap:8px 0; align-items:baseline;
             font-size:18px; color:var(--muted); }
  .ci { display:flex; flex-wrap:wrap; gap:4px 26px; }
  .ck { font-family:"Ubuntu Sans Mono","Noto Sans Mono",monospace; font-size:14px; letter-spacing:.08em; color:var(--accent); }
  .ci b { color:var(--text); font-weight:700; }'''


def page(d):
    t = THEMES[d.get("theme", "dark")]
    extra_css = ((CREDITS_CSS if "credits" in d else "") + (ROUTES_CSS if "routes" in d else "")
                 + (FAMILY_CSS if "families" in d else "") + d.get("extra_css", ""))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{d["page_title"]}</title>
<!-- Generated by make_infographic.py; edit the numbers there, not here. -->
<style>
  :root {{ {t["vars"]} }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  html, body {{ width:1600px; height:900px; overflow:hidden; background:var(--bg); }}
  body {{ position:relative; font-family:"Ubuntu Sans","Noto Sans","DejaVu Sans",sans-serif; color:var(--text); }}
  .bg {{ position:absolute; inset:0; background:
      radial-gradient(700px 420px at 85% 8%, {t["glow"][0]}, transparent 70%),
      radial-gradient(600px 400px at 0% 100%, {t["glow"][1]}, transparent 70%), var(--bg); }}
  .wrap {{ position:absolute; inset:56px 64px 40px 64px; display:flex; flex-direction:column; }}
  .eyebrow {{ font-family:"Ubuntu Sans Mono","Noto Sans Mono",monospace; font-size:18px; letter-spacing:.08em; color:var(--accent); }}
  h1 {{ margin-top:12px; font-size:54px; line-height:1.05; font-weight:800; letter-spacing:-.02em; }}
  h1 .a {{ color:var(--accent); }}
  .sub {{ margin-top:10px; font-size:22px; color:var(--muted); }}
  .kpis {{ display:flex; gap:16px; margin-top:28px; }}
  .kpi {{ flex:1; background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:16px 20px; }}
  .kpi .n {{ font-size:42px; font-weight:800; letter-spacing:-.02em; }}
  .kpi .l {{ margin-top:4px; font-size:18px; color:var(--muted); }}
  .legend {{ display:flex; gap:28px; margin-top:26px; font-size:17px; color:var(--muted); align-items:center; }}
  .sw {{ display:inline-block; width:14px; height:14px; border-radius:3px; margin-right:8px; vertical-align:-2px; }}
  .charts {{ display:flex; gap:16px; margin-top:12px; flex:1; }}
  .panel {{ flex:1; background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:18px 22px 16px; display:flex; flex-direction:column; }}
  .ptitle {{ font-size:20px; font-weight:700; }}
  .psub {{ margin-top:2px; font-size:15px; color:var(--dim); }}
  .plot {{ flex:1; display:flex; flex-direction:column; justify-content:space-around; margin-top:8px; }}
  .group {{ display:flex; flex-direction:column; gap:10px; }}
  .gtitle {{ font-size:17px; font-weight:700; color:var(--text); }}
  .gdesc {{ font-size:15px; color:var(--muted); margin-top:-8px; }}
  .row {{ display:grid; grid-template-columns:140px 1fr; align-items:center; padding-right:96px; }}
  .rlabel {{ font-size:16px; color:var(--muted); }}
  .tracks {{ display:flex; flex-direction:column; gap:2px; border-left:1px solid var(--base); }}
  .track {{ height:20px; display:flex; align-items:center; }}
  .bar {{ height:20px; border-radius:0 4px 4px 0; }}
  .val {{ margin-left:8px; font-size:16px; color:var(--muted); white-space:nowrap; }}
  .val.strong {{ color:var(--text); font-weight:700; }}
  .foot {{ display:flex; justify-content:space-between; margin-top:14px; font-size:15px; color:var(--dim); }}
  .foot b {{ color:var(--muted); font-weight:500; }}{extra_css}
</style></head>
<body><div class="bg"></div>
<div class="wrap">
  <div class="eyebrow">{d["eyebrow"]}</div>
  <h1>{d["h1"]}</h1>
  <div class="sub">{d["sub"]}</div>{credits(d)}{families(d)}
{routes(d, t) if "routes" in d else charts(d, t)}
  <div class="foot">
    <span>{d["foot"]}</span>
    <span><b>Cinderella-Man</b>/how-to-enable-p2p-on-nvidia-gpus</span>
  </div>
</div></body></html>
'''


# Pascal pair: two P106-090 in the test rig, NVIDIA's stock driver 580 (docs/pascal-and-volta.md). No NCCL run, so the
# panels use p2pBandwidthLatencyTest only: latency GPU- and CPU-timed (mean of both directions), bandwidth both ways
# and one way (mean of both directions).
PASCAL = {
    "latency": [
        {"title": "Copying data from one GPU to the other",
         "desc": "time until a small copy arrives, timed on the GPU, both directions averaged",
         "max": 25, "unit": "µs", "rows": [
            ("P106-090 pair", 22.3, 3.1),
        ]},
        {"title": "The same copy, as the program sees it",
         "desc": "time per copy measured by the calling program, launch overhead included",
         "max": 25, "unit": "µs", "rows": [
            ("P106-090 pair", 5.7, 1.7),
        ]},
    ],
    "nccl": [
        {"title": "Both GPUs sending to each other at once",
         "desc": "where P2P helps: each direction crosses the slow links once instead of twice",
         "max": 1.6, "unit": "GB/s", "rows": [
            ("P106-090 pair", 1.09, 1.48),
        ]},
        {"title": "One GPU sending to the other",
         "desc": "capped by the cards' own PCIe 1.0 x4 link either way, both directions averaged",
         "max": 1.6, "unit": "GB/s", "rows": [
            ("P106-090 pair", 0.78, 0.84),
        ]},
    ],
    "kpis": [
        ("0 patches", "NVIDIA's own driver 580"),
        ("7×", "less delay between the GPUs"),
        ("+36%", "faster two-way data exchange"),
        ("0 errors", "in 1.5 TB of test copies"),
    ],
    "page_title": "Pascal P2P infographic",
    "eyebrow": "UBUNTU 26.04 · NVIDIA DRIVER 580, UNPATCHED · PASCAL",
    "h1": 'GPU-to-GPU <span class="a">P2P</span> on P106-090 mining cards',
    "sub": "Two Pascal mining cards (GP106) on NVIDIA's last Pascal driver, no patch, no override. Every byte checked.",
    "nccl_psub": "GB/s, higher is better · p2pBandwidthLatencyTest",
    "foot": "Needs IOMMU passthrough and a board that routes peer traffic (a mining board did not). The cards' PCIe 1.0 x4 link caps bandwidth.",
}


# Two generations: RTX 2060 (04:00.0) + RTX 3060 (2b:00.0), docs/mixed-generations.md. Light theme, one half per route,
# the same tiles and charts in both. Copies and 4-byte delays: medians of a p2p_iomem soak (one direction at a time,
# 10 runs per direction at 1 GiB with the libcuda copy, which times route 1 and route 2 in the same run; 5 per direction
# at 64 MiB without it for host RAM). Checked data: route 1 from the integrity test, route 2 from that soak (62 GiB).
_COPY = {"title": "Copying from one GPU to the other",
         "desc": "GB/s, higher is better · one direction at a time",
         "max": 3.5, "unit": "GB/s"}
_DELAY = {"title": "A 4-byte copy from one GPU to the other",
          "desc": "microseconds, lower is better · timed on the GPU",
          "max": 20, "unit": "µs"}
MIXED = {
    "theme": "light",
    "routes": [
        {"tag": "ROUTE 1 · CUDA PEER ACCESS",
         "tech": "Patched driver + a patched copy of libcuda",
         "how": "CUDA treats the pair like two cards of one generation, so NCCL, PyTorch and vLLM use it unchanged "
                "(NCCL all-reduce +43%). 4 bytes changed in a copy; runs as a normal user.",
         "kpis": [("0 errors", "in 1.4 TiB of checked copies"),
                  ("7–9×", "less delay on a 4-byte copy")],
         "groups": [
             dict(_DELAY, rows=[("2060 → 3060", 18.0, 2.07), ("3060 → 2060", 13.5, 2.06)]),
             dict(_COPY, rows=[("2060 → 3060", 2.78, 3.28), ("3060 → 2060", 2.48, 2.93)]),
         ]},
        {"tag": "ROUTE 2 · I/O MEMORY",
         "tech": "Patched driver + the other card's memory window",
         "how": "Vulkan maps one card's VRAM through its BAR1; CUDA on the other card registers it as I/O memory and "
                "writes straight in. Only programs written for it; root and one opt-in fork key.",
         "kpis": [("0 errors", "in 62 GiB of checked copies"),
                  ("8–10×", "less delay on a 4-byte copy")],
         "groups": [
             dict(_DELAY, rows=[("2060 → 3060", 18.0, 1.78), ("3060 → 2060", 13.5, 1.77)]),
             dict(_COPY, rows=[("2060 → 3060", 2.78, 3.28), ("3060 → 2060", 2.48, 2.93)]),
         ]},
    ],
    "credits": [
        ("aikitoria", "libcuda patch for mixed generations"),
        ("noon-at-cgn", "P2P driver and BAR1 resize"),
        ("tinygrad", "original GeForce P2P patch"),
        ("NVIDIA", "open kernel modules"),
    ],
    "added": "<b>The USB-C step:</b> the 2060's own USB-C functions blocked its memory window. Removed, it grows from "
             "256 MB to 8 GB, enough for the driver to allow P2P.",
    "page_title": "Two generations P2P infographic",
    "eyebrow": "UBUNTU 26.04 · PATCHED OPEN DRIVER 615.71.09 · TURING + AMPERE · EXPERIMENTAL",
    "h1": 'GPU-to-GPU <span class="a">P2P</span> across two GPU generations',
    "sub": "An RTX 2060 (Turing) and an RTX 3060 (Ampere), which CUDA normally refuses to pair. Two routes, every byte checked.",
    "foot": "Both routes hit the same ceiling: the 2060's PCIe 3.0 x4 chipset slot. Experimental: one pair, one board, one driver.",
}


# Tesla V100 + RTX 3090: memory sharing through a dma-buf (docs/volta-dmabuf.md). One method, navy theme.
# Copies and 4-byte delays: dma-buf soak medians (20 x 1 GiB) vs CUDA's own cudaMemcpyPeer through system memory
# (staged_baseline.cu). Model: llama.cpp tensor-parallel Qwen3.8-27B Q8_0, upstream all-reduce through system memory vs
# the all-reduce through shared video memory, llama-bench 3 reps (split by layer noted in the descriptions).
VOLTA = {
    "theme": "navy",
    # a little less height for the tiles, so the two charts per panel get space between them
    "extra_css": "\n  .kpis { margin-top:20px; } .kpi { padding:12px 20px; } .kpi .n { font-size:38px; }"
                 "\n  .legend { margin-top:18px; } .plot { gap:14px; }",
    "latency": [
        {"title": "A 4-byte copy from one GPU to the other",
         "desc": "time per copy, timed on the GPU · medians of 20 runs",
         "max": 20, "unit": "µs", "rows": [
            ("3090 → V100", 17.0, 2.39),
            ("V100 → 3090", 19.5, 3.11),
        ]},
        {"title": "One generated token, Qwen3.8-27B in llama.cpp",
         "desc": "tensor-parallel, an all-reduce after every layer; split by layer: 42.9 ms",
         "max": 45, "unit": "ms", "rows": [
            ("both cards", 35.8, 34.6),
        ]},
    ],
    "nccl": [
        {"title": "Copying from one GPU to the other",
         "desc": "GB/s, one direction at a time; both ways at once: 5.62 GB/s",
         "max": 3.5, "unit": "GB/s", "rows": [
            ("3090 → V100", 2.50, 2.96),
            ("V100 → 3090", 2.79, 3.26),
        ]},
        {"title": "Reading a prompt, Qwen3.8-27B in llama.cpp",
         "desc": "tokens per second, tensor-parallel; split by layer: 504 at 2048",
         "max": 500, "unit": "t/s", "rows": [
            ("512 tokens", 347, 410),
            ("2048 tokens", 347, 412),
        ]},
    ],
    "kpis": [
        ("0 errors", "20 × 1 GiB, every word checked"),
        ("6–7×", "less delay on a small copy"),
        ("5.62 GB/s", "both ways at once, like CUDA P2P"),
        ("+18%", "faster prompts, Qwen3.8-27B"),
    ],
    "credits": [
        ("Christian König / amdgpu", "dma-buf technique"),
        ("NVIDIA", "open nvidia-drm"),
        ("pwilkin", "llama.cpp all-reduce"),
        ("TheBlueMatt", "first direct Vulkan all-reduce"),
    ],
    "added": "<b>Here:</b> one opt-in nvidia-drm option (127 lines) and a llama.cpp all-reduce through shared video "
             "memory (233 lines). No NVIDIA binary modified.",
    "legend": ("Through system memory (what CUDA and upstream llama.cpp do)",
               "Shared video memory: straight into the other card"),
    "page_title": "V100 + 3090 memory sharing infographic",
    "eyebrow": "UBUNTU 26.04 · NVIDIA DRIVER 580 · VOLTA + AMPERE · EXPERIMENTAL",
    "h1": 'Memory <span class="a">sharing</span>: Tesla V100 + RTX 3090',
    "sub": "Each card copies straight into the other's video memory through a dma-buf, on NVIDIA's own driver, which "
           "reports the pair as not supported.",
    "latency_ptitle": "Delay",
    "latency_psub": "lower is better",
    "nccl_ptitle": "Speed",
    "nccl_psub": "higher is better",
    "foot": "Not CUDA P2P: CUDA, NCCL and vLLM can't use it, Vulkan programs can. The V100's PCIe 3.0 x4 slot caps every path.",
}


# Four generations around one Tesla V100 (docs/volta-dmabuf.md): memory sharing through a dma-buf, slate theme.
# Copies and 4-byte delays: soak medians (20 runs per pair) vs CUDA's cudaMemcpyPeer through system memory
# (staged_baseline.cu). "Into the V100": the other card writes into the V100's memory (1 GiB); "from the V100": the V100
# writes into the other card's memory (1 GiB; the GTX 1070 Ti's fixed 256 MB window allows 192 MiB).
GENERATIONS = {
    "theme": "slate",
    "extra_css": "\n  .wrap { inset:46px 64px 32px 64px; } h1 { font-size:50px; } .sub { margin-top:8px; }"
                 "\n  .kpis { margin-top:14px; } .kpi { padding:8px 18px; } .kpi .n { font-size:30px; }"
                 " .kpi .l { font-size:16px; margin-top:0; }"
                 "\n  .legend { margin-top:12px; } .charts { margin-top:10px; } .panel { padding:14px 22px 12px; }"
                 "\n  .plot { margin-top:6px; } .group { gap:5px; } .gdesc:empty { display:none; }"
                 "\n  .track, .bar { height:16px; } .val { font-size:15px; } .foot { margin-top:10px; }",
    "latency": [
        {"title": "From the other card into the V100's memory",
         "desc": "",
         "max": 23, "unit": "µs", "rows": [
            ("RTX 3090", 17.0, 2.39),
            ("RTX 2060", 19.9, 2.15),
            ("GTX 1070 Ti", 18.4, 1.51),
        ]},
        {"title": "From the V100 into the other card's memory",
         "desc": "",
         "max": 23, "unit": "µs", "rows": [
            ("RTX 3090", 19.5, 3.11),
            ("RTX 2060", 21.5, 3.18),
            ("GTX 1070 Ti", 20.4, 2.17),
        ]},
    ],
    "nccl": [
        {"title": "From the other card into the V100's memory",
         "desc": "",
         "max": 3.6, "unit": "GB/s", "rows": [
            ("RTX 3090", 2.50, 2.96),
            ("RTX 2060", 2.27, 2.91),
            ("GTX 1070 Ti", 2.82, 2.96),
        ]},
        {"title": "From the V100 into the other card's memory",
         "desc": "",
         "max": 3.6, "unit": "GB/s", "rows": [
            ("RTX 3090", 2.79, 3.26),
            ("RTX 2060", 2.51, 3.28),
            ("GTX 1070 Ti", 3.22, 3.28),
        ]},
    ],
    "kpis": [
        ("4 generations", "Pascal, Volta, Turing, Ampere"),
        ("0 errors", "60 soak runs, every word checked"),
        ("5.6 GB/s", "both ways at once, every pair"),
        ("6–12×", "less delay on a small copy"),
    ],
    "credits": [
        ("Christian König / amdgpu", "dma-buf technique"),
        ("NVIDIA", "open nvidia-drm"),
        ("noon-at-cgn", "geforce-bar1-resize (the 2060's 8 GB window)"),
    ],
    "added": "<b>Here:</b> one opt-in nvidia-drm option (127 lines), and the USB-C step that lets the 2060's window grow. "
             "No NVIDIA binary modified.",
    "families_label": "DRIVER 580",
    "families": [
        ("Maxwell", None),
        ("Pascal", "GTX 1070 Ti"),
        ("Volta", "Tesla V100"),
        ("Turing", "RTX 2060"),
        ("Ampere", "RTX 3090"),
        ("Ada", None),
        ("Hopper", None),
        ("Blackwell", None),
    ],
    "legend": ("Through system memory (what CUDA does for these pairs)",
               "Shared video memory: straight into the other card"),
    "page_title": "Four generations memory sharing infographic",
    "eyebrow": "UBUNTU 26.04 · NVIDIA DRIVER 580 · FOUR GENERATIONS · EXPERIMENTAL",
    "h1": 'Memory <span class="a">sharing</span> across four GPU generations',
    "sub": "A Tesla V100 shares video memory with a GTX 1070 Ti, an RTX 2060 and an RTX 3090: pairs CUDA refuses to connect.",
    "latency_ptitle": "Delay",
    "latency_psub": "a 4-byte copy timed on the GPU · microseconds, lower is better",
    "nccl_ptitle": "Speed",
    "nccl_psub": "GB/s, one direction at a time · higher is better",
    "foot": "Not CUDA P2P: Vulkan programs only. The V100's PCIe 3.0 x4 chipset slot caps every pair.",
}


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["rtx30", "turing", "pascal", "mixed", "volta", "generations"]
    for name, data, fname in [("rtx30", DATA, "infographic.html"), ("turing", TURING, "infographic-turing.html"),
                              ("pascal", PASCAL, "infographic-pascal.html"),
                              ("mixed", MIXED, "infographic-mixed.html"),
                              ("volta", VOLTA, "infographic-volta.html"),
                              ("generations", GENERATIONS, "infographic-generations.html")]:
        if name in which:
            out = Path(__file__).with_name(fname)
            out.write_text(page(data))
            print(f"wrote {out}")
