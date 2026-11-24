#!/usr/bin/env python3
"""Build docs/assets/infographic*.html (1600x900: RTX 30, Turing, Pascal, two generations) from the numbers below.

Render to PNG (Firefox is a snap on Ubuntu, so keep the files under $HOME):
  python3 make_infographic.py
  firefox --headless --no-remote --profile <empty dir> --window-size=1600,900 \
          --screenshot infographic.png "file://$PWD/infographic.html"
  (the same for infographic-turing, -pascal and -mixed)
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
}


def fmt(v, unit):
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


def legend(t):
    return f'''  <div class="legend">
    <span><span class="sw" style="background:{t["bars"][0]}"></span>Without P2P: through the computer's main memory</span>
    <span><span class="sw" style="background:{t["bars"][1]}"></span>With P2P: straight to the other GPU</span>
  </div>'''


def charts(d, t):
    """Default body: four KPI tiles, the legend, then a delay panel and a bandwidth panel."""
    return f'''  <div class="kpis">{kpis(d["kpis"])}
  </div>
{legend(t)}
  <div class="charts">
    <div class="panel">
      <div class="ptitle">Delay between the GPUs</div>
      <div class="psub">{d.get("latency_psub", "microseconds, lower is better · NVIDIA tools")}</div>
      <div class="plot">{groups(d["latency"], t["bars"])}</div>
    </div>
    <div class="panel">
      <div class="ptitle">How fast the GPUs exchange data</div>
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
    return f'''{legend(t)}
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
    extra_css = (CREDITS_CSS if "credits" in d else "") + (ROUTES_CSS if "routes" in d else "")
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
  <div class="sub">{d["sub"]}</div>{credits(d)}
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


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["rtx30", "turing", "pascal", "mixed"]
    for name, data, fname in [("rtx30", DATA, "infographic.html"), ("turing", TURING, "infographic-turing.html"),
                              ("pascal", PASCAL, "infographic-pascal.html"),
                              ("mixed", MIXED, "infographic-mixed.html")]:
        if name in which:
            out = Path(__file__).with_name(fname)
            out.write_text(page(data))
            print(f"wrote {out}")
