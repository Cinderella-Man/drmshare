#!/usr/bin/env python3
"""Build docs/assets/infographic.html and infographic-turing.html (1600x900) from the benchmark numbers below.

Render to PNG (Firefox is a snap on Ubuntu, so keep the files under $HOME):
  python3 make_infographic.py
  firefox --headless --no-remote --profile <empty dir> --window-size=1600,900 \
          --screenshot infographic.png "file://$PWD/infographic.html"
  (the same for infographic-turing)
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
        ("8 GB + 4 GB", "memory windows, up from 256 MB"),
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


def fmt(v, unit):
    return f"{v:.1f} {unit}" if unit == "µs" else f"{v:.2f} {unit}"


def groups(gs):
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
            <div class="track"><div class="bar" style="width:{w_off:.2f}%;background:{GRAY}"></div><span class="val">{fmt(off, g["unit"])}</span></div>
            <div class="track"><div class="bar" style="width:{w_on:.2f}%;background:{GREEN}"></div><span class="val strong">{fmt(on, g["unit"])}</span></div>
          </div></div>''')
        out.append(f'<div class="group">{head}{"".join(rows)}</div>')
    return "".join(out)


def kpis(ks):
    for n, l in ks:
        if n is None or l is None:
            raise SystemExit(f"missing KPI: {n!r} {l!r}")
    return "".join(f'\n    <div class="kpi"><div class="n">{n}</div><div class="l">{l}</div></div>' for n, l in ks)


def page(d):
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{d["page_title"]}</title>
<!-- Generated by make_infographic.py; edit the numbers there, not here. -->
<style>
  :root {{ --bg:#07090a; --panel:#0c100e; --line:#1c2621; --base:#2a332e; --text:#eef3ef; --muted:#9aa69f; --dim:#6b7670; --accent:#76d63a; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  html, body {{ width:1600px; height:900px; overflow:hidden; background:var(--bg); }}
  body {{ position:relative; font-family:"Ubuntu Sans","Noto Sans","DejaVu Sans",sans-serif; color:var(--text); }}
  .bg {{ position:absolute; inset:0; background:
      radial-gradient(700px 420px at 85% 8%, rgba(118,214,58,.12), transparent 70%),
      radial-gradient(600px 400px at 0% 100%, rgba(118,214,58,.06), transparent 70%), var(--bg); }}
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
  .foot b {{ color:var(--muted); font-weight:500; }}
</style></head>
<body><div class="bg"></div>
<div class="wrap">
  <div class="eyebrow">{d["eyebrow"]}</div>
  <h1>{d["h1"]}</h1>
  <div class="sub">{d["sub"]}</div>
  <div class="kpis">{kpis(d["kpis"])}
  </div>
  <div class="legend">
    <span><span class="sw" style="background:{GRAY}"></span>Without P2P: through the computer's main memory</span>
    <span><span class="sw" style="background:{GREEN}"></span>With P2P: straight to the other GPU</span>
  </div>
  <div class="charts">
    <div class="panel">
      <div class="ptitle">Delay between the GPUs</div>
      <div class="psub">microseconds, lower is better · NVIDIA tools</div>
      <div class="plot">{groups(d["latency"])}</div>
    </div>
    <div class="panel">
      <div class="ptitle">How fast the GPUs exchange data</div>
      <div class="psub">{d["nccl_psub"]}</div>
      <div class="plot">{groups(d["nccl"])}</div>
    </div>
  </div>
  <div class="foot">
    <span>{d["foot"]}</span>
    <span><b>Cinderella-Man</b>/how-to-enable-p2p-on-nvidia-gpus</span>
  </div>
</div></body></html>
'''


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["rtx30", "turing"]
    for name, data, fname in [("rtx30", DATA, "infographic.html"), ("turing", TURING, "infographic-turing.html")]:
        if name in which:
            out = Path(__file__).with_name(fname)
            out.write_text(page(data))
            print(f"wrote {out}")
