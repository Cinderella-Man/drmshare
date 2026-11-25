#!/usr/bin/env python3
"""Summarize run.sh result folders into one Markdown table: mean tokens per second over the rounds, per setting.

usage: summarize.py <result folder> [<result folder> ...]
Each folder holds bench-tensor-hostram-r*.md, bench-tensor-shared-r*.md and bench-layer.md (llama-bench -o md) and,
when run.sh ran in full, ppl-*.log (or ppl-*.txt extracts) and chat*.json.
"""
import glob
import json
import os
import re
import sys

TESTS = ["pp512", "pp2048", "tg128"]


def bench(path):
    """{test: t/s} from one llama-bench Markdown table."""
    out = {}
    for line in open(path):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) > 2 and cells[-2] in TESTS:
            out[cells[-2]] = float(cells[-1].split("±")[0])
    return out


def mean(files):
    runs = [bench(f) for f in files]
    runs = [r for r in runs if r]
    if not runs:
        return None
    return {t: sum(r[t] for r in runs) / len(runs) for t in TESTS if all(t in r for r in runs)}, len(runs)


def ppl(folder, label):
    for ext in ("log", "txt"):  # .txt: the lines of the full log that matter (results/ keeps only those)
        try:
            m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", open(f"{folder}/ppl-{label}.{ext}").read())
            if m:
                return f"{m.group(1)} ± {m.group(2)}"
        except OSError:
            pass
    return "-"


def chat(folder):
    for name in ("chat2.json", "chat.json"):
        try:
            t = json.load(open(f"{folder}/{name}"))["timings"]
            return f"{t['predicted_per_second']:.1f}"
        except (OSError, KeyError, ValueError):
            pass
    return "-"


print("| Driver | Setting | Prompt, 512 | Prompt, 2048 | Generation, 128 | Rounds | Perplexity |")
print("|---|---|---|---|---|---|---|")
for folder in sys.argv[1:]:
    name = os.path.basename(folder.rstrip("/"))
    host = mean(sorted(glob.glob(f"{folder}/bench-tensor-hostram-r*.md")))
    shared = mean(sorted(glob.glob(f"{folder}/bench-tensor-shared-r*.md")))
    layer = mean([f"{folder}/bench-layer.md"])
    for label, res, p in (("tensor-parallel, host RAM", host, ppl(folder, "tensor-hostram")),
                          ("tensor-parallel, shared video memory", shared, ppl(folder, "tensor-shared")),
                          ("layer split", layer, "")):
        if res is None:
            continue
        r, n = res
        print(f"| {name} | {label} | {r['pp512']:.0f} | {r['pp2048']:.0f} | {r['tg128']:.2f} | {n} | {p} |")
    if host and shared:
        h, s = host[0], shared[0]
        d = " / ".join(f"{(s[t] / h[t] - 1) * 100:+.1f} %" for t in TESTS)
        print(f"| {name} | shared vs host RAM | {d.replace(' / ', ' | ')} | | |")
    print(f"| {name} | chat through llama-server, generation | | | {chat(folder)} | | |")
