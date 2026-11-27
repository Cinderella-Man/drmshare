#!/usr/bin/env python3
# Give a DFlash / DFlash2 draft GGUF its own token embeddings and output head, copied byte for byte from the target
# GGUF, so the draft references no target tensor. Drafts without them borrow the target's through ctx_other; with a
# tensor-split target (-sm tensor) the borrowed head is split by vocabulary across the cards and the DFlash2 graph's
# ggml_top_k on its logits aborts in ggml-backend-meta.cpp:578 (llama.cpp-drmshare 74ee0f3 = upstream 0ce7f213). With
# its own head the draft can run on one card (-devd), as src/models/dflash.cpp's comment says.
# Pure Python (no numpy): the draft's header and data are kept as they are, the two tensors are appended.
# usage: dflash_add_head.py <draft.gguf> <target.gguf> <out.gguf> [tensor names... default token_embd.weight output.weight]
import os, struct, sys

# ggml type -> (elements per block, bytes per block), the types these files use
TYPE_SIZE = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
             10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
             16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82),
             23: (256, 136), 30: (1, 2)}
SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def rstr(f):
    return f.read(rd(f, "Q")).decode("utf-8")


def skip_val(f, t):
    if t in SC:
        f.read(struct.calcsize("<" + SC[t]))
    elif t == 8:
        rstr(f)
    else:
        at, n = rd(f, "I"), rd(f, "Q")
        for _ in range(n):
            skip_val(f, at)


def parse(path):
    """-> dict(version, nkv, kv_raw, align, tensors=[(name, dims, type, offset)], data_start)"""
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", path
        version, nt, nkv = rd(f, "I"), rd(f, "Q"), rd(f, "Q")
        kv_start, align = f.tell(), 32
        for _ in range(nkv):
            k, t = rstr(f), rd(f, "I")
            if k == "general.alignment":
                align = rd(f, "I")
            else:
                skip_val(f, t)
        kv_end = f.tell()
        f.seek(kv_start)
        kv_raw = f.read(kv_end - kv_start)
        tensors = []
        for _ in range(nt):
            name = rstr(f)
            nd = rd(f, "I")
            dims = [rd(f, "Q") for _ in range(nd)]
            tensors.append((name, dims, rd(f, "I"), rd(f, "Q")))
        data_start = (f.tell() + align - 1) // align * align
    return dict(version=version, nkv=nkv, kv_raw=kv_raw, align=align, tensors=tensors, data_start=data_start)


def nbytes(dims, typ):
    if typ not in TYPE_SIZE:
        raise SystemExit(f"ggml type {typ} not in the table")
    n = 1
    for d in dims:
        n *= d
    blk, sz = TYPE_SIZE[typ]
    assert n % blk == 0
    return n // blk * sz


def copy(src, dst, offset, size, chunk=64 << 20):
    src.seek(offset)
    while size:
        b = src.read(min(chunk, size))
        assert b, "short read"
        dst.write(b)
        size -= len(b)


def main():
    draft_p, target_p, out_p = sys.argv[1:4]
    names = sys.argv[4:] or ["token_embd.weight", "output.weight"]
    d, t = parse(draft_p), parse(target_p)
    have = {x[0] for x in d["tensors"]}
    for n in names:
        if n in have:
            raise SystemExit(f"{draft_p} already has {n}")
    tt = {x[0]: x for x in t["tensors"]}
    add = []
    for n in names:
        if n not in tt:
            raise SystemExit(f"{target_p} has no {n}")
        add.append(tt[n])
    A = d["align"]
    up = lambda x: (x + A - 1) // A * A
    draft_data = os.path.getsize(draft_p) - d["data_start"]
    off = up(draft_data)
    new = []
    for name, dims, typ, toff in add:
        new.append((name, dims, typ, off, toff, nbytes(dims, typ)))
        off = up(off + nbytes(dims, typ))
    with open(out_p + ".part", "wb") as o, open(draft_p, "rb") as df, open(target_p, "rb") as tf:
        o.write(b"GGUF" + struct.pack("<IQQ", d["version"], len(d["tensors"]) + len(new), d["nkv"]))
        o.write(d["kv_raw"])
        for name, dims, typ, toff in d["tensors"]:
            nb = name.encode()
            o.write(struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims)))
            o.write(b"".join(struct.pack("<Q", x) for x in dims) + struct.pack("<IQ", typ, toff))
        for name, dims, typ, noff, _, _ in new:
            nb = name.encode()
            o.write(struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims)))
            o.write(b"".join(struct.pack("<Q", x) for x in dims) + struct.pack("<IQ", typ, noff))
        o.write(b"\0" * (up(o.tell()) - o.tell()))
        base = o.tell()
        copy(df, o, d["data_start"], draft_data)
        for name, dims, typ, noff, toff, size in new:
            o.write(b"\0" * (base + noff - o.tell()))
            copy(tf, o, t["data_start"] + toff, size)
            print(f"{name}: {dims} type {typ}, {size} bytes from the target")
    os.rename(out_p + ".part", out_p)
    print(f"{out_p}: {len(d['tensors'])} + {len(new)} tensors, {os.path.getsize(out_p)} bytes")


if __name__ == "__main__":
    main()
