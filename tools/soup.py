"""Average two checkpoints (model soup) and save a candidate.

Usage:
    python -m tools.soup --a training/runs/bc_map2/ckpt.pt \
        --b training/runs/bc_use/ckpt.pt --alpha 0.6 --out training/runs/soup_06.pt
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of A (tier-1 policy)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a = torch.load(args.a, map_location="cpu")["model"]
    b = torch.load(args.b, map_location="cpu")["model"]
    out = {}
    for k in a:
        if k in b and a[k].shape == b[k].shape:
            out[k] = args.alpha * a[k].float() + (1 - args.alpha) * b[k].float()
        else:
            out[k] = a[k]
    torch.save({"model": out, "config": {"hidden": 384}}, args.out)
    print("wrote", args.out, "alpha", args.alpha)


if __name__ == "__main__":
    main()
