"""Behaviour cloning on privileged-expert demonstrations (Baseline A).

Losses:
  * cross-entropy on the six factorised action heads
  * auxiliary heads: keys held, enemy visible, door ahead, exit visible,
    exit bearing (13 classes), exit distance (6 bins), key colours visible,
    damage over the next 8 decisions
  * Monte-Carlo value head on the environment reward

Usage:
    python -m training.bc --data training/data/demos_all --epochs 6
"""

from __future__ import annotations

import argparse
import json
import os
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from submission.model import (AUX_DAMAGE, AUX_DOOR, AUX_ENEMY, AUX_EXIT_BEARING,
                              AUX_EXIT_DIST, AUX_EXIT_VIS, AUX_KEY_COLOURS, AUX_KEYS,
                              HEAD_SIZES, DoomNet, vector_to_actions)


def load_episodes(data_dir: str | Path, min_steps: int = 8) -> list[dict]:
    dirs = [Path(p) for p in str(data_dir).split(",") if p.strip()]
    eps = []
    rows = []
    for d in dirs:
        for line in (d / "manifest.jsonl").read_text().splitlines():
            row = json.loads(line)
            row["_dir"] = str(d)
            rows.append(row)
    for row in rows:
        path = Path(row["_dir"]) / row["file"]
        if not path.exists():
            continue
        with np.load(path) as z:
            ep = {k: z[k] for k in z.files}
        if len(ep["action"]) < min_steps:
            continue
        ep["meta"] = row
        ep["heads"] = np.stack([np.array(vector_to_actions(a), np.int64) for a in ep["action"]])
        if "map" not in ep:
            from submission.mapmem import MapBuilder
            n = len(ep["action"])
            builder = MapBuilder()
            maps = np.zeros((n, 5, 32, 32), np.uint8)
            poses = np.zeros((n, 6), np.float32)
            prev = None
            execs = ep["exec"] if "exec" in ep else ep["action"]
            for t in range(n):
                obs = {"rays": {"distance": ep["dist"][t], "hit_type": ep["hit"][t]},
                       "visible_entities": {"features": ep["efeat"][t], "entity_type": ep["etype"][t],
                                            "mask": ep["emask"][t]},
                       "vitals": ep["vitals"][t]}
                builder.update(obs, prev)
                maps[t] = builder.crop()
                poses[t] = builder.pose()
                prev = execs[t]
            ep["map"] = maps
            ep["pose"] = poses
        eps.append(ep)
    return eps


def window_episodes(eps: list[dict], window: int = 128, stride: int = 64) -> list[dict]:
    """Split long episodes into overlapping windows so BC batches stay small."""
    if window <= 0:
        return eps
    out = []
    for ep in eps:
        n = len(ep["action"])
        for start in range(0, max(1, n), stride):
            end = min(n, start + window)
            if end - start < 8:
                continue
            w = {k: v[start:end] for k, v in ep.items()
                 if isinstance(v, np.ndarray) and len(v) == n}
            w["meta"] = ep["meta"]
            out.append(w)
    return out


def collate(batch: list[dict], device: str) -> dict:
    T = max(len(ep["action"]) for ep in batch)
    B = len(batch)
    out = {
        "rays_dist": torch.zeros(B, T, 64),
        "rays_hit": torch.zeros(B, T, 64, dtype=torch.long),
        "ent_feat": torch.zeros(B, T, 32, 5),
        "ent_type": torch.zeros(B, T, 32, dtype=torch.long),
        "ent_mask": torch.zeros(B, T, 32),
        "vitals": torch.zeros(B, T, 17),
        "rgb": torch.zeros(B, T, 3, 30, 40),
        "prev_action": torch.zeros(B, T, 12),
        "map": torch.zeros(B, T, 5, 32, 32),
        "pose": torch.zeros(B, T, 6),
        "heads": torch.zeros(B, T, 6, dtype=torch.long),
        "keys": torch.zeros(B, T, 6),
        "aux": torch.zeros(B, T, 35),
        "bear": torch.full((B, T), -100, dtype=torch.long),
        "dist": torch.full((B, T), -100, dtype=torch.long),
        "reward": torch.zeros(B, T),
        "mask": torch.zeros(B, T),
    }
    for i, ep in enumerate(batch):
        n = len(ep["action"])
        out["rays_dist"][i, :n] = torch.from_numpy(ep["dist"].astype(np.float32))
        out["rays_hit"][i, :n] = torch.from_numpy(ep["hit"].astype(np.int64))
        out["ent_feat"][i, :n] = torch.from_numpy(ep["efeat"].astype(np.float32))
        out["ent_type"][i, :n] = torch.from_numpy(ep["etype"].astype(np.int64))
        out["ent_mask"][i, :n] = torch.from_numpy(ep["emask"].astype(np.float32))
        out["vitals"][i, :n] = torch.from_numpy(ep["vitals"].astype(np.float32))
        out["rgb"][i, :n] = torch.from_numpy(ep["rgb"].astype(np.float32)) / 255.0
        out["prev_action"][i, :n] = torch.from_numpy(ep["prev"].astype(np.float32))
        out["map"][i, :n] = torch.from_numpy(ep["map"].astype(np.float32))
        out["pose"][i, :n] = torch.from_numpy(ep["pose"].astype(np.float32))
        out["heads"][i, :n] = torch.from_numpy(ep["heads"])
        out["keys"][i, :n] = torch.from_numpy(ep["keys"].astype(np.float32))
        out["reward"][i, :n] = torch.from_numpy(ep["reward"].astype(np.float32))
        out["mask"][i, :n] = 1.0
        ev = torch.from_numpy(ep["exit_visible"].astype(np.float32))
        out["aux"][i, :n, AUX_KEYS] = out["keys"][i, :n]
        enemy = (((ep["etype"] == 1) | (ep["etype"] == 2) | (ep["etype"] == 3))
                 & (ep["emask"] > 0)).any(axis=1).astype(np.float32)
        door = np.isin(ep["hit"][:, 0], (2, 3)).astype(np.float32)
        out["aux"][i, :n, AUX_ENEMY] = torch.from_numpy(enemy)
        out["aux"][i, :n, AUX_DOOR] = torch.from_numpy(door)
        out["aux"][i, :n, AUX_EXIT_VIS] = ev
        out["aux"][i, :n, AUX_KEY_COLOURS] = torch.from_numpy(ep["key_colours"].astype(np.float32))
        out["aux"][i, :n, AUX_DAMAGE] = torch.from_numpy(ep["damage8"].astype(np.float32))
        vis = ev > 0.5
        b = torch.from_numpy(ep["exit_bearing"].astype(np.int64))
        d = torch.from_numpy(ep["exit_dist"].astype(np.int64))
        out["bear"][i, :n] = torch.where(vis, b, torch.full_like(b, -100))
        out["dist"][i, :n] = torch.where(vis, d, torch.full_like(d, -100))
    gamma = 0.995
    ret = torch.zeros(B, T)
    running = torch.zeros(B)
    for t in range(T - 1, -1, -1):
        running = out["reward"][:, t] + gamma * running * (out["mask"][:, t] > 0)
        ret[:, t] = running
    out["returns"] = ret
    return {k: v.to(device) for k, v in out.items()}


def batch_loss(net: DoomNet, batch: dict, anchor: DoomNet | None = None,
               anchor_weight: float = 0.0) -> tuple[torch.Tensor, dict]:
    emb = net.encode({k: batch[k] for k in
                      ("rays_dist", "rays_hit", "ent_feat", "ent_type", "ent_mask", "vitals", "rgb", "prev_action", "map", "pose")})
    logits, value, aux, _ = net.forward_seq(emb)
    m = batch["mask"]
    denom = m.sum().clamp(min=1.0)
    ce = 0.0
    front = batch["rays_hit"][..., [0, 1, 2, 62, 63]]
    door_ahead = ((front == 2) | (front == 3)).any(-1).float()
    for j, part in enumerate(torch.split(logits, HEAD_SIZES, dim=-1)):
        ce_j = F.cross_entropy(part.reshape(-1, part.shape[-1]),
                               batch["heads"][..., j].reshape(-1), reduction="none")
        w = m.reshape(-1)
        if j == 4:  # USE: rare positives, critical for doors
            w = w * (1.0 + 10.0 * door_ahead.reshape(-1) + 5.0 * batch["heads"][..., 4].reshape(-1).float())
        ce = ce + (ce_j * w).sum() / w.sum().clamp(min=1.0)

    def bce(idx, target, pos_weight=1.0):
        logit = aux[..., idx] if isinstance(idx, int) else aux[..., idx]
        b = F.binary_cross_entropy_with_logits(
            logit, target, reduction="none",
            pos_weight=torch.tensor(pos_weight, device=target.device))
        if b.ndim > 2:
            b = b.mean(-1)
        return (b * m).sum() / denom

    keys = bce(AUX_KEYS, batch["aux"][..., AUX_KEYS], 4.0)
    enemy = bce(AUX_ENEMY, batch["aux"][..., AUX_ENEMY], 2.0)
    door = bce(AUX_DOOR, batch["aux"][..., AUX_DOOR], 4.0)
    evis = bce(AUX_EXIT_VIS, batch["aux"][..., AUX_EXIT_VIS], 12.0)
    kcol = bce(AUX_KEY_COLOURS, batch["aux"][..., AUX_KEY_COLOURS], 6.0)
    bear = F.cross_entropy(aux[..., AUX_EXIT_BEARING].reshape(-1, 13),
                           batch["bear"].reshape(-1), ignore_index=-100, reduction="none")
    bear = (bear * m.reshape(-1)).sum() / denom
    dist = F.cross_entropy(aux[..., AUX_EXIT_DIST].reshape(-1, 6),
                           batch["dist"].reshape(-1), ignore_index=-100, reduction="none")
    dist = (dist * m.reshape(-1)).sum() / denom
    dmg = F.mse_loss(aux[..., AUX_DAMAGE], batch["aux"][..., AUX_DAMAGE], reduction="none")
    dmg = (dmg * m).sum() / denom
    aux_loss = keys + enemy + door + evis + kcol + bear + dist + dmg
    v_loss = (F.mse_loss(value[..., 0], batch["returns"], reduction="none") * m).sum() / denom
    loss = ce + 0.2 * aux_loss + 0.5 * v_loss

    kl = torch.zeros((), device=loss.device)
    if anchor is not None and anchor_weight > 0:
        with torch.no_grad():
            a_emb = anchor.encode({k: batch[k] for k in
                                   ("rays_dist", "rays_hit", "ent_feat", "ent_type", "ent_mask",
                                    "vitals", "rgb", "prev_action", "map", "pose")})
            a_logits, _, _, _ = anchor.forward_seq(a_emb)
        # preserve the base policy on non-door frames; let *any* visible door frame change
        door_any = ((batch["rays_hit"] == 2) | (batch["rays_hit"] == 3)).any(-1).float()
        keep = ((1.0 - door_any) * m)
        kl_terms = []
        for cur, ref in zip(torch.split(logits, HEAD_SIZES, dim=-1), torch.split(a_logits, HEAD_SIZES, dim=-1)):
            kl_terms.append(F.kl_div(F.log_softmax(cur, -1), F.softmax(ref, -1),
                                     reduction="none").sum(-1))
        kl = (torch.stack(kl_terms, 0).sum(0) * keep).sum() / keep.sum().clamp(min=1.0)
        loss = loss + anchor_weight * kl
    return loss, {"ce": ce.detach(), "aux": aux_loss.detach(), "value": v_loss.detach(),
                  "exit": evis.detach(), "bear": bear.detach(), "kl": kl.detach()}


def evaluate_quick(net: DoomNet, device: str, n_maps: int = 4, seed: int = 10_000) -> dict:
    from training.evaluate import run_agent_episode
    from training.maps.gen_wad import generate_map
    net.eval()
    res = []
    for i in range(n_maps):
        spec = generate_map(seed + i, tier=2 + (i % 3), out_dir="training/data/eval_maps",
                            no_doors=True, no_monsters=True)
        r = run_agent_episode(net, str(spec.path), "MAP01", device=device, ticks=4200)
        res.append(r)
    net.train()
    return {"exit_rate": float(np.mean([r["exit"] for r in res])),
            "median_steps": float(np.median([r["steps"] for r in res])),
            "proxy": float(np.mean([r["proxy"] for r in res]))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="training/data/demos_all")
    ap.add_argument("--out", default="training/runs/bc_a")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", default="")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--drill-repeat", type=int, default=3)
    ap.add_argument("--freeze", default="", choices=("", "head", "trunk"),
                    help="head: train actor only; trunk: freeze encoders+GRU")
    ap.add_argument("--anchor", default="", help="frozen checkpoint for a KL anchor")
    ap.add_argument("--anchor-weight", type=float, default=0.5)
    args = ap.parse_args()

    torch.set_num_threads(min(8, os.cpu_count() or 4))
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    raw = load_episodes(args.data)
    if args.drill_repeat > 1:
        drills = [e for e in raw if e.get("meta", {}).get("tag") == "drill"]
        raw = raw + drills * (args.drill_repeat - 1)
    eps = window_episodes(raw, args.window, args.stride)
    if not eps:
        raise SystemExit(f"no episodes in {args.data}")
    rng = np.random.default_rng(0)
    rng.shuffle(eps)
    n_val = max(1, int(len(eps) * args.val_frac))
    val, train = eps[:n_val], eps[n_val:]
    print(f"{len(train)} train / {len(val)} val episodes, device={device}")

    net = DoomNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
        print("resumed", args.resume)
    anchor = None
    if args.anchor:
        a_ck = torch.load(args.anchor, map_location=device)
        anchor = DoomNet(hidden=a_ck.get("config", {}).get("hidden", 384))
        anchor.load_state_dict(a_ck["model"] if "model" in a_ck else a_ck)
        anchor.to(device).eval()
        for p_ in anchor.parameters():
            p_.requires_grad_(False)
        print("anchor", args.anchor)
    if args.freeze == "head":
        for name, p_ in net.named_parameters():
            p_.requires_grad_(name.startswith("actor"))
        print("frozen: all but actor")
    elif args.freeze == "trunk":
        for name, p_ in net.named_parameters():
            p_.requires_grad_(name.startswith("actor") or name.startswith("critic") or name.startswith("aux"))
        print("frozen: encoders + GRU + trunk")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = math.ceil(len(train) / args.batch)
    for epoch in range(args.epochs):
        t0 = time.time()
        by_len = sorted(range(len(train)), key=lambda i: len(train[i]["action"]))
        chunks = [by_len[i:i + args.batch] for i in range(0, len(by_len), args.batch)]
        order_chunks = rng.permutation(len(chunks))
        totals = {"ce": 0.0, "aux": 0.0, "value": 0.0, "loss": 0.0, "exit": 0.0, "bear": 0.0}
        n = 0
        for b in range(steps_per_epoch):
            idx = chunks[order_chunks[b]]
            batch = collate([train[i] for i in idx], device)
            loss, parts = batch_loss(net, batch, anchor, args.anchor_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            totals["loss"] += float(loss.detach())
            for k, v in parts.items():
                if k in totals:
                    totals[k] += float(v)
            n += 1
            if (b + 1) % 10 == 0:
                print(f"  epoch {epoch} {b + 1}/{steps_per_epoch} "
                      f"loss={totals['loss'] / n:.3f} ce={totals['ce'] / n:.3f} "
                      f"aux={totals['aux'] / n:.3f} v={totals['value'] / n:.3f}", flush=True)
        val_loss = None
        if val:
            net.eval()
            with torch.no_grad():
                vb = []
                for i in range(0, len(val), args.batch):
                    batch = collate(val[i:i + args.batch], device)
                    l, _ = batch_loss(net, batch, anchor, args.anchor_weight)
                    vb.append(float(l))
                val_loss = float(np.mean(vb))
            net.train()
        metrics = evaluate_quick(net, device, seed=10_000)
        torch.save({"model": net.state_dict(), "config": {"hidden": net.hidden},
                    "epoch": epoch, "val_loss": val_loss, "metrics": metrics}, out / "ckpt.pt")
        torch.save({"model": net.state_dict(), "config": {"hidden": net.hidden}},
                   out / f"ckpt_e{epoch}.pt")
        row = {"epoch": epoch, "loss": totals["loss"] / n, "ce": totals["ce"] / n,
               "aux": totals["aux"] / n, "value": totals["value"] / n,
               "val_loss": val_loss, **metrics, "seconds": time.time() - t0}
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"epoch {epoch} done in {time.time() - t0:.0f}s train={totals['loss'] / n:.3f} "
              f"val={val_loss if val_loss is None else round(val_loss, 3)} eval={metrics}", flush=True)
    print("saved", out / "ckpt.pt")


if __name__ == "__main__":
    main()
