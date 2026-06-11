"""Generate the simulated event log and ground-truth user table.

Usage:
    python -m sim.generate --users 100000 --days 28 --out data/run1
"""
import argparse
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # box-drawing on cp1252 consoles

import numpy as np
import polars as pl

from . import archetypes as A
from .config import SimConfig
from .world import World

FLUSH_EVERY = 10_000  # users per event-frame flush


def _assign_archetypes(rng, cfg):
    """Returns list of (label,) per user index, shuffled."""
    counts = {k: int(cfg.n_users * v) for k, v in cfg.mix.items()}
    counts["normal"] = cfg.n_users - sum(v for k, v in counts.items() if k != "normal")
    labels = sum(([k] * v for k, v in counts.items()), [])
    rng.shuffle(labels)
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=20_000)
    ap.add_argument("--days", type=int, default=28)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/run")
    ap.add_argument("--evasion", type=float, default=0.0,
                    help="max attacker evasion level; each attacker entity "
                         "samples e ~ U(0, evasion)")
    args = ap.parse_args()

    cfg = SimConfig(n_users=args.users, days=args.days, seed=args.seed,
                    out_dir=args.out)
    rng = np.random.default_rng(cfg.seed)
    world = World(rng)
    os.makedirs(cfg.out_dir, exist_ok=True)

    labels = _assign_archetypes(rng, cfg)

    # pre-build coordination structures (rings/farms) for grouped archetypes
    bot_idx = [i for i, l in enumerate(labels) if l == "spam_bot"]
    farmer_idx = [i for i, l in enumerate(labels) if l == "account_farmer"]

    def sample_e():
        return float(rng.uniform(0, args.evasion)) if args.evasion > 0 else 0.0

    bot_farm_of, farms = {}, []
    i = 0
    while i < len(bot_idx):
        size = min(int(rng.integers(15, 60)), len(bot_idx) - i)
        farms.append(A.make_bot_farm(rng, world, cfg, e=sample_e()))
        for j in range(i, i + size):
            bot_farm_of[bot_idx[j]] = (len(farms) - 1, j - i)
        i += size

    ring_of, rings = {}, []
    i = 0
    while i < len(farmer_idx):
        size = min(int(rng.integers(10, 60)), len(farmer_idx) - i)
        rings.append(A.make_farmer_ring(rng, world, cfg, size, e=sample_e()))
        for j in range(i, i + size):
            ring_of[farmer_idx[j]] = (len(rings) - 1, j - i)
        i += size

    user_rows = []
    event_frames = []
    buf = {k: [] for k in ["user_id", *A.EVENT_COLS]}
    t_start = time.time()
    n_events_total = 0

    def flush():
        nonlocal buf, n_events_total
        if buf["user_id"]:
            n_events_total += len(buf["user_id"])
            event_frames.append(pl.DataFrame(buf))
            buf = {k: [] for k in ["user_id", *A.EVENT_COLS]}

    for i, label in enumerate(labels):
        uid = f"u{i:06d}"
        if label == "normal":
            user, ev = A.gen_normal(rng, world, cfg)
        elif label == "spam_bot":
            fi, _ = bot_farm_of[i]
            user, ev = A.gen_spam_bot(rng, world, cfg, farms[fi], f"farm{fi:04d}")
        elif label == "account_farmer":
            ri, idx = ring_of[i]
            user, ev = A.gen_farmer(rng, world, cfg, rings[ri], f"ring{ri:04d}", idx)
        elif label == "prompt_sprayer":
            user, ev = A.gen_sprayer(rng, world, cfg, e=sample_e())
        else:
            user, ev = A.gen_thief(rng, world, cfg, e=sample_e())

        user["user_id"] = uid
        user_rows.append(user)
        n = len(ev)
        buf["user_id"].extend([uid] * n)
        for k in A.EVENT_COLS:
            buf[k].extend(ev.cols[k])

        if (i + 1) % FLUSH_EVERY == 0:
            flush()
            print(f"  {i + 1}/{cfg.n_users} users, "
                  f"{n_events_total + len(buf['user_id']):,} events, "
                  f"{time.time() - t_start:.0f}s", flush=True)
    flush()

    users = pl.DataFrame(user_rows, infer_schema_length=None).select(
        "user_id", "label", "subtype", "ring_id", "username", "created_at",
        "signup_country", "payment_hash", "tz_offset", "takeover_ts", "evasion")

    # noisy training labels: missed enforcement + bad abuse reports
    noise = rng.random(len(users))
    is_abuse = users["label"] != "normal"
    noisy = users["label"].to_list()
    fp_classes = ["spam_bot", "prompt_sprayer"]
    for j in range(len(noisy)):
        if is_abuse[j] and noise[j] < cfg.miss_rate:
            noisy[j] = "normal"
        elif not is_abuse[j] and noise[j] < cfg.fp_rate:
            noisy[j] = fp_classes[int(rng.integers(2))]
    users = users.with_columns(pl.Series("label_noisy", noisy))

    events = pl.concat(event_frames).sort(["user_id", "ts"])
    events.write_parquet(os.path.join(cfg.out_dir, "events.parquet"))
    users.write_parquet(os.path.join(cfg.out_dir, "users.parquet"))

    print(f"\nWrote {len(events):,} events for {len(users):,} users "
          f"to {cfg.out_dir} in {time.time() - t_start:.0f}s")
    summary = (events.join(users.select("user_id", "label"), on="user_id")
               .group_by("label").agg(pl.len().alias("events"),
                                      pl.col("user_id").n_unique().alias("active_users"))
               .join(users.group_by("label").len().rename({"len": "users"}), on="label")
               .sort("label"))
    print(summary)


if __name__ == "__main__":
    main()
