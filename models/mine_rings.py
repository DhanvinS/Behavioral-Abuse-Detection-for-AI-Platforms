"""Unsupervised coordination mining: connected components over shared identity.

Union-find over users connected by shared device fingerprints, payment
instruments, and low-fanout IPs. High-fanout IPs (campus/corporate NAT,
CGNAT) are excluded by a degree cap — sharing a university egress IP with
300 people is not evidence of coordination; sharing a device fingerprint is.

Usage:
    python -m models.mine_rings --data data/run2
"""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IP_FANOUT_CAP = 10     # IPs shared by more users than this are treated as NAT
MIN_COMPONENT = 5      # flag components with at least this many users


class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/run2")
    args = ap.parse_args()

    users = pl.read_parquet(os.path.join(args.data, "users.parquet")).sort("user_id")
    ev = pl.scan_parquet(os.path.join(args.data, "events.parquet"))
    uid_map = {u: i for i, u in enumerate(users["user_id"].to_list())}
    n = len(uid_map)
    uf = UnionFind(n)

    def link_on(pairs_df, key):
        """Union users sharing the same key value."""
        linked = 0
        for group in (pairs_df.group_by(key)
                      .agg(pl.col("user_id").unique().alias("us"))
                      .filter(pl.col("us").list.len() > 1)["us"].to_list()):
            base = uid_map[group[0]]
            for u in group[1:]:
                uf.union(base, uid_map[u])
                linked += 1
        return linked

    ud = ev.select("user_id", "device_fp").unique().collect(engine="streaming")
    l_dev = link_on(ud, "device_fp")

    up = users.filter(pl.col("payment_hash").is_not_null()).select(
        "user_id", "payment_hash")
    l_pay = link_on(up, "payment_hash")

    ui = ev.select("user_id", "ip").unique().collect(engine="streaming")
    ip_fanout = ui.group_by("ip").len()
    low_fanout = ip_fanout.filter(
        (pl.col("len") > 1) & (pl.col("len") <= IP_FANOUT_CAP))["ip"]
    l_ip = link_on(ui.filter(pl.col("ip").is_in(low_fanout)), "ip")
    n_nat = len(ip_fanout.filter(pl.col("len") > IP_FANOUT_CAP))
    print(f"links: device {l_dev:,}, payment {l_pay:,}, ip {l_ip:,} "
          f"({n_nat} NAT-like IPs excluded by fanout cap {IP_FANOUT_CAP})")

    roots = np.array([uf.find(i) for i in range(n)])
    comp_ids, comp_sizes = np.unique(roots, return_counts=True)
    flagged_roots = set(comp_ids[comp_sizes >= MIN_COMPONENT])
    flagged = np.array([r in flagged_roots for r in roots])

    labels = users["label"].to_numpy()
    is_coord = np.isin(labels, ["account_farmer", "spam_bot"])
    tp = int((flagged & is_coord).sum())
    prec = tp / max(flagged.sum(), 1)
    rec = tp / max(is_coord.sum(), 1)
    print(f"\n=== ring mining (components >= {MIN_COMPONENT} users) ===")
    print(f"flagged users: {int(flagged.sum()):,}")
    print(f"precision vs coordinated abuse (farmer+bot): {prec:.3f}")
    print(f"recall of coordinated abuse: {rec:.3f}")
    for lab in np.unique(labels):
        m = labels == lab
        print(f"  {lab:<16} flagged {int((flagged & m).sum()):>5} / {int(m.sum()):>5}")

    # per-component purity for the largest components
    comp_of = {r: k for k, r in enumerate(comp_ids)}
    big = np.argsort(-comp_sizes)[:10]
    print("\nlargest components (size, % abusive, dominant label):")
    for k in big:
        m = roots == comp_ids[k]
        if comp_sizes[k] < MIN_COMPONENT:
            continue
        lab, cnt = np.unique(labels[m], return_counts=True)
        dom = lab[cnt.argmax()]
        pct = 100 * np.isin(labels[m], ["account_farmer", "spam_bot"]).mean()
        print(f"  {comp_sizes[k]:>5}  {pct:5.1f}%  {dom}")

    out = pl.DataFrame({
        "user_id": users["user_id"],
        "ring_flag": flagged,
        "component_size": comp_sizes[[comp_of[r] for r in roots]].astype(np.int64),
    })
    out_dir = os.path.join(args.data, "rings")
    os.makedirs(out_dir, exist_ok=True)
    out.write_parquet(os.path.join(out_dir, "flags.parquet"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"precision": round(prec, 4), "recall": round(rec, 4),
                   "flagged": int(flagged.sum()),
                   "ip_fanout_cap": IP_FANOUT_CAP,
                   "min_component": MIN_COMPONENT}, f, indent=2)
    print(f"\nsaved flags to {out_dir}")


if __name__ == "__main__":
    main()
