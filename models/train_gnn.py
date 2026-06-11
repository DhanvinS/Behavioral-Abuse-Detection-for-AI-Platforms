"""Heterogeneous GNN over the user-IP-device-payment graph.

Coordinated abuse (farming rings, bot farms) is invisible per-account but
obvious structurally: hundreds of accounts sharing a handful of device
fingerprints, payment instruments, or egress IPs. User nodes carry the same
tabular features as the XGBoost baseline, so any gain over the baseline is
attributable to message passing over shared infrastructure.

Usage:
    python -m models.train_gnn --data data/run2
"""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from models.train_xgb import NON_FEATURES, evasion_slices  # noqa: E402

ASN_TYPES = ["residential", "mobile", "datacenter", "university", "corporate"]


def build_graph(data_dir):
    feats = pl.read_parquet(os.path.join(data_dir, "features.parquet")).sort("user_id")
    ev = pl.scan_parquet(os.path.join(data_dir, "events.parquet"))

    user_ids = feats["user_id"].to_list()
    user_map = {u: i for i, u in enumerate(user_ids)}

    # entity edge lists (unique pairs)
    ui = (ev.group_by(["user_id", "ip"])
          .agg(pl.len(), pl.col("asn_type").first())
          .collect(engine="streaming"))
    ud = ev.select("user_id", "device_fp").unique().collect(engine="streaming")
    up = (feats.filter(pl.col("payment_hash").is_not_null())
          .select("user_id", "payment_hash"))

    def index_entities(values):
        uniq = sorted(set(values))
        return {v: i for i, v in enumerate(uniq)}, len(uniq)

    ip_map, n_ip = index_entities(ui["ip"].to_list())
    dev_map, n_dev = index_entities(ud["device_fp"].to_list())
    pay_map, n_pay = index_entities(up["payment_hash"].to_list())

    data = HeteroData()

    # user features: identical columns to the XGBoost baseline, standardized
    feat_cols = [c for c in feats.columns
                 if c not in NON_FEATURES and feats[c].dtype.is_numeric()]
    X = feats.select(feat_cols).fill_null(-1.0).to_numpy().astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    data["user"].x = torch.from_numpy(X)

    # ip features: log degree + asn_type one-hot
    ip_x = np.zeros((n_ip, 1 + len(ASN_TYPES)), dtype=np.float32)
    ip_deg = np.zeros(n_ip)
    for ip, at in zip(ui["ip"].to_list(), ui["asn_type"].to_list()):
        i = ip_map[ip]
        ip_deg[i] += 1
        ip_x[i, 1 + ASN_TYPES.index(at)] = 1.0
    ip_x[:, 0] = np.log1p(ip_deg)
    data["ip"].x = torch.from_numpy(ip_x)

    def degree_x(pairs, mapping, n):
        deg = np.zeros(n, dtype=np.float32)
        for v in pairs:
            deg[mapping[v]] += 1
        return torch.from_numpy(np.log1p(deg))[:, None]

    data["device"].x = degree_x(ud["device_fp"].to_list(), dev_map, n_dev)
    data["payment"].x = degree_x(up["payment_hash"].to_list(), pay_map, n_pay)

    def edges(src_ids, dst_ids, dst_map):
        src = torch.tensor([user_map[u] for u in src_ids], dtype=torch.long)
        dst = torch.tensor([dst_map[v] for v in dst_ids], dtype=torch.long)
        return torch.stack([src, dst])

    e_ui = edges(ui["user_id"].to_list(), ui["ip"].to_list(), ip_map)
    e_ud = edges(ud["user_id"].to_list(), ud["device_fp"].to_list(), dev_map)
    e_up = edges(up["user_id"].to_list(), up["payment_hash"].to_list(), pay_map)
    data["user", "uses", "ip"].edge_index = e_ui
    data["ip", "used_by", "user"].edge_index = e_ui.flip(0)
    data["user", "has", "device"].edge_index = e_ud
    data["device", "of", "user"].edge_index = e_ud.flip(0)
    data["user", "pays", "payment"].edge_index = e_up
    data["payment", "of", "user"].edge_index = e_up.flip(0)

    print(f"graph: {len(user_ids):,} users, {n_ip:,} ips, {n_dev:,} devices, "
          f"{n_pay:,} payments, {e_ui.size(1) + e_ud.size(1) + e_up.size(1):,} edges")
    return data, feats


class HeteroSAGE(torch.nn.Module):
    def __init__(self, edge_types, hidden=64, n_classes=5):
        super().__init__()
        self.conv1 = HeteroConv({et: SAGEConv((-1, -1), hidden)
                                 for et in edge_types}, aggr="sum")
        self.conv2 = HeteroConv({et: SAGEConv((hidden, hidden), hidden)
                                 for et in edge_types}, aggr="sum")
        self.head = torch.nn.Linear(hidden, n_classes)

    def forward(self, x_dict, ei_dict):
        h = {k: F.relu(v) for k, v in self.conv1(x_dict, ei_dict).items()}
        h = {k: F.relu(v) for k, v in self.conv2(h, ei_dict).items()}
        return self.head(h["user"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/run2")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data, feats = build_graph(args.data)
    classes = sorted(feats["label"].unique().to_list())
    cls_idx = {c: i for i, c in enumerate(classes)}
    y_clean = np.array([cls_idx[v] for v in feats["label"].to_list()])
    y_noisy = np.array([cls_idx[v] for v in feats["label_noisy"].to_list()])

    idx_tr, idx_te = train_test_split(
        np.arange(len(y_clean)), test_size=0.3, random_state=args.seed,
        stratify=y_clean)

    freq = np.bincount(y_noisy[idx_tr], minlength=len(classes)).astype(float)
    cls_w = torch.tensor(len(idx_tr) / (len(classes) * np.maximum(freq, 1)),
                         dtype=torch.float32, device=dev)
    y_t = torch.from_numpy(y_noisy).to(dev)
    tr_t = torch.from_numpy(idx_tr).to(dev)

    data = data.to(dev)
    model = HeteroSAGE(data.edge_types, n_classes=len(classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)

    print(f"training {args.epochs} epochs full-batch ({dev})...")
    for epoch in range(args.epochs):
        model.train()
        logits = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(logits[tr_t], y_t[tr_t], weight=cls_w)
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch + 1}: loss {float(loss):.4f}")

    model.eval()
    with torch.no_grad():
        proba = F.softmax(model(data.x_dict, data.edge_index_dict), -1).cpu().numpy()

    y_te = y_clean[idx_te]
    metrics = {"classes": classes, "per_class_pr_auc": {}}
    print(f"\n=== hetero GraphSAGE (test n={len(idx_te):,}) ===")

    xgb_path = os.path.join(args.data, "baseline", "metrics.json")
    xgb_m = json.load(open(xgb_path)) if os.path.exists(xgb_path) else None
    hdr = f"{'class':<16}{'support':>8}{'PR-AUC':>9}"
    print(hdr + ("{:>12}".format("XGB PR-AUC") if xgb_m else ""))
    for c, i in cls_idx.items():
        ap_ = average_precision_score((y_te == i).astype(int), proba[idx_te, i])
        metrics["per_class_pr_auc"][c] = round(float(ap_), 4)
        row = f"{c:<16}{int((y_te == i).sum()):>8}{ap_:>9.3f}"
        if xgb_m:
            row += f"{xgb_m['per_class_pr_auc'].get(c, float('nan')):>12.3f}"
        print(row)

    abuse_score = 1.0 - proba[:, cls_idx["normal"]]
    y_abuse = (y_te != cls_idx["normal"]).astype(int)
    metrics["abuse_pr_auc"] = round(float(
        average_precision_score(y_abuse, abuse_score[idx_te])), 4)
    print(f"abuse-vs-normal PR-AUC: {metrics['abuse_pr_auc']:.3f}")

    metrics["abuse_pr_auc_by_evasion"] = evasion_slices(
        feats, idx_te, y_te, cls_idx["normal"], abuse_score[idx_te], "abuse PR-AUC")

    is_test = np.zeros(len(y_clean), dtype=bool)
    is_test[idx_te] = True
    out_dir = os.path.join(args.data, "gnn")
    os.makedirs(out_dir, exist_ok=True)
    pl.DataFrame({
        "user_id": feats["user_id"],
        "gnn_abuse": abuse_score.astype(np.float32),
        **{f"gnn_{c}": proba[:, i].astype(np.float32)
           for c, i in cls_idx.items() if c != "normal"},
        "is_test": is_test,
    }).write_parquet(os.path.join(out_dir, "scores.parquet"))
    torch.save(model.state_dict(), os.path.join(out_dir, "sage.pt"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved scores + model to {out_dir}")


if __name__ == "__main__":
    main()
