"""Sequence model over per-account event streams (GRU, multi-task).

Two heads:
  1. supervised 5-class account classification (trained on noisy labels)
  2. self-supervised next-event prediction (delta bucket + prompt-cluster
     bucket), trained ONLY on normal-labeled accounts -> a behavioral
     language model of "normal". Per-event NLL spikes localize account
     takeover: the attacker can fake marginals, but not the victim's
     conditional behavior.

Usage:
    python -m models.train_seq --data data/run2
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MAX_LEN = 384
N_DELTA_BUCKETS = 8
N_CLUSTER_BUCKETS = 32
ASN_TYPES = ["residential", "mobile", "datacenter", "university", "corporate"]
ENDPOINTS = ["/v1/chat", "/v1/completions", "/v1/embeddings"]
DAY = 86400


# ---------------------------------------------------------------------------
# data prep
# ---------------------------------------------------------------------------
def build_tensors(data_dir):
    ev = (pl.scan_parquet(os.path.join(data_dir, "events.parquet"))
          .sort(["user_id", "ts"])
          .with_columns(
              delta=pl.col("ts").diff().over("user_id").cast(pl.Float64).fill_null(0.0),
              ip_chg=(pl.col("ip") != pl.col("ip").shift(1).over("user_id"))
              .cast(pl.Float32).fill_null(0.0),
              dev_chg=(pl.col("device_fp") != pl.col("device_fp").shift(1).over("user_id"))
              .cast(pl.Float32).fill_null(0.0),
              cc_chg=(pl.col("country") != pl.col("country").shift(1).over("user_id"))
              .cast(pl.Float32).fill_null(0.0),
              hour=((pl.col("ts") % DAY) / 3600.0).cast(pl.Float32),
          )
          .select("user_id", "ts", "delta", "hour", "tokens_used", "success",
                  "ip_chg", "dev_chg", "cc_chg", "asn_type", "endpoint",
                  "prompt_cluster")
          .collect(engine="streaming"))

    # keep only each user's most recent MAX_LEN events
    ev = (ev.with_columns(
        rev_rank=pl.col("ts").cum_count(reverse=True).over("user_id"))
        .filter(pl.col("rev_rank") <= MAX_LEN)
        .drop("rev_rank")
        .sort(["user_id", "ts"]))

    uid_arr = ev["user_id"].to_numpy()
    uids, starts = np.unique(uid_arr, return_index=True)
    order = np.argsort(starts)
    uids, starts = uids[order], starts[order]
    ends = np.append(starts[1:], len(uid_arr))

    asn_map = {a: i for i, a in enumerate(ASN_TYPES)}
    ep_map = {e: i for i, e in enumerate(ENDPOINTS)}
    asn_idx = np.array([asn_map[a] for a in ev["asn_type"].to_list()], dtype=np.int64)
    ep_idx = np.array([ep_map[e] for e in ev["endpoint"].to_list()], dtype=np.int64)
    cl_idx = np.array([zlib.crc32(c.encode()) % N_CLUSTER_BUCKETS
                       for c in ev["prompt_cluster"].to_list()], dtype=np.int64)
    delta = ev["delta"].to_numpy()
    delta_bucket = np.clip(np.floor(np.log10(np.maximum(delta, 1.0))), 0,
                           N_DELTA_BUCKETS - 1).astype(np.int64)
    hour = ev["hour"].to_numpy().astype(np.float32)
    num = np.stack([
        np.log1p(delta).astype(np.float32) / 14.0,
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.log1p(ev["tokens_used"].to_numpy()).astype(np.float32) / 10.0,
        ev["success"].to_numpy().astype(np.float32),
        ev["ip_chg"].to_numpy(), ev["dev_chg"].to_numpy(), ev["cc_chg"].to_numpy(),
    ], axis=1)
    ts_all = ev["ts"].to_numpy()

    n = len(uids)
    X_num = np.zeros((n, MAX_LEN, num.shape[1]), dtype=np.float32)
    X_cat = np.zeros((n, MAX_LEN, 3), dtype=np.int64)   # asn, ep, cluster
    Y_next = np.zeros((n, MAX_LEN, 2), dtype=np.int64)  # delta bucket, cluster
    TS = np.zeros((n, MAX_LEN), dtype=np.int64)
    lengths = (ends - starts).astype(np.int64)
    for i, (s, t) in enumerate(zip(starts, ends)):
        L = t - s
        X_num[i, :L] = num[s:t]
        X_cat[i, :L, 0] = asn_idx[s:t]
        X_cat[i, :L, 1] = ep_idx[s:t]
        X_cat[i, :L, 2] = cl_idx[s:t]
        Y_next[i, :L, 0] = delta_bucket[s:t]
        Y_next[i, :L, 1] = cl_idx[s:t]
        TS[i, :L] = ts_all[s:t]
    return uids, lengths, X_num, X_cat, Y_next, TS


class SeqModel(nn.Module):
    def __init__(self, n_num=8, hidden=128, n_classes=5):
        super().__init__()
        self.asn_emb = nn.Embedding(len(ASN_TYPES), 4)
        self.ep_emb = nn.Embedding(len(ENDPOINTS), 3)
        self.cl_emb = nn.Embedding(N_CLUSTER_BUCKETS, 12)
        self.proj = nn.Linear(n_num + 4 + 3 + 12, 64)
        self.gru = nn.GRU(64, hidden, batch_first=True)
        self.cls = nn.Linear(hidden * 2, n_classes)
        self.next_delta = nn.Linear(hidden, N_DELTA_BUCKETS)
        self.next_cl = nn.Linear(hidden, N_CLUSTER_BUCKETS)

    def forward(self, xn, xc, lengths):
        x = torch.cat([xn, self.asn_emb(xc[..., 0]), self.ep_emb(xc[..., 1]),
                       self.cl_emb(xc[..., 2])], dim=-1)
        h, _ = self.gru(F.relu(self.proj(x)))           # [B, T, H]
        mask = (torch.arange(h.size(1), device=h.device)[None, :]
                < lengths[:, None]).float()
        mean = (h * mask[..., None]).sum(1) / lengths[:, None].clamp(min=1)
        last = h[torch.arange(h.size(0)), (lengths - 1).clamp(min=0)]
        logits = self.cls(torch.cat([mean, last], dim=-1))
        return logits, self.next_delta(h), self.next_cl(h), mask


def lm_nll(pred_d, pred_c, y_next, mask):
    """Per-position next-event NLL; position t predicts event t+1."""
    nll_d = F.cross_entropy(pred_d[:, :-1].transpose(1, 2), y_next[:, 1:, 0],
                            reduction="none")
    nll_c = F.cross_entropy(pred_c[:, :-1].transpose(1, 2), y_next[:, 1:, 1],
                            reduction="none")
    return (nll_d + nll_c) * mask[:, 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/run2")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    users = pl.read_parquet(os.path.join(args.data, "users.parquet"))
    classes = sorted(users["label"].unique().to_list())
    cls_idx = {c: i for i, c in enumerate(classes)}
    normal_i = cls_idx["normal"]

    print("building sequence tensors...")
    uids, lengths, X_num, X_cat, Y_next, TS = build_tensors(args.data)
    umeta = users.filter(pl.col("user_id").is_in(uids.tolist()))
    umeta = umeta.sort("user_id")  # uids from np.unique are sorted too
    assert umeta["user_id"].to_list() == sorted(uids.tolist())
    su = np.argsort(uids)
    uids, lengths = uids[su], lengths[su]
    X_num, X_cat, Y_next, TS = X_num[su], X_cat[su], Y_next[su], TS[su]
    y_clean = np.array([cls_idx[v] for v in umeta["label"].to_list()])
    y_noisy = np.array([cls_idx[v] for v in umeta["label_noisy"].to_list()])
    evasion = umeta["evasion"].to_numpy()
    takeover = umeta["takeover_ts"].to_numpy()

    # same split protocol as the baseline: stratified 70/30 on clean labels
    idx_tr, idx_te = train_test_split(
        np.arange(len(uids)), test_size=0.3, random_state=args.seed,
        stratify=y_clean)

    freq = np.bincount(y_noisy[idx_tr], minlength=len(classes)).astype(float)
    cls_w = torch.tensor(len(idx_tr) / (len(classes) * np.maximum(freq, 1)),
                         dtype=torch.float32, device=dev)

    model = SeqModel(n_classes=len(classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    B = 128

    print(f"training on {len(idx_tr):,} accounts ({dev})...")
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.default_rng(epoch).permutation(idx_tr)
        tot_cls = tot_lm = nb = 0.0
        for b in range(0, len(perm), B):
            sl = perm[b:b + B]
            xn = torch.from_numpy(X_num[sl]).to(dev)
            xc = torch.from_numpy(X_cat[sl]).to(dev)
            yn = torch.from_numpy(Y_next[sl]).to(dev)
            ln = torch.from_numpy(lengths[sl]).to(dev)
            yc = torch.from_numpy(y_noisy[sl]).to(dev)
            logits, pd, pc, mask = model(xn, xc, ln)
            loss_cls = F.cross_entropy(logits, yc, weight=cls_w)
            # LM trained only on (noisily) normal accounts
            is_norm = (yc == normal_i).float()[:, None]
            nll = lm_nll(pd, pc, yn, mask) * is_norm
            loss_lm = nll.sum() / (mask[:, 1:] * is_norm).sum().clamp(min=1)
            loss = loss_cls + 0.5 * loss_lm
            opt.zero_grad(); loss.backward(); opt.step()
            tot_cls += float(loss_cls); tot_lm += float(loss_lm); nb += 1
        print(f"  epoch {epoch + 1}: cls {tot_cls / nb:.4f}  lm {tot_lm / nb:.4f}")

    # ---- inference over all accounts -------------------------------------
    model.eval()
    probs = np.zeros((len(uids), len(classes)), dtype=np.float32)
    anom = np.zeros(len(uids), dtype=np.float32)
    anom_ts = np.zeros(len(uids), dtype=np.int64)
    W = 16  # smoothing window for NLL spike detection
    with torch.no_grad():
        for b in range(0, len(uids), B):
            sl = np.arange(b, min(b + B, len(uids)))
            xn = torch.from_numpy(X_num[sl]).to(dev)
            xc = torch.from_numpy(X_cat[sl]).to(dev)
            yn = torch.from_numpy(Y_next[sl]).to(dev)
            ln = torch.from_numpy(lengths[sl]).to(dev)
            logits, pd, pc, mask = model(xn, xc, ln)
            probs[sl] = F.softmax(logits, dim=-1).cpu().numpy()
            nll = (lm_nll(pd, pc, yn, mask)).cpu().numpy()  # [B, T-1]
            m = mask[:, 1:].cpu().numpy()
            for j, i in enumerate(sl):
                L = int(lengths[i]) - 1
                if L < W * 2:
                    continue
                s = np.convolve(nll[j, :L], np.ones(W) / W, mode="valid")
                anom[i] = float(s.max() - np.median(s))
                anom_ts[i] = TS[i, int(s.argmax()) + W // 2 + 1]

    y_te = y_clean[idx_te]
    metrics = {"classes": classes, "per_class_pr_auc": {}}
    print(f"\n=== sequence model (test n={len(idx_te):,}) ===")
    print(f"{'class':<16}{'support':>8}{'PR-AUC':>9}")
    for c, i in cls_idx.items():
        ap_ = average_precision_score((y_te == i).astype(int), probs[idx_te, i])
        metrics["per_class_pr_auc"][c] = round(float(ap_), 4)
        print(f"{c:<16}{int((y_te == i).sum()):>8}{ap_:>9.3f}")

    abuse_score = 1.0 - probs[:, normal_i]
    y_abuse = (y_te != normal_i).astype(int)
    metrics["abuse_pr_auc"] = round(float(
        average_precision_score(y_abuse, abuse_score[idx_te])), 4)
    print(f"abuse-vs-normal PR-AUC: {metrics['abuse_pr_auc']:.3f}")

    # ---- takeover detection via LM perplexity spike ----------------------
    te_mask_thief = y_te == cls_idx["token_thief"]
    te_mask_norm = y_te == normal_i
    scored = lengths[idx_te] >= W * 2 + 1
    sel = (te_mask_thief | te_mask_norm) & scored
    auc = roc_auc_score(te_mask_thief[sel].astype(int), anom[idx_te][sel])
    metrics["ato_anomaly_auc"] = round(float(auc), 4)
    print(f"\nATO detection (LM NLL spike), thief-vs-normal AUC: {auc:.3f}")

    thief_sel = idx_te[te_mask_thief & scored]
    if len(thief_sel):
        err_h = np.abs(anom_ts[thief_sel] - takeover[thief_sel]) / 3600.0
        metrics["ato_localization_median_hours"] = round(float(np.median(err_h)), 2)
        print(f"takeover localization: median error "
              f"{np.median(err_h):.1f}h (n={len(thief_sel)})")

    is_test = np.zeros(len(uids), dtype=bool)
    is_test[idx_te] = True
    out_dir = os.path.join(args.data, "seq")
    os.makedirs(out_dir, exist_ok=True)
    pl.DataFrame({
        "user_id": uids.tolist(),
        "seq_abuse": abuse_score,
        "seq_anomaly": anom,
        "seq_anomaly_ts": anom_ts,
        **{f"seq_{c}": probs[:, i] for c, i in cls_idx.items() if c != "normal"},
        "is_test": is_test,
    }).write_parquet(os.path.join(out_dir, "scores.parquet"))
    torch.save(model.state_dict(), os.path.join(out_dir, "gru.pt"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved scores + model to {out_dir}")


if __name__ == "__main__":
    main()
