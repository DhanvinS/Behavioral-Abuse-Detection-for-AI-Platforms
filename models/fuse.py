"""Risk fusion + tiered decision layer.

Integrity is a system, not a classifier. This layer:
  1. stacks all detector outputs (XGBoost, sequence, GNN, ring mining,
     spray clusters) with a logistic regression,
  2. maps fused risk to enforcement tiers via explicit cost analysis —
     a false suspension costs real users; a missed thief costs real money.

Stacking protocol: detectors were trained on the 70% train split, so their
train-split scores are overfit. The fusion model is therefore fit on a
random half of the TEST split (noisy labels — what enforcement would have)
and evaluated on the held-out half (clean labels).

Usage:
    python -m models.fuse --data data/run2
"""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# expected cost per account (arbitrary units, documented assumptions):
COST_FP = {"suspend": 100.0,   # lost legit customer + support load
           "challenge": 5.0,   # friction, some churn
           "rate_limit": 1.0,  # mild degradation
           "monitor": 0.0}
COST_FN = {"spam_bot": 10.0, "account_farmer": 20.0,
           "prompt_sprayer": 15.0, "token_thief": 200.0}
# action effectiveness: fraction of abuse cost prevented if applied
EFFECT = {"suspend": 1.0, "challenge": 0.8, "rate_limit": 0.4, "monitor": 0.0}
PRECISION_TARGETS = {"suspend": 0.99, "challenge": 0.90, "rate_limit": 0.60}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/run2")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    d = args.data

    base = pl.read_parquet(os.path.join(d, "features.parquet")).select(
        "user_id", "label", "label_noisy", "evasion").sort("user_id")
    xgb = pl.read_parquet(os.path.join(d, "baseline", "scores.parquet"))
    seq = pl.read_parquet(os.path.join(d, "seq", "scores.parquet")).select(
        "user_id", "seq_abuse", "seq_anomaly")
    gnn = pl.read_parquet(os.path.join(d, "gnn", "scores.parquet")).select(
        "user_id", "gnn_abuse")
    rings = pl.read_parquet(os.path.join(d, "rings", "flags.parquet"))
    spray = pl.read_parquet(os.path.join(d, "spray", "flags.parquet"))

    df = (base.join(xgb.select("user_id", "xgb_abuse", "is_test"), on="user_id")
          .join(seq, on="user_id", how="left")
          .join(gnn, on="user_id")
          .join(rings, on="user_id")
          .join(spray, on="user_id")
          .with_columns(
              seq_missing=pl.col("seq_abuse").is_null().cast(pl.Float32),
              seq_abuse=pl.col("seq_abuse").fill_null(0.5),
              seq_anomaly=pl.col("seq_anomaly").fill_null(0.0),
              ring_flag=pl.col("ring_flag").cast(pl.Float32),
              spray_sig=(pl.col("spray_events") + 1).log(),
              comp_size=(pl.col("component_size")).log1p(),
          ))

    fcols = ["xgb_abuse", "seq_abuse", "seq_anomaly", "gnn_abuse",
             "ring_flag", "spray_sig", "comp_size", "seq_missing"]
    X = df.select(fcols).to_numpy().astype(np.float64)
    y_clean = (df["label"] != "normal").to_numpy().astype(int)
    y_noisy = (df["label_noisy"] != "normal").to_numpy().astype(int)
    labels = df["label"].to_numpy()
    evasion = df["evasion"].to_numpy()
    te = df["is_test"].to_numpy()

    rng = np.random.default_rng(args.seed)
    te_idx = np.where(te)[0]
    half = rng.permutation(len(te_idx))
    fit_idx = te_idx[half[:len(half) // 2]]
    ev_idx = te_idx[half[len(half) // 2:]]

    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=2000, class_weight="balanced"))
    lr.fit(X[fit_idx], y_noisy[fit_idx])
    risk = lr.predict_proba(X)[:, 1]

    print(f"=== fused risk (eval n={len(ev_idx):,}) ===")
    rows = [("fused", risk), ("xgb_abuse", X[:, 0]), ("seq_abuse", X[:, 1]),
            ("gnn_abuse", X[:, 3])]
    metrics = {"abuse_pr_auc": {}}
    for name, s in rows:
        ap_ = average_precision_score(y_clean[ev_idx], s[ev_idx])
        metrics["abuse_pr_auc"][name] = round(float(ap_), 4)
        print(f"  {name:<12} abuse PR-AUC {ap_:.4f}")

    print("\nfused abuse PR-AUC by evasion level:")
    metrics["fused_by_evasion"] = {}
    is_norm_ev = y_clean[ev_idx] == 0
    for lo, hi in ((0.0, 0.33), (0.33, 0.66), (0.66, 1.01)):
        pos = (y_clean[ev_idx] == 1) & (evasion[ev_idx] >= lo) & (evasion[ev_idx] < hi)
        if pos.sum() < 5:
            continue
        m = pos | is_norm_ev
        ap_ = average_precision_score(pos[m].astype(int), risk[ev_idx][m])
        metrics["fused_by_evasion"][f"{lo:.2f}-{hi:.2f}"] = round(float(ap_), 4)
        print(f"  e in [{lo:.2f},{hi:.2f}): {ap_:.4f}  (n={int(pos.sum())})")

    coef = lr.named_steps["logisticregression"].coef_[0]
    print("\nfusion weights (standardized):")
    for c, w in sorted(zip(fcols, coef), key=lambda t: -abs(t[1])):
        print(f"  {c:<12} {w:+.2f}")
    metrics["fusion_weights"] = {c: round(float(w), 3) for c, w in zip(fcols, coef)}

    # ---- tiered actions: thresholds from precision targets on the fit half --
    def threshold_for(target, scores, truth):
        order = np.argsort(-scores)
        tp = np.cumsum(truth[order])
        prec = tp / np.arange(1, len(order) + 1)
        ok = np.where((prec >= target) & (tp > 0))[0]
        if len(ok) == 0:
            return np.inf
        return scores[order][ok[-1]]

    thr = {a: threshold_for(t, risk[fit_idx], y_noisy[fit_idx])
           for a, t in PRECISION_TARGETS.items()}
    # enforce monotone tiers: suspend >= challenge >= rate_limit
    thr["challenge"] = min(thr["challenge"], thr["suspend"])
    thr["rate_limit"] = min(thr["rate_limit"], thr["challenge"])

    def assign(r):
        if r >= thr["suspend"]:
            return "suspend"
        if r >= thr["challenge"]:
            return "challenge"
        if r >= thr["rate_limit"]:
            return "rate_limit"
        return "monitor"

    tiers = np.array([assign(r) for r in risk])
    print(f"\n=== decision tiers (thresholds fit @ precision targets "
          f"{PRECISION_TARGETS}) ===")
    print(f"{'tier':<12}{'n':>7}{'abusive':>9}{'precision':>11}")
    metrics["tiers"] = {}
    cost = 0.0
    for a in ["suspend", "challenge", "rate_limit", "monitor"]:
        m = (tiers == "" + a) & np.isin(np.arange(len(tiers)), ev_idx)
        n = int(m.sum())
        ab = int(y_clean[m].sum())
        p = ab / max(n, 1)
        print(f"{a:<12}{n:>7}{ab:>9}{p:>11.3f}")
        metrics["tiers"][a] = {"n": n, "abusive": ab, "precision": round(p, 3)}
        # expected cost: FPs pay action cost; FNs pay residual abuse cost
        fp_cost = COST_FP[a] * (n - ab)
        fn_cost = sum(COST_FN[l] * (1 - EFFECT[a])
                      for l in labels[m] if l != "normal")
        cost += fp_cost + fn_cost
    do_nothing = sum(COST_FN[l] for l in labels[ev_idx] if l != "normal")
    print(f"\nexpected cost (eval half): {cost:,.0f} vs do-nothing "
          f"{do_nothing:,.0f}  ({100 * (1 - cost / do_nothing):.1f}% reduction)")
    metrics["expected_cost"] = round(cost, 1)
    metrics["cost_do_nothing"] = round(do_nothing, 1)

    out = df.select("user_id", "label", "evasion").with_columns(
        pl.Series("risk", risk.astype(np.float32)),
        pl.Series("tier", tiers),
        pl.Series("is_eval", np.isin(np.arange(len(tiers)), ev_idx)))
    out_dir = os.path.join(d, "fusion")
    os.makedirs(out_dir, exist_ok=True)
    out.write_parquet(os.path.join(out_dir, "decisions.parquet"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved decisions to {out_dir}")


if __name__ == "__main__":
    main()
