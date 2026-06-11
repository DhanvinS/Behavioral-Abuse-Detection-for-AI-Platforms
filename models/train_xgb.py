"""XGBoost baseline on per-account tabular features.

Trains on noisy labels (label_noisy: simulated enforcement gaps / bad reports),
evaluates against clean ground truth. Metrics: per-class PR-AUC and
precision@k on the fused abuse score — never accuracy (heavy class imbalance).

Usage:
    python -m models.train_xgb --data data/run1
"""
import argparse
import json
import os

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

NON_FEATURES = {"user_id", "label", "label_noisy", "subtype", "ring_id",
                "username", "created_at", "signup_country", "payment_hash",
                "takeover_ts", "first_ts", "last_ts", "tz_offset"}


def precision_at_k(y_true_abuse: np.ndarray, score: np.ndarray, k: int) -> float:
    idx = np.argsort(-score)[:k]
    return float(y_true_abuse[idx].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pl.read_parquet(os.path.join(args.data, "features.parquet"))
    feat_cols = [c for c in df.columns if c not in NON_FEATURES
                 and df[c].dtype.is_numeric()]
    X = df.select(feat_cols).fill_null(-1.0).to_numpy().astype(np.float32)

    classes = sorted(df["label"].unique().to_list())
    cls_idx = {c: i for i, c in enumerate(classes)}
    y_clean = np.array([cls_idx[v] for v in df["label"].to_list()])
    y_noisy = np.array([cls_idx[v] for v in df["label_noisy"].to_list()])

    idx_tr, idx_te = train_test_split(
        np.arange(len(y_clean)), test_size=0.3, random_state=args.seed,
        stratify=y_clean)

    # class-balanced sample weights against heavy imbalance
    freq = np.bincount(y_noisy[idx_tr], minlength=len(classes)).astype(float)
    w = (len(idx_tr) / (len(classes) * np.maximum(freq, 1)))[y_noisy[idx_tr]]

    clf = XGBClassifier(
        n_estimators=400, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        device="cuda", eval_metric="mlogloss", random_state=args.seed)
    try:
        clf.fit(X[idx_tr], y_noisy[idx_tr], sample_weight=w)
    except Exception as e:  # no usable GPU -> CPU fallback
        print(f"CUDA training failed ({e}); falling back to CPU")
        clf.set_params(device="cpu")
        clf.fit(X[idx_tr], y_noisy[idx_tr], sample_weight=w)

    proba = clf.predict_proba(X[idx_te])
    y_te = y_clean[idx_te]

    metrics = {"classes": classes, "n_test": len(idx_te), "per_class_pr_auc": {}}
    print(f"\n=== XGBoost baseline (test n={len(idx_te):,}) ===")
    print(f"{'class':<16}{'support':>8}{'PR-AUC':>9}")
    for c, i in cls_idx.items():
        support = int((y_te == i).sum())
        ap_ = average_precision_score((y_te == i).astype(int), proba[:, i])
        metrics["per_class_pr_auc"][c] = round(float(ap_), 4)
        print(f"{c:<16}{support:>8}{ap_:>9.3f}")

    abuse_score = 1.0 - proba[:, cls_idx["normal"]]
    y_abuse = (y_te != cls_idx["normal"]).astype(int)
    metrics["abuse_pr_auc"] = round(float(
        average_precision_score(y_abuse, abuse_score)), 4)
    print(f"\nabuse-vs-normal PR-AUC: {metrics['abuse_pr_auc']:.3f} "
          f"(base rate {y_abuse.mean():.3f})")

    n_abuse = int(y_abuse.sum())
    metrics["precision_at_k"] = {}
    for k in (100, 500, 1000):
        if k <= len(y_abuse):
            p = precision_at_k(y_abuse, abuse_score, k)
            metrics["precision_at_k"][k] = round(p, 4)
            cap = min(1.0, n_abuse / k)
            print(f"precision@{k}: {p:.3f} (max achievable {cap:.3f})")

    order = np.argsort(-clf.feature_importances_)[:15]
    print("\ntop features (gain):")
    metrics["top_features"] = []
    for i in order:
        print(f"  {feat_cols[i]:<24}{clf.feature_importances_[i]:.3f}")
        metrics["top_features"].append(feat_cols[i])

    out_dir = os.path.join(args.data, "baseline")
    os.makedirs(out_dir, exist_ok=True)
    clf.save_model(os.path.join(out_dir, "xgb.ubj"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model + metrics to {out_dir}")


if __name__ == "__main__":
    main()
