"""Semantic coordination detection: paraphrase campaigns across accounts.

Embeds prompts with MiniLM, clusters embedding space, then flags clusters
that look like campaigns rather than topics: semantically tight, rare across
the user base, but used intensively by the accounts that touch them.
Keyword filters miss paraphrased jailbreaks; embedding clusters don't.

Also validates the assumption (used by the sequence model) that ground-truth
intent clusters are recoverable from text alone, by reporting cluster purity.

Usage:
    python -m models.spray_detect --data data/run2
"""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl
import torch
from sklearn.cluster import MiniBatchKMeans

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

K = 96                 # embedding clusters (>> true semantic cluster count)
TIGHT_MIN = 0.60       # min mean cosine-to-centroid: "one campaign, many wordings"
COVERAGE_MAX = 0.02    # flagged clusters touch <= 2% of accounts (rare)
INTENSITY_MIN = 8.0    # ...but those accounts hit them hard (events/account)
ACCOUNT_MIN_EVENTS = 5 # events in flagged clusters to flag an account


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/run2")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ev = (pl.scan_parquet(os.path.join(args.data, "events.parquet"))
          .select("user_id", "prompt_text", "prompt_cluster")
          .collect(engine="streaming"))
    users = pl.read_parquet(os.path.join(args.data, "users.parquet")).sort("user_id")
    n_accounts = ev["user_id"].n_unique()

    uniq = ev["prompt_text"].unique().to_list()
    print(f"embedding {len(uniq):,} unique prompts on {dev}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2", device=dev)
    emb = model.encode(uniq, batch_size=512, show_progress_bar=False,
                       normalize_embeddings=True, convert_to_numpy=True)

    km = MiniBatchKMeans(n_clusters=K, random_state=0, n_init=3, batch_size=4096)
    text_cluster = km.fit_predict(emb)
    centroids = km.cluster_centers_ / np.linalg.norm(
        km.cluster_centers_, axis=1, keepdims=True)
    tightness_txt = (emb * centroids[text_cluster]).sum(1)

    cmap = pl.DataFrame({"prompt_text": uniq,
                         "emb_cluster": text_cluster.astype(np.int32),
                         "tight": tightness_txt.astype(np.float32)})
    evc = ev.join(cmap, on="prompt_text", how="left")

    stats = (evc.group_by("emb_cluster").agg(
        n_events=pl.len(),
        n_accs=pl.col("user_id").n_unique(),
        tight=pl.col("tight").mean(),
        # ground-truth namespace purity, for validation only
        top_truth=pl.col("prompt_cluster").mode().first(),
        truth_purity=(pl.col("prompt_cluster") == pl.col("prompt_cluster")
                      .mode().first()).mean(),
    ).with_columns(
        coverage=pl.col("n_accs") / n_accounts,
        intensity=pl.col("n_events") / pl.col("n_accs"),
    ))

    flagged_cl = stats.filter(
        (pl.col("tight") >= TIGHT_MIN)
        & (pl.col("coverage") <= COVERAGE_MAX)
        & (pl.col("intensity") >= INTENSITY_MIN))
    print(f"\nflagged {len(flagged_cl)}/{K} embedding clusters as campaigns:")
    for r in flagged_cl.sort("n_events", descending=True).head(12).iter_rows(named=True):
        print(f"  cluster {r['emb_cluster']:>3}: {r['n_events']:>7,} events, "
              f"{r['n_accs']:>4} accts, tight {r['tight']:.2f}, "
              f"truth={r['top_truth']} ({r['truth_purity']:.0%} pure)")
    mean_purity = flagged_cl["truth_purity"].mean()

    acc = (evc.filter(pl.col("emb_cluster").is_in(flagged_cl["emb_cluster"].implode()))
           .group_by("user_id").agg(spray_events=pl.len()))
    flagged_users = set(acc.filter(
        pl.col("spray_events") >= ACCOUNT_MIN_EVENTS)["user_id"].to_list())

    labels = users["label"].to_numpy()
    evasion = users["evasion"].to_numpy()
    flag = np.array([u in flagged_users for u in users["user_id"].to_list()])
    is_target = np.isin(labels, ["prompt_sprayer", "spam_bot"])
    prec = (flag & is_target).sum() / max(flag.sum(), 1)
    print(f"\n=== spray/campaign detection ===")
    print(f"flagged accounts: {int(flag.sum()):,} "
          f"(precision vs sprayer+spam: {prec:.3f})")
    res = {"cluster_truth_purity": round(float(mean_purity), 4),
           "precision": round(float(prec), 4), "recall": {}}
    for lab in ["prompt_sprayer", "spam_bot", "normal"]:
        m = labels == lab
        r = (flag & m).sum() / max(m.sum(), 1)
        res["recall"][lab] = round(float(r), 4)
        print(f"  {lab:<16} flagged {int((flag & m).sum()):>5} / {int(m.sum()):>5}")
    slow = (labels == "prompt_sprayer") & (evasion > 0.6)
    if slow.sum():
        r = (flag & slow).sum() / slow.sum()
        res["recall"]["prompt_sprayer_slow(e>0.6)"] = round(float(r), 4)
        print(f"  slow sprayers (e>0.6): flagged {int((flag & slow).sum())} / "
              f"{int(slow.sum())}  <- invisible to tabular baseline")

    out_dir = os.path.join(args.data, "spray")
    os.makedirs(out_dir, exist_ok=True)
    spray_events = dict(zip(acc["user_id"].to_list(), acc["spray_events"].to_list()))
    pl.DataFrame({
        "user_id": users["user_id"],
        "spray_flag": flag,
        "spray_events": [spray_events.get(u, 0) for u in users["user_id"].to_list()],
    }).write_parquet(os.path.join(out_dir, "flags.parquet"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved flags to {out_dir}")


if __name__ == "__main__":
    main()
