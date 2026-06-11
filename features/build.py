"""Per-account behavioral features from the raw event log.

Deliberately account-local: no cross-account (graph) signals here. The gap
between this feature set and the GNN on coordinated abuse is the headline
comparison of the project.

Usage:
    python -m features.build --data data/run1
"""
import argparse
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # box-drawing on cp1252 consoles

DAY = 86400


def _entropy_of(ev: pl.LazyFrame, cat: str, out_name: str) -> pl.LazyFrame:
    """Shannon entropy of the per-user distribution of a categorical column."""
    counts = ev.group_by(["user_id", cat]).len()
    totals = counts.group_by("user_id").agg(pl.col("len").sum().alias("tot"))
    return (counts.join(totals, on="user_id")
            .with_columns((pl.col("len") / pl.col("tot")).alias("p"))
            .group_by("user_id")
            .agg((-(pl.col("p") * pl.col("p").log())).sum().alias(out_name)))


def build_features(data_dir: str) -> pl.DataFrame:
    ev = (pl.scan_parquet(os.path.join(data_dir, "events.parquet"))
          .sort(["user_id", "ts"])
          .with_columns(
              delta=pl.col("ts").diff().over("user_id").cast(pl.Float64),
              hour=((pl.col("ts") % DAY) // 3600).cast(pl.Int32),
              day=(pl.col("ts") // DAY).cast(pl.Int64),
              hour_bucket=(pl.col("ts") // 3600).cast(pl.Int64),
              prompt_len=pl.col("prompt_text").str.len_chars(),
          )
          .with_columns(
              # log-spaced inter-event-time buckets for timing-entropy
              delta_bucket=(pl.col("delta").clip(1, None).log(base=10)
                            .floor().clip(0, 7)),
          ))

    base = ev.group_by("user_id").agg(
        n_events=pl.len(),
        first_ts=pl.col("ts").min(),
        last_ts=pl.col("ts").max(),
        active_days=pl.col("day").n_unique(),
        n_ips=pl.col("ip").n_unique(),
        n_asns=pl.col("asn").n_unique(),
        n_countries=pl.col("country").n_unique(),
        n_devices=pl.col("device_fp").n_unique(),
        n_uas=pl.col("user_agent").n_unique(),
        dc_frac=(pl.col("asn_type") == "datacenter").mean(),
        night_frac=(pl.col("hour") < 6).mean(),
        # NOTE: no features over prompt_cluster — those are ground-truth
        # intent labels for evaluation only; using them would leak the label.
        # Content signals enter via the embedding pipeline (models/spray_detect).
        dup_prompt_frac=1 - pl.col("prompt_text").n_unique() / pl.len(),
        mean_prompt_len=pl.col("prompt_len").mean(),
        tokens_mean=pl.col("tokens_used").mean(),
        tokens_p95=pl.col("tokens_used").quantile(0.95),
        tokens_total=pl.col("tokens_used").sum(),
        success_rate=pl.col("success").mean(),
        delta_mean=pl.col("delta").mean(),
        delta_median=pl.col("delta").median(),
        delta_std=pl.col("delta").std(),
        delta_p10=pl.col("delta").quantile(0.10),
        delta_p90=pl.col("delta").quantile(0.90),
        frac_fast=(pl.col("delta") < 10).mean(),
        chat_frac=(pl.col("endpoint") == "/v1/chat").mean(),
    )

    hourly_peak = (ev.group_by(["user_id", "hour_bucket"]).len()
                   .group_by("user_id")
                   .agg(pl.col("len").max().alias("max_events_1h")))

    feats = (base
             .join(_entropy_of(ev, "hour", "hour_entropy"), on="user_id", how="left")
             .join(_entropy_of(ev, "delta_bucket", "timing_entropy"), on="user_id", how="left")
             .join(hourly_peak, on="user_id", how="left")
             .with_columns(
                 events_per_active_day=pl.col("n_events") / pl.col("active_days"),
                 delta_cv=pl.col("delta_std") / pl.col("delta_mean"),
                 burstiness=((pl.col("delta_std") - pl.col("delta_mean"))
                             / (pl.col("delta_std") + pl.col("delta_mean"))),
             ))

    users = pl.scan_parquet(os.path.join(data_dir, "users.parquet"))
    out = (users.join(feats, on="user_id", how="left")
           .with_columns(
               account_age_days=(pl.lit(0.0) + (pl.col("last_ts").max()
                                 - pl.col("created_at")) / DAY),
               dormancy_days=((pl.col("first_ts") - pl.col("created_at")) / DAY),
           )
           .collect(engine="streaming"))

    # zero-event accounts (dormant farm accounts): real and informative
    fill_zero = ["n_events", "active_days", "n_ips", "n_asns", "n_countries",
                 "n_devices", "n_uas", "tokens_total", "max_events_1h"]
    out = out.with_columns([pl.col(c).fill_null(0) for c in fill_zero])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/run")
    args = ap.parse_args()

    out = build_features(args.data)
    path = os.path.join(args.data, "features.parquet")
    out.write_parquet(path)
    print(f"Wrote {out.shape[0]:,} rows x {out.shape[1]} cols to {path}")
    print(out.group_by("label").agg(
        pl.len(), pl.col("n_events").mean().round(1),
        pl.col("delta_cv").mean().round(2),
        pl.col("dup_prompt_frac").mean().round(3),
        pl.col("n_devices").mean().round(2)).sort("label"))


if __name__ == "__main__":
    main()
