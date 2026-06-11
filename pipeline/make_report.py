"""Consolidate run artifacts into a stored report: metrics + figures + summary.

Reads every stage's metrics.json / parquet outputs under a data dir and
writes a self-contained report (markdown + PNG figures + merged metrics)
under reports/<run-name>/.

Usage:
    python -m pipeline.make_report --data data/run2
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STAGES = {"xgb": "baseline", "seq": "seq", "gnn": "gnn",
          "rings": "rings", "spray": "spray", "fusion": "fusion"}
COLORS = {"fused": "#2563eb", "xgb_abuse": "#f59e0b", "seq_abuse": "#10b981",
          "gnn_abuse": "#8b5cf6"}


def load_metrics(data_dir):
    out = {}
    for name, sub in STAGES.items():
        p = os.path.join(data_dir, sub, "metrics.json")
        if os.path.exists(p):
            with open(p) as f:
                out[name] = json.load(f)
    return out


def fig_detector_comparison(m, figs):
    if "fusion" not in m:
        return None
    d = m["fusion"]["abuse_pr_auc"]
    names = list(d.keys())
    vals = [d[k] for k in names]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(names, vals, color=[COLORS.get(n, "#94a3b8") for n in names])
    ax.set_ylim(0.9, 1.001)
    ax.set_ylabel("abuse-vs-normal PR-AUC")
    ax.set_title("Detector comparison (held-out eval)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.4f}",
                ha="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    path = os.path.join(figs, "detector_comparison.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_evasion(m, figs):
    series = []
    if "xgb" in m and m["xgb"].get("abuse_pr_auc_by_evasion"):
        series.append(("XGBoost (tabular)", m["xgb"]["abuse_pr_auc_by_evasion"],
                       COLORS["xgb_abuse"]))
    if "gnn" in m and m["gnn"].get("abuse_pr_auc_by_evasion"):
        series.append(("GraphSAGE", m["gnn"]["abuse_pr_auc_by_evasion"],
                       COLORS["gnn_abuse"]))
    if "fusion" in m and m["fusion"].get("fused_by_evasion"):
        series.append(("Fused system", m["fusion"]["fused_by_evasion"],
                       COLORS["fused"]))
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name, d, c in series:
        ks = sorted(d.keys())
        xs = [f"[{k.split('-')[0]},{k.split('-')[1]})" for k in ks]
        ax.plot(xs, [d[k] for k in ks], marker="o", label=name, color=c,
                linewidth=2)
    ax.set_xlabel("attacker evasion level")
    ax.set_ylabel("abuse PR-AUC")
    ax.set_title("Adversarial robustness: degradation under evasion")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    path = os.path.join(figs, "evasion_robustness.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_per_class(m, figs):
    dets = [(k, lbl) for k, lbl in
            [("xgb", "XGBoost"), ("seq", "GRU"), ("gnn", "GraphSAGE")]
            if k in m and "per_class_pr_auc" in m[k]]
    if not dets:
        return None
    classes = [c for c in m[dets[0][0]]["per_class_pr_auc"] if c != "normal"]
    x = np.arange(len(classes))
    w = 0.8 / len(dets)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    palette = ["#f59e0b", "#10b981", "#8b5cf6"]
    for i, (k, lbl) in enumerate(dets):
        vals = [m[k]["per_class_pr_auc"].get(c, np.nan) for c in classes]
        ax.bar(x + i * w - 0.4 + w / 2, vals, width=w, label=lbl,
               color=palette[i % 3])
    ax.set_xticks(x, classes, fontsize=9)
    ax.set_ylim(0.8, 1.005)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Per-class PR-AUC by detector")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    path = os.path.join(figs, "per_class_pr_auc.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_risk_by_label(data_dir, figs):
    p = os.path.join(data_dir, "fusion", "decisions.parquet")
    if not os.path.exists(p):
        return None
    dec = pl.read_parquet(p).filter(pl.col("is_eval"))
    labels = sorted(dec["label"].unique().to_list())
    data = [dec.filter(pl.col("label") == l)["risk"].to_numpy() for l in labels]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, l in zip(bp["boxes"], labels):
        patch.set_facecolor("#94a3b8" if l == "normal" else "#ef4444")
        patch.set_alpha(0.6)
    ax.set_ylabel("fused risk score")
    ax.set_title("Fused risk by true label (eval half)")
    ax.tick_params(axis="x", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    path = os.path.join(figs, "risk_by_label.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_ato_example(data_dir, figs):
    upath = os.path.join(data_dir, "users.parquet")
    spath = os.path.join(data_dir, "seq", "scores.parquet")
    if not (os.path.exists(upath) and os.path.exists(spath)):
        return None
    users = pl.read_parquet(upath)
    seq = pl.read_parquet(spath)
    cand = (users.filter(pl.col("label") == "token_thief")
            .join(seq.select("user_id", "seq_anomaly", "seq_anomaly_ts"),
                  on="user_id")
            .filter(pl.col("seq_anomaly_ts") > 0)
            .sort("seq_anomaly", descending=True))
    if len(cand) == 0:
        return None
    row = cand.row(0, named=True)
    ev = (pl.scan_parquet(os.path.join(data_dir, "events.parquet"))
          .filter(pl.col("user_id") == row["user_id"])
          .select("ts").collect())
    hourly = (ev.with_columns(hour=pl.col("ts") // 3600)
              .group_by("hour").len().sort("hour"))
    t0 = hourly["hour"].min()
    xs = (hourly["hour"].to_numpy() - t0) / 24.0
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(xs, hourly["len"].to_numpy(), linewidth=1, color="#334155")
    ax.axvline((row["takeover_ts"] / 3600 - t0) / 24, color="#ef4444",
               linewidth=2, label="true takeover")
    ax.axvline((row["seq_anomaly_ts"] / 3600 - t0) / 24, color="#f59e0b",
               linestyle="--", linewidth=2, label="detected NLL spike")
    ax.set_xlabel("days since first event")
    ax.set_ylabel("requests / hour")
    ax.set_title(f"Account takeover localization ({row['user_id']}, "
                 f"evasion={row['evasion']:.2f})")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    path = os.path.join(figs, "ato_timeline.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def write_summary(m, fig_paths, out_dir, data_dir):
    lines = [f"# Run report — `{data_dir}`",
             f"_generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC_", ""]
    if "fusion" in m:
        f = m["fusion"]
        lines += ["## Headline", "",
                  "| detector | abuse PR-AUC |", "|---|---|"]
        lines += [f"| {k} | {v:.4f} |" for k, v in f["abuse_pr_auc"].items()]
        lines += ["", f"Expected cost: **{f['expected_cost']:,}** vs do-nothing "
                  f"{f['cost_do_nothing']:,} "
                  f"(**{100 * (1 - f['expected_cost'] / f['cost_do_nothing']):.1f}%"
                  f" reduction**)", ""]
        lines += ["### Enforcement tiers", "",
                  "| tier | n | abusive | precision |", "|---|---|---|---|"]
        lines += [f"| {t} | {d['n']} | {d['abusive']} | {d['precision']:.3f} |"
                  for t, d in f["tiers"].items()]
        lines.append("")
    for key, title in [("xgb", "XGBoost baseline"), ("seq", "Sequence model"),
                       ("gnn", "GraphSAGE")]:
        if key not in m:
            continue
        lines += [f"## {title}", "", "| class | PR-AUC |", "|---|---|"]
        lines += [f"| {c} | {v:.4f} |"
                  for c, v in m[key]["per_class_pr_auc"].items()]
        if key == "seq":
            lines += ["", f"ATO anomaly AUC: **{m[key]['ato_anomaly_auc']}**, "
                      f"median localization "
                      f"{m[key].get('ato_localization_median_hours', '?')}h"]
        lines.append("")
    if "rings" in m:
        r = m["rings"]
        lines += ["## Ring mining (unsupervised)", "",
                  f"- flagged {r['flagged']:,} accounts, precision "
                  f"{r['precision']:.3f}, recall {r['recall']:.3f} "
                  f"(IP fanout cap {r['ip_fanout_cap']})", ""]
    if "spray" in m:
        s = m["spray"]
        lines += ["## Spray detection (embeddings)", "",
                  f"- cluster ground-truth purity {s['cluster_truth_purity']:.2%},"
                  f" precision {s['precision']:.3f}",
                  "- recall: " + ", ".join(f"{k} {v:.2%}"
                                           for k, v in s["recall"].items()), ""]
    lines += ["## Figures", ""]
    lines += [f"![{os.path.basename(p)}](figs/{os.path.basename(p)})"
              for p in fig_paths if p]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def make_report(data_dir):
    run_name = os.path.basename(os.path.normpath(data_dir))
    out_dir = os.path.join("reports", run_name)
    figs = os.path.join(out_dir, "figs")
    os.makedirs(figs, exist_ok=True)

    m = load_metrics(data_dir)
    if not m:
        print(f"no metrics found under {data_dir}; run the pipeline first")
        return None
    fig_paths = [fig_detector_comparison(m, figs), fig_evasion(m, figs),
                 fig_per_class(m, figs), fig_risk_by_label(data_dir, figs),
                 fig_ato_example(data_dir, figs)]
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    write_summary(m, fig_paths, out_dir, data_dir)
    n_figs = sum(p is not None for p in fig_paths)
    print(f"report written to {out_dir} ({n_figs} figures, summary.md, "
          f"metrics.json)")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/run2")
    make_report(ap.parse_args().data)
