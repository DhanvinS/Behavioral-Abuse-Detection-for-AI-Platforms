"""Run the full pipeline with persistent logs, then build the report.

Each stage's console output is streamed live AND saved to
reports/<run-name>/logs/<NN>_<stage>.log. On completion (or failure) a
manifest with timings/exit codes is written; on success the consolidated
report (summary.md + figures) is generated.

Usage:
    python -m pipeline.run_all --users 20000 --days 28 --evasion 1.0 --data data/run2
    python -m pipeline.run_all --data data/run2 --skip-sim   # re-run models only
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from pipeline.make_report import make_report

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_stage(name, args_list, log_dir, idx):
    log_path = os.path.join(log_dir, f"{idx:02d}_{name}.log")
    cmd = [sys.executable, "-u", "-m"] + args_list
    print(f"\n=== [{idx}] {name}: {' '.join(args_list)} ===", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\nstarted {datetime.now(timezone.utc)}\n\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        proc.wait()
        dt = time.time() - t0
        log.write(f"\nexit code {proc.returncode}, {dt:.1f}s\n")
    print(f"--- {name}: exit {proc.returncode} in {dt:.1f}s "
          f"(log: {log_path})", flush=True)
    return {"stage": name, "exit_code": proc.returncode,
            "seconds": round(dt, 1), "log": log_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/run2")
    ap.add_argument("--users", type=int, default=20_000)
    ap.add_argument("--days", type=int, default=28)
    ap.add_argument("--evasion", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-sim", action="store_true",
                    help="reuse existing events/features, re-run models only")
    args = ap.parse_args()

    run_name = os.path.basename(os.path.normpath(args.data))
    log_dir = os.path.join("reports", run_name, "logs")
    os.makedirs(log_dir, exist_ok=True)

    stages = []
    if not args.skip_sim:
        stages += [
            ("generate", ["sim.generate", "--users", str(args.users),
                          "--days", str(args.days), "--seed", str(args.seed),
                          "--evasion", str(args.evasion), "--out", args.data]),
            ("features", ["features.build", "--data", args.data]),
        ]
    stages += [
        ("xgboost", ["models.train_xgb", "--data", args.data]),
        ("sequence", ["models.train_seq", "--data", args.data]),
        ("gnn", ["models.train_gnn", "--data", args.data]),
        ("rings", ["models.mine_rings", "--data", args.data]),
        ("spray", ["models.spray_detect", "--data", args.data]),
        ("fusion", ["models.fuse", "--data", args.data]),
    ]

    manifest = {"data": args.data, "args": vars(args),
                "started": str(datetime.now(timezone.utc)), "stages": []}
    ok = True
    for i, (name, cmd) in enumerate(stages, 1):
        res = run_stage(name, cmd, log_dir, i)
        manifest["stages"].append(res)
        if res["exit_code"] != 0:
            ok = False
            print(f"\nSTOPPING: stage '{name}' failed (see {res['log']})")
            break

    manifest["finished"] = str(datetime.now(timezone.utc))
    manifest["success"] = ok
    with open(os.path.join("reports", run_name, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    if ok:
        print("\n=== building report ===")
        make_report(args.data)
        total = sum(s["seconds"] for s in manifest["stages"])
        print(f"\npipeline complete in {total / 60:.1f} min; "
              f"logs + report under reports/{run_name}/")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
