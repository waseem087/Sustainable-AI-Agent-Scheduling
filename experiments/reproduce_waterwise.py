"""Reproduce the PPoPP25_WaterWise baseline on the local host and summarize real logs.

Runs the unmodified WaterWise optimizers (base, carbon_opt, water_opt, waterwise)
from baseline/waterwise/ under documented hardware/software conditions, plus a
controlled diagnostic comparing default os.listdir() trace ordering vs. numeric
app1..app10 sorting.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pulp

ROOT = Path(__file__).resolve().parents[1]
WW_DIR = ROOT / "baseline" / "waterwise"
RAW_OUT_DIR = ROOT / "results" / "baseline_raw"
SUMMARY_DIR = ROOT / "results" / "baseline"


def collect_environment() -> dict[str, Any]:
    """Capture actual host hardware and Python environment metadata."""
    cpu_name = platform.processor() or "Unknown"
    ram_gb = None
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name); "
                    "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)",
                ],
                text=True,
                timeout=10,
            ).strip().splitlines()
            if len(out) >= 2:
                cpu_name = out[0].strip()
                ram_gb = float(out[1].strip())
        except Exception:
            pass
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_name,
        "logical_cpus": os.cpu_count(),
        "ram_gb": ram_gb,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pulp": pulp.__version__,
        "solver": "PuLP PULP_CBC_CMD",
    }


def parse_waterwise_outputs(out_dir: Path, optimizer: str, threshold: float) -> dict[str, Any]:
    """Parse carbon, water, latency, and optional overhead JSON logs."""
    opt_dir = out_dir / optimizer
    if optimizer == "base":
        c_path = opt_dir / "carbon.json"
        w_path = opt_dir / "water.json"
        t_path = opt_dir / "time.json"
        oh_path = None
        cw_idx = 2
        lat_idx = 3  # base.py: (seq, dst, mean_exe, sample_exe + transfer + 60*(j - start))
    elif optimizer in ("carbon_opt", "water_opt"):
        c_path = opt_dir / f"carbon_{1 + threshold}.json"
        w_path = opt_dir / f"water_{1 + threshold}.json"
        t_path = opt_dir / f"time_{1 + threshold}.json"
        oh_path = opt_dir / f"overhead_{1 + threshold}.json"
        cw_idx = 1  # (region, value)
        lat_idx = 2  # (region, transfer+exe, transfer+exe+queue)
    else:
        c_path = opt_dir / f"carbon_{1 + threshold}.json"
        w_path = opt_dir / f"water_{1 + threshold}.json"
        t_path = opt_dir / f"time_{1 + threshold}.json"
        oh_path = opt_dir / f"overhead_{1 + threshold}.json"
        cw_idx = 2  # (seq, region, value)
        lat_idx = 4  # waterwise: index 4 includes queueing delay

    c_data = json.loads(c_path.read_text(encoding="utf-8"))
    w_data = json.loads(w_path.read_text(encoding="utf-8"))
    t_data = json.loads(t_path.read_text(encoding="utf-8"))

    carbons = [float(item[cw_idx]) for step in c_data.values() for item in step]
    waters = [float(item[cw_idx]) for step in w_data.values() for item in step]
    latencies = [float(item[lat_idx]) for step in t_data.values() for item in step]

    mean_oh_ms = None
    if oh_path and oh_path.exists():
        oh_vals = json.loads(oh_path.read_text(encoding="utf-8"))
        if oh_vals:
            mean_oh_ms = float(np.mean(oh_vals) * 1000.0)

    n_jobs = len(carbons)
    total_c = float(np.sum(carbons))
    total_w = float(np.sum(waters))
    lat_arr = np.array(latencies, dtype=float)
    return {
        "completed_jobs": n_jobs,
        "total_carbon": round(total_c, 6),
        "carbon_per_job": round(total_c / n_jobs, 6) if n_jobs else 0.0,
        "total_water": round(total_w, 6),
        "water_per_job": round(total_w / n_jobs, 6) if n_jobs else 0.0,
        "mean_latency_s": round(float(np.mean(lat_arr)), 6) if n_jobs else 0.0,
        "p95_latency_s": round(float(np.percentile(lat_arr, 95)), 6) if n_jobs else 0.0,
        "p99_latency_s": round(float(np.percentile(lat_arr, 99)), 6) if n_jobs else 0.0,
        "mean_step_overhead_ms": round(mean_oh_ms, 4) if mean_oh_ms is not None else None,
    }


def run_one(
    optimizer: str,
    trace_name: str,
    threshold: float,
    seed: int,
    time_length: int = 1440,
    node_capacity: int = 35,
    numeric_sort: bool = False,
) -> dict[str, Any]:
    """Execute one WaterWise run in a clean subprocess and parse its outputs."""
    tag = f"{trace_name}_{optimizer}_th{threshold}_s{seed}_{'numsort' if numeric_sort else 'listdir'}"
    run_dir = RAW_OUT_DIR / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    if numeric_sort:
        # Inline wrapper that overrides only utils.read_selected_traces to sort app1..app10 numerically
        code = (
            "import os, re, pandas as pd, utils, fire, main\n"
            "from pathlib import Path\n"
            "def sorted_traces(which_trace):\n"
            "    app_lst = ['black','cache','canneal','dedup','graph','memory','serving','media','netdedup','swaptions']\n"
            "    d = f'{Path(utils.__file__).parents[0]}/{which_trace}_trace/'\n"
            "    files = sorted([f for f in os.listdir(d) if f.endswith('.csv')], "
            "key=lambda x: int(re.search(r'(\\d+)', x).group(1)))\n"
            "    return [pd.read_csv(os.path.join(d, f), header=None).iloc[:, 0].tolist() for f in files], app_lst\n"
            "utils.read_selected_traces = sorted_traces\n"
            "main.read_selected_traces = sorted_traces\n"
            "fire.Fire(main.main)\n"
        )
        cmd = [
            sys.executable,
            "-c",
            code,
            f"--optimizer={optimizer}",
            f"--path={run_dir.as_posix()}",
            f"--node_capacity={node_capacity}",
            f"--trace_name={trace_name}",
            f"--threshold={threshold}",
            f"--time_length={time_length}",
            f"--seed={seed}",
        ]
    else:
        cmd = [
            sys.executable,
            "main.py",
            f"--optimizer={optimizer}",
            f"--path={run_dir.as_posix()}",
            f"--node_capacity={node_capacity}",
            f"--trace_name={trace_name}",
            f"--threshold={threshold}",
            f"--time_length={time_length}",
            f"--seed={seed}",
        ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=WW_DIR, capture_output=True, text=True)
    wall_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Run failed ({tag}):\n{proc.stderr}")

    metrics = parse_waterwise_outputs(run_dir, optimizer, threshold)
    return {
        "trace": trace_name,
        "optimizer": optimizer,
        "threshold": threshold,
        "seed": seed,
        "time_length_min": time_length,
        "trace_order": "numeric_sorted" if numeric_sort else "os_listdir",
        "wall_time_s": round(wall_s, 3),
        **metrics,
    }


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)

    env = collect_environment()
    (SUMMARY_DIR / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print("Environment:", json.dumps(env))

    runs: list[tuple[str, str, float, int, bool]] = [
        # 1. Diagnostic: os.listdir() vs numeric sort on Borg (1 day, seed=40, th=0.5)
        ("base", "borg", 0.5, 40, False),
        ("waterwise", "borg", 0.5, 40, False),
        ("base", "borg", 0.5, 40, True),
        ("waterwise", "borg", 0.5, 40, True),
        # 2. Threshold sweep on Borg (seed=40, default listdir)
        ("waterwise", "borg", 0.25, 40, False),
        ("waterwise", "borg", 0.75, 40, False),
        ("waterwise", "borg", 1.0, 40, False),
        ("carbon_opt", "borg", 0.5, 40, False),
        ("water_opt", "borg", 0.5, 40, False),
        # 3. Multi-seed on Borg (seeds 41, 42)
        ("base", "borg", 0.5, 41, False),
        ("waterwise", "borg", 0.5, 41, False),
        ("base", "borg", 0.5, 42, False),
        ("waterwise", "borg", 0.5, 42, False),
        # 4. Multi-seed on Alibaba trace (seeds 40, 41, 42)
        ("base", "alibaba", 0.5, 40, False),
        ("waterwise", "alibaba", 0.5, 40, False),
        ("carbon_opt", "alibaba", 0.5, 40, False),
        ("water_opt", "alibaba", 0.5, 40, False),
        ("base", "alibaba", 0.5, 41, False),
        ("waterwise", "alibaba", 0.5, 41, False),
        ("base", "alibaba", 0.5, 42, False),
        ("waterwise", "alibaba", 0.5, 42, False),
    ]

    rows: list[dict[str, Any]] = []
    for idx, (opt, trace, th, seed, nsort) in enumerate(runs, 1):
        print(f"[{idx}/{len(runs)}] Running {trace} | {opt} | th={th} | seed={seed} | nsort={nsort} ...", flush=True)
        row = run_one(opt, trace, th, seed, numeric_sort=nsort)
        rows.append(row)
        print(
            f"   -> jobs={row['completed_jobs']} carbon={row['total_carbon']:.2f} "
            f"water={row['total_water']:.2f} mean_lat={row['mean_latency_s']:.2f}s "
            f"p95={row['p95_latency_s']:.2f}s wall={row['wall_time_s']}s",
            flush=True,
        )

    csv_path = SUMMARY_DIR / "baseline_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
