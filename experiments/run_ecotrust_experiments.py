"""Run paired EcoTrust-DT experiments (A1-A5) across 5 seeds and normal/stress scenarios."""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WW_DIR = ROOT / "baseline" / "waterwise"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(WW_DIR))

import utils as ww_utils  # noqa: E402
from ecotrust_dt import (  # noqa: E402
    AIRequest,
    DataClass,
    EcoTrustScheduler,
    SchedulerConfig,
    SiteSpec,
    SiteState,
)

OUT_DIR = ROOT / "results" / "ecotrust_dt"

CLOUD_REGIONS = ["zurich", "madrid", "oregon", "milan", "mumbai"]
REGION_JURISDICTION = {
    "zurich": "eu",
    "madrid": "eu",
    "milan": "eu",
    "oregon": "us",
    "mumbai": "apac",
}
ARMS = [
    "A1_round_robin",
    "A2_current_policy",
    "A3_ecotrust_dt",
    "A4_stale_dt",
    "A4_noisy_dt",
    "A5_carbon_only",
    "A5_water_only",
]


def load_real_env_series() -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Load real regional CI and combined WI(WSF) time series from baseline/waterwise/data."""
    ci = ww_utils.get_carbon_intensity(CLOUD_REGIONS)
    ewif = ww_utils.get_ewif(CLOUD_REGIONS)
    wue = ww_utils.get_wue(CLOUD_REGIONS)
    wsf = ww_utils.get_wsf(CLOUD_REGIONS)
    wi = ww_utils.get_wi(CLOUD_REGIONS, ewif, wsf, wue, pue=1.2)
    return ci, wi


def load_arrival_counts(steps: int = 1440) -> list[int]:
    """Load real minute-level arrival counts from Borg trace (numerically sorted)."""
    trace_dir = WW_DIR / "borg_trace"
    files = sorted(
        [f for f in trace_dir.iterdir() if f.suffix == ".csv"],
        key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0),
    )
    totals = np.zeros(steps, dtype=int)
    for fp in files:
        col = pd.read_csv(fp, header=None).iloc[:steps, 0].to_numpy(dtype=int)
        totals[: len(col)] += col
    return totals.tolist()


def build_federated_sites(
    ci: dict[str, list[float]],
    wi: dict[str, list[float]],
    step: int,
    stress: bool = False,
) -> Dict[str, SiteState]:
    """Build 5 cloud regions + 3 edge sites using real environmental telemetry."""
    rtts = {
        "eu": {"eu": 0.025, "us": 0.115, "apac": 0.160},
        "us": {"eu": 0.115, "us": 0.025, "apac": 0.175},
        "apac": {"eu": 0.160, "us": 0.175, "apac": 0.025},
    }
    specs = [
        SiteSpec("zurich-cloud", "eu", "cloud", 14, 80.0, True, 1.18, 0.14, rtts["eu"]),
        SiteSpec("madrid-cloud", "eu", "cloud", 12, 80.0, True, 1.22, 0.11, rtts["eu"]),
        SiteSpec("milan-cloud", "eu", "cloud", 12, 48.0, False, 1.24, 0.12, rtts["eu"]),
        SiteSpec("oregon-cloud", "us", "cloud", 16, 80.0, True, 1.15, 0.09, rtts["us"]),
        SiteSpec("mumbai-cloud", "apac", "cloud", 12, 48.0, False, 1.28, 0.08, rtts["apac"]),
        SiteSpec("zurich-edge", "eu", "edge", 4, 24.0, True, 1.32, 0.16, {"eu": 0.008, "us": 0.120, "apac": 0.165}),
        SiteSpec("oregon-edge", "us", "edge", 4, 24.0, True, 1.30, 0.11, {"eu": 0.120, "us": 0.008, "apac": 0.180}),
        SiteSpec("mumbai-edge", "apac", "edge", 4, 24.0, False, 1.35, 0.10, {"eu": 0.165, "us": 0.180, "apac": 0.008}),
    ]
    cloud_map = {
        "zurich-cloud": "zurich",
        "madrid-cloud": "madrid",
        "milan-cloud": "milan",
        "oregon-cloud": "oregon",
        "mumbai-cloud": "mumbai",
        "zurich-edge": "zurich",
        "oregon-edge": "oregon",
        "mumbai-edge": "mumbai",
    }
    states: Dict[str, SiteState] = {}
    for sp in specs:
        cname = cloud_map[sp.site_id]
        c_val = float(ci[cname][step])
        w_val = float(wi[cname][step])
        avail = True
        drought = False
        p_mult = 1.0
        if stress:
            # Carbon spike in Zurich (steps 400-600)
            if cname == "zurich" and 400 <= step < 600:
                c_val *= 2.4
            # Drought restriction in Madrid (steps 500-800)
            if cname == "madrid" and 500 <= step < 800:
                drought = True
            # Outage in Oregon cloud (steps 700-760)
            if sp.site_id == "oregon-cloud" and 700 <= step < 760:
                avail = False
            # Price surge in EU during evening peak (steps 900-1100)
            if sp.region == "eu" and 900 <= step < 1100:
                p_mult = 1.5
        states[sp.site_id] = SiteState(
            spec=sp,
            available=avail,
            drought_restricted=drought,
            carbon_intensity=c_val,
            water_intensity=w_val,
            price_multiplier=p_mult,
        )
    return states


def update_site_telemetry(
    sites: Dict[str, SiteState],
    ci: dict[str, list[float]],
    wi: dict[str, list[float]],
    step: int,
    stress: bool,
) -> None:
    fresh = build_federated_sites(ci, wi, step, stress=stress)
    for sid, st in sites.items():
        f = fresh[sid]
        st.available = f.available
        st.drought_restricted = f.drought_restricted
        st.carbon_intensity = f.carbon_intensity
        st.water_intensity = f.water_intensity
        st.price_multiplier = f.price_multiplier


def generate_workload(arrivals_per_min: list[int], seed: int, stress: bool) -> list[AIRequest]:
    """Generate deterministic AI-agent workload with explicit synthetic privacy/SLO labels."""
    rng = random.Random(seed)
    reqs: list[AIRequest] = []
    data_classes = [
        DataClass.PUBLIC,
        DataClass.INTERNAL,
        DataClass.CONFIDENTIAL,
        DataClass.RESTRICTED,
    ]
    dc_weights = [0.35, 0.30, 0.25, 0.10]
    src_regions = ["eu", "us", "apac"]
    src_weights = [0.45, 0.35, 0.20]

    req_idx = 0
    for step, raw_n in enumerate(arrivals_per_min):
        n = max(1, int(round(raw_n * 0.75)))
        if stress and ((300 <= step < 450) or (900 <= step < 1050)):
            n = int(round(n * 2.2))
        for k in range(n):
            dc = rng.choices(data_classes, weights=dc_weights, k=1)[0]
            src = rng.choices(src_regions, weights=src_weights, k=1)[0]
            if dc in (DataClass.CONFIDENTIAL, DataClass.RESTRICTED):
                allowed = (src,)
                attest = True
            elif dc == DataClass.INTERNAL:
                allowed = (src, "eu") if src != "eu" else ("eu", "us")
                attest = rng.random() < 0.4
            else:
                allowed = ("eu", "us", "apac")
                attest = False

            is_batch = rng.random() < 0.25
            if is_batch:
                wtype = "agent_batch_rag"
                gpu_mem = rng.choice([24.0, 48.0])
                svc_s = rng.uniform(12.0, 35.0)
                slo_s = rng.uniform(60.0, 120.0)
                energy_kwh = svc_s * (0.00032 if gpu_mem == 48.0 else 0.00022)
                weights = (0.35, 0.35, 0.05, 0.10, 0.10, 0.05)
            else:
                wtype = "llm_inference"
                gpu_mem = rng.choice([16.0, 24.0])
                svc_s = rng.uniform(1.8, 7.5)
                slo_s = rng.uniform(10.0, 25.0)
                energy_kwh = svc_s * 0.00024
                weights = (0.28, 0.28, 0.18, 0.06, 0.14, 0.06)

            arrival_s = step * 60.0 + (k / max(1, n)) * 59.0
            reqs.append(
                AIRequest(
                    request_id=f"req_{seed}_{step}_{k}_{req_idx}",
                    arrival_step=step,
                    arrival_s=arrival_s,
                    src_region=src,
                    workload_type=wtype,
                    gpu_memory_gb=gpu_mem,
                    energy_kwh=energy_kwh,
                    service_time_s=svc_s,
                    slo_seconds=slo_s,
                    data_class=dc,
                    allowed_regions=allowed,
                    requires_attestation=attest,
                    weights=weights,
                )
            )
            req_idx += 1
    return reqs


def run_arm(
    arm: str,
    requests: list[AIRequest],
    ci: dict[str, list[float]],
    wi: dict[str, list[float]],
    seed: int,
    stress: bool,
) -> dict[str, Any]:
    sites = build_federated_sites(ci, wi, 0, stress=stress)
    sched = EcoTrustScheduler(sites, SchedulerConfig(arm=arm, seed=seed))  # type: ignore[arg-type]

    by_step: dict[int, list[AIRequest]] = {}
    for r in requests:
        by_step.setdefault(r.arrival_step, []).append(r)

    for step in range(1440):
        update_site_telemetry(sites, ci, wi, step, stress=stress)
        sched.sync_telemetry(step * 60.0)
        for req in by_step.get(step, []):
            sched.schedule_request(req, req.arrival_s)

    decs = sched.decisions
    completed = [d for d in decs if d.selected_site is not None]
    n_comp = len(completed)
    n_tot = len(decs)
    lats = np.array([d.total_latency_s for d in completed], dtype=float)
    total_c = float(sum(d.actual_carbon for d in completed))
    total_w = float(sum(d.actual_water for d in completed))
    total_cost = float(sum(d.cost for d in completed))
    slo_met = sum(1 for d in completed if d.slo_met)
    violations = sum(1 for d in completed if d.policy_violation)
    audit_ok, _ = sched.audit.verify_chain() if sched.audit else (False, "disabled")
    mae = sched.twin.mae_summary()

    return {
        "scenario": "stress" if stress else "normal",
        "arm": arm,
        "seed": seed,
        "total_requests": n_tot,
        "completed_requests": n_comp,
        "rejected_requests": n_tot - n_comp,
        "carbon_per_1k": round(total_c / max(1, n_comp) * 1000.0, 4),
        "water_per_1k": round(total_w / max(1, n_comp) * 1000.0, 4),
        "cost_per_1k": round(total_cost / max(1, n_comp) * 1000.0, 4),
        "mean_latency_s": round(float(np.mean(lats)), 4) if n_comp else 0.0,
        "p50_latency_s": round(float(np.percentile(lats, 50)), 4) if n_comp else 0.0,
        "p95_latency_s": round(float(np.percentile(lats, 95)), 4) if n_comp else 0.0,
        "p99_latency_s": round(float(np.percentile(lats, 99)), 4) if n_comp else 0.0,
        "slo_attainment_pct": round(slo_met / max(1, n_tot) * 100.0, 3),
        "policy_violations": violations,
        "audit_verified": audit_ok,
        "audit_records": len(sched.audit.entries) if sched.audit else 0,
        "mae_carbon": round(mae["mae_carbon"], 6),
        "mae_water": round(mae["mae_water"], 6),
        "mae_queue_s": round(mae["mae_queue_s"], 4),
        "mean_decision_ms": round(float(np.mean(sched.decision_times_ms)), 4),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ci, wi = load_real_env_series()
    arrivals = load_arrival_counts(1440)
    seeds = [40, 41, 42, 43, 44]

    all_rows: list[dict[str, Any]] = []
    for stress in (False, True):
        sc_name = "stress" if stress else "normal"
        for seed in seeds:
            reqs = generate_workload(arrivals, seed=seed, stress=stress)
            print(f"Scenario={sc_name} seed={seed} requests={len(reqs)}", flush=True)
            for arm in ARMS:
                row = run_arm(arm, reqs, ci, wi, seed=seed, stress=stress)
                all_rows.append(row)
                print(
                    f"  {arm:18s} | C/1k={row['carbon_per_1k']:8.3f} | W/1k={row['water_per_1k']:6.3f} "
                    f"| p95={row['p95_latency_s']:6.2f}s | SLO={row['slo_attainment_pct']:6.2f}% "
                    f"| viol={row['policy_violations']:4d} | dec={row['mean_decision_ms']:.3f}ms",
                    flush=True,
                )

    runs_csv = OUT_DIR / "runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    df = pd.DataFrame(all_rows)
    num_cols = [
        "completed_requests",
        "rejected_requests",
        "carbon_per_1k",
        "water_per_1k",
        "cost_per_1k",
        "mean_latency_s",
        "p95_latency_s",
        "p99_latency_s",
        "slo_attainment_pct",
        "policy_violations",
        "mae_carbon",
        "mae_water",
        "mae_queue_s",
        "mean_decision_ms",
    ]
    summary = df.groupby(["scenario", "arm"], sort=False)[num_cols].agg(["mean", "std"]).round(4)
    summary.to_csv(OUT_DIR / "summary_by_arm.csv")
    print(f"Saved {len(all_rows)} runs to {runs_csv}")


if __name__ == "__main__":
    main()
