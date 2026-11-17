"""Generate publication-grade IEEE figures directly from real experimental CSV logs."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 300,
        }
    )

    # ------------------------------------------------------------------
    # Fig 1: WaterWise Baseline Reproduction (Threshold sweep + Ordering)
    # ------------------------------------------------------------------
    bdf = pd.read_csv(ROOT / "results" / "baseline" / "baseline_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))

    # Panel A: Borg threshold sweep (seed=40, os_listdir)
    ww_sweep = bdf[
        (bdf["trace"] == "borg")
        & (bdf["optimizer"] == "waterwise")
        & (bdf["seed"] == 40)
        & (bdf["trace_order"] == "os_listdir")
    ].sort_values("threshold")
    ax1 = axes[0]
    ax1_r = ax1.twinx()
    l1 = ax1.plot(ww_sweep["threshold"], ww_sweep["total_carbon"] / 1000.0, "o-", color="#1f77b4", label="Carbon (k-units)")
    l2 = ax1_r.plot(ww_sweep["threshold"], ww_sweep["total_water"], "s--", color="#2ca02c", label="Water (units)")
    ax1.set_xlabel("Delay Tolerance Threshold")
    ax1.set_ylabel("Total Carbon (x10^3)", color="#1f77b4")
    ax1_r.set_ylabel("Total Water", color="#2ca02c")
    ax1.set_title("(a) WaterWise Delay-Threshold Sweep (Borg)")
    ax1.grid(True, linestyle=":", alpha=0.5)
    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right", frameon=True)

    # Panel B: Borg vs Alibaba (seed 40-42 mean +/- std)
    ax2 = axes[1]
    groups = [
        ("Borg\nBase", bdf[(bdf["trace"] == "borg") & (bdf["optimizer"] == "base") & (bdf["trace_order"] == "os_listdir")]),
        ("Borg\nWaterWise", bdf[(bdf["trace"] == "borg") & (bdf["optimizer"] == "waterwise") & (bdf["threshold"] == 0.5) & (bdf["trace_order"] == "os_listdir")]),
        ("Alibaba\nBase", bdf[(bdf["trace"] == "alibaba") & (bdf["optimizer"] == "base")]),
        ("Alibaba\nWaterWise", bdf[(bdf["trace"] == "alibaba") & (bdf["optimizer"] == "waterwise") & (bdf["threshold"] == 0.5)]),
    ]
    x = np.arange(len(groups))
    c_means = [g[1]["carbon_per_job"].mean() for g in groups]
    c_stds = [g[1]["carbon_per_job"].std() for g in groups]
    ax2.bar(x, c_means, yerr=c_stds, capsize=3, color=["#7f7f7f", "#1f77b4", "#7f7f7f", "#1f77b4"], width=0.55)
    ax2.set_xticks(x)
    ax2.set_xticklabels([g[0] for g in groups])
    ax2.set_ylabel("Carbon per Completed Job")
    ax2.set_title("(b) Multi-Seed Baseline (Borg & Alibaba)")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_waterwise_baseline.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_waterwise_baseline.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 2: EcoTrust-DT Carbon-Water Pareto & Policy Violations (Normal vs Stress)
    # ------------------------------------------------------------------
    edf = pd.read_csv(ROOT / "results" / "ecotrust_dt" / "runs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    arm_labels = {
        "A1_round_robin": "A1 (RR)",
        "A2_current_policy": "A2 (Current)",
        "A3_ecotrust_dt": "A3 (EcoTrust-DT)",
        "A4_stale_dt": "A4 (Stale)",
        "A4_noisy_dt": "A4 (Noisy)",
        "A5_carbon_only": "A5 (C-Only)",
        "A5_water_only": "A5 (W-Only)",
    }

    # Panel A: Carbon vs Water trade-off (Normal, policy-filtered A2-A5)
    ax = axes[0]
    norm_df = edf[(edf["scenario"] == "normal") & (edf["arm"] != "A1_round_robin")]
    agg = norm_df.groupby("arm", sort=False)[["carbon_per_1k", "water_per_1k"]].agg(["mean", "std"])
    colors = ["#ff7f0e", "#1f77b4", "#9467bd", "#8c564b", "#2ca02c", "#d62728"]
    for idx, (arm, row) in enumerate(agg.iterrows()):
        ax.errorbar(
            row[("carbon_per_1k", "mean")],
            row[("water_per_1k", "mean")],
            xerr=row[("carbon_per_1k", "std")],
            yerr=row[("water_per_1k", "std")],
            fmt="o",
            capsize=3,
            color=colors[idx % len(colors)],
            label=arm_labels[arm],
        )
    ax.set_xlabel("Carbon per 1,000 Requests")
    ax.set_ylabel("Water per 1,000 Requests")
    ax.set_title("(a) Policy-Safe Carbon–Water Trade-Off (Normal)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=7, frameon=True)

    # Panel B: Carbon per 1k across all arms (Normal vs Stress)
    ax = axes[1]
    arms_order = list(arm_labels.keys())
    x = np.arange(len(arms_order))
    w = 0.36
    norm_means = [edf[(edf["scenario"] == "normal") & (edf["arm"] == a)]["carbon_per_1k"].mean() for a in arms_order]
    stress_means = [edf[(edf["scenario"] == "stress") & (edf["arm"] == a)]["carbon_per_1k"].mean() for a in arms_order]
    ax.bar(x - w / 2, norm_means, w, label="Normal", color="#1f77b4")
    ax.bar(x + w / 2, stress_means, w, label="Stress", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([arm_labels[a].split(" ")[0] for a in arms_order])
    ax.set_ylabel("Carbon per 1,000 Requests")
    ax.set_title("(b) Carbon Footprint Across Arms (Normal vs. Stress)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_ecotrust_pareto.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_ecotrust_pareto.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 3: Policy Violations & Decision Overhead
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axes[0]
    viol_norm = [edf[(edf["scenario"] == "normal") & (edf["arm"] == a)]["policy_violations"].mean() for a in arms_order]
    viol_stress = [edf[(edf["scenario"] == "stress") & (edf["arm"] == a)]["policy_violations"].mean() for a in arms_order]
    ax.bar(x - w / 2, viol_norm, w, label="Normal", color="#ff7f0e")
    ax.bar(x + w / 2, viol_stress, w, label="Stress", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([arm_labels[a].split(" ")[0] for a in arms_order])
    ax.set_ylabel("Mean Policy Violations / Run")
    ax.set_title("(a) Hard Policy Enforcement (0 Violations in A2–A5)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)

    ax = axes[1]
    oh_means = [edf[edf["arm"] == a]["mean_decision_ms"].mean() for a in arms_order]
    oh_stds = [edf[edf["arm"] == a]["mean_decision_ms"].std() for a in arms_order]
    ax.bar(x, oh_means, yerr=oh_stds, capsize=3, color="#2ca02c", width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([arm_labels[a].split(" ")[0] for a in arms_order])
    ax.set_ylabel("Mean Decision Time (ms)")
    ax.set_title("(b) Per-Request Scheduling Overhead")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_policy_and_overhead.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_policy_and_overhead.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated 3 figures in", FIG_DIR)


if __name__ == "__main__":
    main()
