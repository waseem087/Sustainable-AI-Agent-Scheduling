# EcoTrust-DT: Sustainable AI-Agent Scheduling

Reproducible research codebase and IEEE conference manuscript for **EcoTrust-DT: Policy-Enforced Carbon- and Water-Aware Scheduling with Lightweight Digital-Twin Lookahead for AI-Agent Workloads**.

For environment installation instructions, see [SETUP.md](SETUP.md).

## Repository Structure
- `baseline/waterwise/` — Official `PPoPP25_WaterWise` baseline (`Apache-2.0`, Jiang et al., PPoPP '25, DOI: `10.1145/3710848.3710891`).
- `src/ecotrust_dt/` — Proposed EcoTrust-DT framework (`HardPolicyFilter`, `LightweightDigitalTwin`, `EcoTrustScheduler`, `HashChainedAuditLog`).
- `tests/` — Automated unit and invariant tests (`test_ecotrust_dt.py`).
- `experiments/` — Reproducible runners (`reproduce_waterwise.py`, `run_ecotrust_experiments.py`, `generate_figures.py`).
- `results/` — Machine-generated CSV/JSON summaries (`results/baseline/`, `results/ecotrust_dt/`) and figures (`results/figures/`).
- `main.tex` — IEEE conference manuscript (`\documentclass[conference]{IEEEtran}`) built strictly from verified local results.

## Step-by-Step Procedure to Reproduce All Results

### 1. Run Unit & Invariant Tests (~0.01 s)
Verifies hard policy enforcement (`0` violations by construction), SHA-256 audit-chain tamper detection, and all 7 scheduler arms:
```bash
python -m unittest discover -s tests -v
```

### 2. Reproduce All 70 Paired EcoTrust-DT Runs (~1–2 min)
Runs arms `A1`–`A5` across 5 seeds (`40–44`) in `normal` (12,558 requests) and `stress` (15,430 requests) regimes:
```bash
python experiments/run_ecotrust_experiments.py
```
Outputs:
- `results/ecotrust_dt/runs.csv` (all 70 individual runs)
- `results/ecotrust_dt/summary_by_arm.csv` (5-seed `mean` and `std` per arm/scenario)

### 3. Regenerate All Paper Figures (~3 s)
Generates `fig1_waterwise_baseline.pdf/.png`, `fig2_ecotrust_pareto.pdf/.png`, and `fig3_policy_and_overhead.pdf/.png`:
```bash
python experiments/generate_figures.py
```

### 4. Reproduce the Official `PPoPP25_WaterWise` Baseline
- **Quick Live Demo (~60 s, 1-day Borg `base` + `waterwise`, `seed=40`, `threshold=0.5`):**
  ```bash
  cd baseline/waterwise
  python main.py --optimizer=base --path=../../results/demo_run --node_capacity=35 --trace_name=borg --threshold=0.5 --time_length=1440 --seed=40
  python main.py --optimizer=waterwise --path=../../results/demo_run --node_capacity=35 --trace_name=borg --threshold=0.5 --time_length=1440 --seed=40
  cd ../..
  ```
- **Full 21-Run Suite (~30–35 min, Borg + Alibaba + threshold sweep + `os.listdir()` vs. numeric sort):**
  ```bash
  python experiments/reproduce_waterwise.py
  ```
  Outputs:
  - `results/baseline/environment.json`
  - `results/baseline/baseline_summary.csv`

## Baseline Attribution
`baseline/waterwise/` preserves the official [PPoPP25_WaterWise](https://github.com/LukeJYK/PPoPP25_WaterWise) artifact under its original Apache-2.0 license (`baseline/waterwise/LICENSE`).
