# EcoTrust-DT: Sustainable AI-Agent Scheduling

Reproducible research codebase and IEEE manuscript for **EcoTrust-DT: Digital-Twin Enabled Multi-Objective Carbon and Water-Aware Scheduling with Policy Enforcement for Privacy-Sensitive AI-Agent Workloads in Federated Cloud--Edge Systems**.

## Repository Structure
- `baseline/waterwise/` — Official `PPoPP25_WaterWise` baseline (`Apache-2.0`, attributed to Jiang et al., PPoPP '25, DOI: `10.1145/3710848.3710891`).
- `src/ecotrust_dt/` — Proposed EcoTrust-DT framework (Hard Policy Filter, Lightweight Digital Twin, Multi-Objective EcoScore Scheduler, Hash-Chained Audit Layer). *(Added in Step 4)*
- `experiments/` — Reproducible experiment runners, paired arm evaluations (`A0`–`A5`), stress scenarios, and plotting scripts. *(Added in Steps 3 & 5)*
- `results/` — Machine-generated JSON/CSV logs and figures from real local runs. *(Added in Steps 3 & 5)*
- `main.tex` — IEEE conference manuscript built strictly from verified, reproducible results.

## Baseline Attribution
The `baseline/waterwise/` directory preserves the official [PPoPP25_WaterWise](https://github.com/LukeJYK/PPoPP25_WaterWise) artifact under its original Apache-2.0 license (`baseline/waterwise/LICENSE`).
