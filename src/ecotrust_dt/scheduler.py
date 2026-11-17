"""Multi-objective policy-safe scheduler supporting paired arms A1-A5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal
import time

from .audit import HashChainedAuditLog
from .digital_twin import LightweightDigitalTwin, TwinMode
from .models import AIRequest, PlacementDecision, SiteState
from .policy_filter import HardPolicyFilter

ArmName = Literal[
    "A1_round_robin",
    "A2_current_policy",
    "A3_ecotrust_dt",
    "A4_stale_dt",
    "A4_noisy_dt",
    "A5_carbon_only",
    "A5_water_only",
]


@dataclass
class SchedulerConfig:
    arm: ArmName = "A3_ecotrust_dt"
    horizon_steps: int = 5
    stale_lag: int = 10
    noise_std_frac: float = 0.25
    seed: int = 40
    enable_audit: bool = True


class EcoTrustScheduler:
    """Executes paired scheduling arms over identical arrivals and telemetry."""

    def __init__(self, sites: Dict[str, SiteState], config: SchedulerConfig) -> None:
        self.sites = sites
        self.config = config
        self._rr_idx = 0
        self._site_ids = list(sites.keys())

        twin_mode: TwinMode = "current"
        if config.arm == "A3_ecotrust_dt":
            twin_mode = "lookahead"
        elif config.arm == "A4_stale_dt":
            twin_mode = "stale"
        elif config.arm == "A4_noisy_dt":
            twin_mode = "noisy"

        self.twin = LightweightDigitalTwin(
            horizon_steps=config.horizon_steps,
            mode=twin_mode,
            stale_lag=config.stale_lag,
            noise_std_frac=config.noise_std_frac,
            seed=config.seed,
        )
        self.audit = HashChainedAuditLog() if config.enable_audit else None
        self.decisions: List[PlacementDecision] = []
        self.decision_times_ms: List[float] = []

    @staticmethod
    def _minmax(values: Dict[str, float]) -> Dict[str, float]:
        if not values:
            return {}
        lo = min(values.values())
        hi = max(values.values())
        span = hi - lo
        if span <= 1e-12:
            return {k: 0.0 for k in values}
        return {k: (v - lo) / span for k, v in values.items()}

    def sync_telemetry(self, now_s: float) -> None:
        self.twin.sync_step(self.sites, now_s)

    def schedule_request(self, req: AIRequest, now_s: float) -> PlacementDecision:
        t0 = time.perf_counter()
        arm = self.config.arm

        # A1_round_robin: non-policy, non-sustainability sanity baseline
        if arm == "A1_round_robin":
            avail_ids = [sid for sid, st in self.sites.items() if st.available]
            if not avail_ids:
                chosen = None
                feasible = []
                rejected = {sid: ["site_outage"] for sid in self.sites}
            else:
                chosen = avail_ids[self._rr_idx % len(avail_ids)]
                self._rr_idx += 1
                feasible = avail_ids
                rejected = {}
            return self._finalize_decision(req, chosen, feasible, rejected, {}, now_s, t0)

        # All other arms (A2-A5) enforce HardPolicyFilter first
        preds = {sid: self.twin.predict_site(sid, st, now_s) for sid, st in self.sites.items()}
        pred_waits = {sid: preds[sid]["queue_wait_s"] for sid in self.sites}

        feasible, rejected = HardPolicyFilter.filter_sites(
            req, self.sites, now_s, predicted_waits_s=pred_waits, enforce_slo_filter=True
        )
        # If SLO filter leaves zero sites, relax only SLO (keep residency, security, drought, outage hard)
        if not feasible:
            feasible, rejected = HardPolicyFilter.filter_sites(
                req, self.sites, now_s, predicted_waits_s=pred_waits, enforce_slo_filter=False
            )

        if not feasible:
            return self._finalize_decision(req, None, feasible, rejected, preds, now_s, t0)

        # Compute raw candidate metrics on feasible set S(r)
        c_raw, w_raw, l_raw, p_raw, q_raw, r_raw = {}, {}, {}, {}, {}, {}
        for sid in feasible:
            st = self.sites[sid]
            pr = preds[sid]
            net_s = st.spec.base_latency_s.get(req.src_region, 0.05)
            c_raw[sid] = req.energy_kwh * pr["carbon_intensity"]
            w_raw[sid] = req.energy_kwh * pr["water_intensity"]
            q_raw[sid] = pr["queue_wait_s"]
            l_raw[sid] = net_s + pr["queue_wait_s"] + req.service_time_s
            p_raw[sid] = req.energy_kwh * pr["price"]
            r_raw[sid] = pr["risk"]

        c_n = self._minmax(c_raw)
        w_n = self._minmax(w_raw)
        l_n = self._minmax(l_raw)
        p_n = self._minmax(p_raw)
        q_n = self._minmax(q_raw)
        r_n = self._minmax(r_raw)

        if arm == "A5_carbon_only":
            alpha, beta, gamma, delta, eps, zeta = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        elif arm == "A5_water_only":
            alpha, beta, gamma, delta, eps, zeta = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        else:
            alpha, beta, gamma, delta, eps, zeta = req.weights

        scores = {
            sid: (
                alpha * c_n[sid]
                + beta * w_n[sid]
                + gamma * l_n[sid]
                + delta * p_n[sid]
                + eps * q_n[sid]
                + zeta * r_n[sid]
            )
            for sid in feasible
        }
        chosen = min(feasible, key=lambda sid: (scores[sid], sid))
        return self._finalize_decision(req, chosen, feasible, rejected, preds, now_s, t0)

    def _finalize_decision(
        self,
        req: AIRequest,
        chosen: str | None,
        feasible: List[str],
        rejected: Dict[str, List[str]],
        preds: Dict[str, Dict[str, float]],
        now_s: float,
        t0: float,
    ) -> PlacementDecision:
        if chosen is None:
            dec = PlacementDecision(
                request_id=req.request_id,
                step=req.arrival_step,
                selected_site=None,
                feasible_sites=feasible,
                rejected_reasons=rejected,
                queue_wait_s=0.0,
                network_latency_s=0.0,
                service_time_s=0.0,
                total_latency_s=0.0,
                slo_met=False,
                predicted_carbon=0.0,
                actual_carbon=0.0,
                predicted_water=0.0,
                actual_water=0.0,
                predicted_queue_wait_s=0.0,
                cost=0.0,
                policy_violation=False,
            )
        else:
            st = self.sites[chosen]
            net_s = st.spec.base_latency_s.get(req.src_region, 0.05)
            q_wait = st.reserve_slot(now_s, req.service_time_s)
            total_lat = net_s + q_wait + req.service_time_s
            actual_c = req.energy_kwh * st.carbon_intensity
            actual_w = req.energy_kwh * st.water_intensity
            cost = req.energy_kwh * st.spec.base_price_per_kwh * st.price_multiplier
            pr = preds.get(chosen, {})
            pred_c = req.energy_kwh * pr.get("carbon_intensity", st.carbon_intensity)
            pred_w = req.energy_kwh * pr.get("water_intensity", st.water_intensity)
            pred_q = pr.get("queue_wait_s", q_wait)
            violation = not HardPolicyFilter.check_invariants(req, st)

            self.twin.record_feedback(pred_c, actual_c, pred_w, actual_w, pred_q, q_wait)
            dec = PlacementDecision(
                request_id=req.request_id,
                step=req.arrival_step,
                selected_site=chosen,
                feasible_sites=feasible,
                rejected_reasons=rejected,
                queue_wait_s=q_wait,
                network_latency_s=net_s,
                service_time_s=req.service_time_s,
                total_latency_s=total_lat,
                slo_met=(total_lat <= req.slo_seconds),
                predicted_carbon=pred_c,
                actual_carbon=actual_c,
                predicted_water=pred_w,
                actual_water=actual_w,
                predicted_queue_wait_s=pred_q,
                cost=cost,
                policy_violation=violation,
            )

        if self.audit is not None:
            self.audit.append(
                request_id=req.request_id,
                step=req.arrival_step,
                data_class=req.data_class.value,
                feasible_sites=feasible,
                rejected_count=len(rejected),
                selected_site=dec.selected_site,
                slo_met=dec.slo_met,
                policy_violation=dec.policy_violation,
                predicted_carbon=dec.predicted_carbon,
                actual_carbon=dec.actual_carbon,
                predicted_water=dec.predicted_water,
                actual_water=dec.actual_water,
            )

        self.decisions.append(dec)
        self.decision_times_ms.append((time.perf_counter() - t0) * 1000.0)
        return dec
