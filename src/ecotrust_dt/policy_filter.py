"""Hard non-negotiable policy filter for residency, security, availability, water, and SLO."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .models import AIRequest, DataClass, SiteState


class HardPolicyFilter:
    """Evaluates F(i, r) = Residency & Security & Availability & WaterPolicy & LatencySLO."""

    @staticmethod
    def filter_sites(
        request: AIRequest,
        sites: Dict[str, SiteState],
        now_s: float,
        predicted_waits_s: Dict[str, float] | None = None,
        enforce_slo_filter: bool = True,
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        feasible: List[str] = []
        rejected: Dict[str, List[str]] = {}

        for site_id, state in sites.items():
            reasons: List[str] = []
            spec = state.spec

            # 1. Availability & GPU hardware capability
            if not state.available:
                reasons.append("site_outage")
            if spec.gpu_memory_gb < request.gpu_memory_gb:
                reasons.append("insufficient_gpu_memory")

            # 2. Data Residency
            if spec.region not in request.allowed_regions:
                reasons.append("residency_violation")

            # 3. Security & Confidential Computing Attestation
            needs_attest = request.requires_attestation or request.data_class in (
                DataClass.CONFIDENTIAL,
                DataClass.RESTRICTED,
            )
            if needs_attest and not spec.supports_attestation:
                reasons.append("missing_attestation")

            # 4. Water-Stress / Drought Policy
            if state.drought_restricted:
                reasons.append("drought_restriction")

            # 5. Latency SLO Guardrail
            if enforce_slo_filter:
                net_s = spec.base_latency_s.get(request.src_region, 0.05)
                q_wait = (
                    predicted_waits_s[site_id]
                    if predicted_waits_s is not None and site_id in predicted_waits_s
                    else state.estimated_queue_wait_s(now_s)
                )
                est_total_s = net_s + q_wait + request.service_time_s
                if est_total_s > request.slo_seconds:
                    reasons.append("slo_exceeded")

            if not reasons:
                feasible.append(site_id)
            else:
                rejected[site_id] = reasons

        return feasible, rejected

    @staticmethod
    def check_invariants(request: AIRequest, chosen_state: SiteState) -> bool:
        """Return True if chosen_state satisfies all hard non-SLO invariants (must never fail)."""
        spec = chosen_state.spec
        if not chosen_state.available:
            return False
        if spec.gpu_memory_gb < request.gpu_memory_gb:
            return False
        if spec.region not in request.allowed_regions:
            return False
        needs_attest = request.requires_attestation or request.data_class in (
            DataClass.CONFIDENTIAL,
            DataClass.RESTRICTED,
        )
        if needs_attest and not spec.supports_attestation:
            return False
        if chosen_state.drought_restricted:
            return False
        return True
