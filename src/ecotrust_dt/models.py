"""Core data models for EcoTrust-DT federated cloud-edge scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class DataClass(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass(frozen=True)
class SiteSpec:
    site_id: str
    region: str
    site_type: str  # "cloud" or "edge"
    gpu_capacity: int
    gpu_memory_gb: float
    supports_attestation: bool
    pue: float
    base_price_per_kwh: float
    base_latency_s: Dict[str, float]  # src_region -> network RTT in seconds


@dataclass
class SiteState:
    spec: SiteSpec
    available: bool
    drought_restricted: bool
    carbon_intensity: float  # gCO2e / kWh
    water_intensity: float   # L / kWh (combined WUE + PUE*EWIF)*(1+WSF)
    price_multiplier: float = 1.0
    slot_free_at_s: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.slot_free_at_s:
            self.slot_free_at_s = [0.0] * self.spec.gpu_capacity

    def free_slots(self, now_s: float) -> int:
        return sum(1 for t in self.slot_free_at_s if t <= now_s)

    def estimated_queue_wait_s(self, now_s: float) -> float:
        earliest = min(self.slot_free_at_s)
        return max(0.0, earliest - now_s)

    def reserve_slot(self, now_s: float, duration_s: float) -> float:
        """Assign job to the earliest-available GPU slot and return queue wait (s)."""
        idx = min(range(len(self.slot_free_at_s)), key=lambda i: self.slot_free_at_s[i])
        start_s = max(now_s, self.slot_free_at_s[idx])
        wait_s = start_s - now_s
        self.slot_free_at_s[idx] = start_s + duration_s
        return wait_s


@dataclass(frozen=True)
class AIRequest:
    request_id: str
    arrival_step: int
    arrival_s: float
    src_region: str
    workload_type: str
    gpu_memory_gb: float
    energy_kwh: float
    service_time_s: float
    slo_seconds: float
    data_class: DataClass
    allowed_regions: Tuple[str, ...]
    requires_attestation: bool
    weights: Tuple[float, float, float, float, float, float]  # alpha, beta, gamma, delta, epsilon, zeta


@dataclass
class PlacementDecision:
    request_id: str
    step: int
    selected_site: Optional[str]
    feasible_sites: List[str]
    rejected_reasons: Dict[str, List[str]]
    queue_wait_s: float
    network_latency_s: float
    service_time_s: float
    total_latency_s: float
    slo_met: bool
    predicted_carbon: float
    actual_carbon: float
    predicted_water: float
    actual_water: float
    predicted_queue_wait_s: float
    cost: float
    policy_violation: bool
