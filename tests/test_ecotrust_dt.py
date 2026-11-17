"""Unit and invariant tests for EcoTrust-DT."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecotrust_dt import (
    AIRequest,
    DataClass,
    EcoTrustScheduler,
    SchedulerConfig,
    SiteSpec,
    SiteState,
)


def make_test_sites() -> dict[str, SiteState]:
    specs = [
        SiteSpec("zurich", "eu", "cloud", 4, 80.0, True, 1.2, 0.12, {"eu": 0.02, "us": 0.12}),
        SiteSpec("oregon", "us", "cloud", 4, 80.0, False, 1.2, 0.08, {"eu": 0.12, "us": 0.02}),
        SiteSpec("madrid", "eu", "cloud", 4, 40.0, True, 1.2, 0.10, {"eu": 0.03, "us": 0.13}),
    ]
    return {
        "zurich": SiteState(specs[0], available=True, drought_restricted=False, carbon_intensity=300.0, water_intensity=5.0),
        "oregon": SiteState(specs[1], available=True, drought_restricted=False, carbon_intensity=10.0, water_intensity=0.5),
        "madrid": SiteState(specs[2], available=True, drought_restricted=True, carbon_intensity=50.0, water_intensity=1.0),
    }


def make_req(req_id: str = "r1", data_class: DataClass = DataClass.CONFIDENTIAL) -> AIRequest:
    return AIRequest(
        request_id=req_id,
        arrival_step=0,
        arrival_s=0.0,
        src_region="eu",
        workload_type="llm_inference",
        gpu_memory_gb=24.0,
        energy_kwh=0.05,
        service_time_s=2.0,
        slo_seconds=10.0,
        data_class=data_class,
        allowed_regions=("eu",),
        requires_attestation=True,
        weights=(0.35, 0.35, 0.10, 0.05, 0.10, 0.05),
    )


class TestEcoTrustDT(unittest.TestCase):
    def test_hard_policy_invariant_blocks_cheapest_ineligible_site(self) -> None:
        sites = make_test_sites()
        sched = EcoTrustScheduler(sites, SchedulerConfig(arm="A3_ecotrust_dt"))
        sched.sync_telemetry(0.0)
        dec = sched.schedule_request(make_req("r_priv", DataClass.RESTRICTED), 0.0)
        # oregon violates residency & attestation; madrid violates drought_restriction -> only zurich is legal
        self.assertEqual(dec.selected_site, "zurich")
        self.assertFalse(dec.policy_violation)
        self.assertIn("residency_violation", dec.rejected_reasons["oregon"])
        self.assertIn("missing_attestation", dec.rejected_reasons["oregon"])
        self.assertIn("drought_restriction", dec.rejected_reasons["madrid"])

    def test_audit_chain_detects_tampering(self) -> None:
        sites = make_test_sites()
        sched = EcoTrustScheduler(sites, SchedulerConfig(arm="A3_ecotrust_dt"))
        sched.sync_telemetry(0.0)
        sched.schedule_request(make_req("r1"), 0.0)
        sched.schedule_request(make_req("r2"), 1.0)
        ok, _ = sched.audit.verify_chain()
        self.assertTrue(ok)
        # Tamper with record 0
        sched.audit.entries[0].actual_carbon += 1.0
        ok_after, msg = sched.audit.verify_chain()
        self.assertFalse(ok_after)
        self.assertIn("Tampered", msg)

    def test_all_arms_execute_cleanly(self) -> None:
        arms = [
            "A1_round_robin",
            "A2_current_policy",
            "A3_ecotrust_dt",
            "A4_stale_dt",
            "A4_noisy_dt",
            "A5_carbon_only",
            "A5_water_only",
        ]
        for arm in arms:
            sites = make_test_sites()
            sched = EcoTrustScheduler(sites, SchedulerConfig(arm=arm))  # type: ignore[arg-type]
            for step in range(5):
                if arm != "A1_round_robin":
                    # Add rising carbon trend on zurich
                    sites["zurich"].carbon_intensity += 15.0
                sched.sync_telemetry(step * 60.0)
                dec = sched.schedule_request(make_req(f"req_{step}"), step * 60.0)
                self.assertIsNotNone(dec.selected_site)
                if arm != "A1_round_robin":
                    self.assertFalse(dec.policy_violation)
            ok, _ = sched.audit.verify_chain()
            self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
