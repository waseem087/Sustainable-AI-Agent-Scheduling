"""Tamper-evident SHA-256 hash-chained audit log for scheduling decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class AuditEntry:
    seq: int
    request_digest: str
    step: int
    data_class: str
    feasible_sites: List[str]
    rejected_count: int
    selected_site: str | None
    slo_met: bool
    policy_violation: bool
    predicted_carbon: float
    actual_carbon: float
    predicted_water: float
    actual_water: float
    prev_hash: str
    record_hash: str


class HashChainedAuditLog:
    """Stores pseudonymous placement metadata in an append-only SHA-256 hash chain."""

    GENESIS_HASH = "0" * 64

    def __init__(self) -> None:
        self.entries: List[AuditEntry] = []

    @staticmethod
    def _hash_payload(payload: Dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def pseudonymize_request(request_id: str) -> str:
        return hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:16]

    def append(
        self,
        request_id: str,
        step: int,
        data_class: str,
        feasible_sites: List[str],
        rejected_count: int,
        selected_site: str | None,
        slo_met: bool,
        policy_violation: bool,
        predicted_carbon: float,
        actual_carbon: float,
        predicted_water: float,
        actual_water: float,
    ) -> AuditEntry:
        seq = len(self.entries)
        prev_hash = self.entries[-1].record_hash if self.entries else self.GENESIS_HASH
        payload = {
            "seq": seq,
            "request_digest": self.pseudonymize_request(request_id),
            "step": step,
            "data_class": data_class,
            "feasible_sites": feasible_sites,
            "rejected_count": rejected_count,
            "selected_site": selected_site,
            "slo_met": slo_met,
            "policy_violation": policy_violation,
            "predicted_carbon": round(predicted_carbon, 8),
            "actual_carbon": round(actual_carbon, 8),
            "predicted_water": round(predicted_water, 8),
            "actual_water": round(actual_water, 8),
            "prev_hash": prev_hash,
        }
        rec_hash = self._hash_payload(payload)
        entry = AuditEntry(**payload, record_hash=rec_hash)
        self.entries.append(entry)
        return entry

    def verify_chain(self) -> Tuple[bool, str]:
        """Verify integrity of every record and link in the hash chain."""
        expected_prev = self.GENESIS_HASH
        for idx, entry in enumerate(self.entries):
            if entry.seq != idx:
                return False, f"Sequence mismatch at index {idx}"
            if entry.prev_hash != expected_prev:
                return False, f"Broken prev_hash link at seq {idx}"
            payload = asdict(entry)
            rec_hash = payload.pop("record_hash")
            if self._hash_payload(payload) != rec_hash:
                return False, f"Tampered record_hash at seq {idx}"
            expected_prev = rec_hash
        return True, f"Verified {len(self.entries)} audit records"
