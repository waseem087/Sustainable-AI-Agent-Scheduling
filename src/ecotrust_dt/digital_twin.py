"""Lightweight Digital Twin: state snapshot, short-horizon predictor, and feedback MAE."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Literal
import random

from .models import SiteState

TwinMode = Literal["lookahead", "current", "stale", "noisy"]


@dataclass
class SiteHistory:
    carbon: Deque[float] = field(default_factory=lambda: deque(maxlen=15))
    water: Deque[float] = field(default_factory=lambda: deque(maxlen=15))
    queue_wait: Deque[float] = field(default_factory=lambda: deque(maxlen=15))
    price: Deque[float] = field(default_factory=lambda: deque(maxlen=15))
    outage_ema: float = 0.0
    carbon_ema: float | None = None
    water_ema: float | None = None
    queue_ema: float | None = None


class LightweightDigitalTwin:
    """Continuously synchronized Digital Twin with short-horizon Holt/EMA lookahead."""

    def __init__(
        self,
        horizon_steps: int = 5,
        ema_alpha: float = 0.40,
        trend_beta: float = 0.25,
        mode: TwinMode = "lookahead",
        stale_lag: int = 10,
        noise_std_frac: float = 0.25,
        seed: int = 40,
    ) -> None:
        self.horizon_steps = horizon_steps
        self.ema_alpha = ema_alpha
        self.trend_beta = trend_beta
        self.mode = mode
        self.stale_lag = stale_lag
        self.noise_std_frac = noise_std_frac
        self._rng = random.Random(seed)
        self.histories: Dict[str, SiteHistory] = {}

        # Feedback error accumulators
        self.n_feedback = 0
        self.abs_err_carbon = 0.0
        self.abs_err_water = 0.0
        self.abs_err_queue = 0.0

    def sync_step(self, sites: Dict[str, SiteState], now_s: float) -> None:
        """Synchronize twin state snapshot S_i(t) from live site telemetry."""
        for site_id, state in sites.items():
            h = self.histories.setdefault(site_id, SiteHistory())
            q_wait = state.estimated_queue_wait_s(now_s)
            h.carbon.append(state.carbon_intensity)
            h.water.append(state.water_intensity)
            h.queue_wait.append(q_wait)
            h.price.append(state.spec.base_price_per_kwh * state.price_multiplier)

            h.outage_ema = (
                0.2 * (0.0 if state.available else 1.0) + 0.8 * h.outage_ema
            )
            h.carbon_ema = (
                state.carbon_intensity
                if h.carbon_ema is None
                else self.ema_alpha * state.carbon_intensity + (1.0 - self.ema_alpha) * h.carbon_ema
            )
            h.water_ema = (
                state.water_intensity
                if h.water_ema is None
                else self.ema_alpha * state.water_intensity + (1.0 - self.ema_alpha) * h.water_ema
            )
            h.queue_ema = (
                q_wait
                if h.queue_ema is None
                else self.ema_alpha * q_wait + (1.0 - self.ema_alpha) * h.queue_ema
            )

    def _forecast_series(self, series: Deque[float], ema_val: float | None, current: float) -> float:
        if self.mode == "current" or len(series) < 2:
            return current
        if self.mode == "stale":
            idx = max(0, len(series) - 1 - self.stale_lag)
            return series[idx]
        if self.mode == "noisy":
            noise = self._rng.gauss(0.0, max(1e-6, abs(current) * self.noise_std_frac))
            return max(0.0, current + noise)

        # Lookahead: damped linear trend + EMA anchor over horizon
        vals = list(series)
        recent_slope = (vals[-1] - vals[max(0, len(vals) - 4)]) / max(1, min(3, len(vals) - 1))
        anchor = ema_val if ema_val is not None else vals[-1]
        # Damped lookahead (0.5 * horizon_steps effective drift)
        pred = 0.65 * vals[-1] + 0.35 * anchor + self.trend_beta * recent_slope * min(3, self.horizon_steps)
        return max(0.0, pred)

    def predict_site(self, site_id: str, state: SiteState, now_s: float) -> Dict[str, float]:
        """Predict near-future carbon intensity, water intensity, queue wait, price, and risk."""
        current_q = state.estimated_queue_wait_s(now_s)
        current_p = state.spec.base_price_per_kwh * state.price_multiplier
        h = self.histories.get(site_id)
        if h is None:
            return {
                "carbon_intensity": state.carbon_intensity,
                "water_intensity": state.water_intensity,
                "queue_wait_s": current_q,
                "price": current_p,
                "risk": 0.0 if state.available else 1.0,
            }

        pred_ci = self._forecast_series(h.carbon, h.carbon_ema, state.carbon_intensity)
        pred_wi = self._forecast_series(h.water, h.water_ema, state.water_intensity)
        pred_q = self._forecast_series(h.queue_wait, h.queue_ema, current_q)
        # Blend instantaneous queue with predicted queue trend
        if self.mode == "lookahead":
            pred_q = 0.60 * current_q + 0.40 * pred_q

        # Risk rises with utilization saturation and recent outage history
        free_ratio = state.free_slots(now_s) / max(1, state.spec.gpu_capacity)
        congestion_risk = max(0.0, 1.0 - free_ratio)
        risk = min(1.0, 0.7 * congestion_risk + 0.3 * h.outage_ema)

        return {
            "carbon_intensity": pred_ci,
            "water_intensity": pred_wi,
            "queue_wait_s": pred_q,
            "price": current_p,
            "risk": risk,
        }

    def record_feedback(
        self,
        pred_carbon: float,
        actual_carbon: float,
        pred_water: float,
        actual_water: float,
        pred_queue_s: float,
        actual_queue_s: float,
    ) -> None:
        self.n_feedback += 1
        self.abs_err_carbon += abs(actual_carbon - pred_carbon)
        self.abs_err_water += abs(actual_water - pred_water)
        self.abs_err_queue += abs(actual_queue_s - pred_queue_s)

    def mae_summary(self) -> Dict[str, float]:
        if self.n_feedback == 0:
            return {"mae_carbon": 0.0, "mae_water": 0.0, "mae_queue_s": 0.0}
        return {
            "mae_carbon": self.abs_err_carbon / self.n_feedback,
            "mae_water": self.abs_err_water / self.n_feedback,
            "mae_queue_s": self.abs_err_queue / self.n_feedback,
        }
