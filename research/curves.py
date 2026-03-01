"""Rate/dividend curve helpers for forward and discounting consistency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass
class CurvePoint:
    """Simple continuous-compounded zero/yield curve point."""

    maturity: float
    rate: float


def _sorted_points(points: Iterable[CurvePoint]) -> List[CurvePoint]:
    pts = sorted([CurvePoint(float(p.maturity), float(p.rate)) for p in points], key=lambda x: x.maturity)
    if not pts:
        raise ValueError("Curve points cannot be empty")
    return pts


def interpolate_rate(points: Iterable[CurvePoint], maturity: float) -> float:
    """Piecewise-linear interpolation with endpoint extrapolation."""

    t = float(maturity)
    pts = _sorted_points(points)
    if t <= pts[0].maturity:
        return pts[0].rate
    if t >= pts[-1].maturity:
        return pts[-1].rate

    mats = np.asarray([p.maturity for p in pts], dtype=float)
    rates = np.asarray([p.rate for p in pts], dtype=float)
    return float(np.interp(t, mats, rates))


def discount_factor(rate_curve: Iterable[CurvePoint], maturity: float) -> float:
    """Discount factor under continuous compounding."""

    r = interpolate_rate(rate_curve, maturity)
    return float(np.exp(-r * max(float(maturity), 0.0)))


def forward_price(
    spot: float,
    maturity: float,
    *,
    rate_curve: Iterable[CurvePoint],
    dividend_curve: Iterable[CurvePoint],
) -> float:
    """Forward price from rate/dividend curves (continuous compounding)."""

    t = max(float(maturity), 0.0)
    r = interpolate_rate(rate_curve, t)
    q = interpolate_rate(dividend_curve, t)
    return float(spot * np.exp((r - q) * t))


def build_flat_curve(rate: float) -> List[CurvePoint]:
    """Helper to build a flat curve."""

    r = float(rate)
    return [CurvePoint(maturity=0.01, rate=r), CurvePoint(maturity=30.0, rate=r)]

