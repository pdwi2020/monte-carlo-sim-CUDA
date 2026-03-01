"""Tests for rate/dividend curve helpers."""

from research.curves import CurvePoint, build_flat_curve, discount_factor, forward_price, interpolate_rate


def test_curve_interpolation_and_forward():
    rate_curve = [CurvePoint(0.5, 0.02), CurvePoint(2.0, 0.03)]
    div_curve = build_flat_curve(0.01)
    r = interpolate_rate(rate_curve, 1.0)
    assert 0.02 <= r <= 0.03
    df = discount_factor(rate_curve, 1.0)
    assert 0.0 < df < 1.0
    fwd = forward_price(100.0, 1.0, rate_curve=rate_curve, dividend_curve=div_curve)
    assert fwd > 0.0
