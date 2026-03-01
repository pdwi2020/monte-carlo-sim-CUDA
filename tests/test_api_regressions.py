"""API regression tests for model routing and response contracts."""

import pytest
import os
import sys

pytest.importorskip("fastapi")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException

from api.main import batch_price_options, convert_payoff_type, price_option_endpoint
from api.schemas import (
    BatchPricingRequest,
    ModelType,
    PayoffTypeSchema,
    PricingRequest,
)
from mc_pricer import PayoffType


def test_convert_payoff_type_includes_american():
    """American payoff enums should map to internal PayoffType."""
    assert convert_payoff_type(PayoffTypeSchema.AMERICAN_CALL) == PayoffType.AMERICAN_CALL
    assert convert_payoff_type(PayoffTypeSchema.AMERICAN_PUT) == PayoffType.AMERICAN_PUT


@pytest.mark.anyio
async def test_local_vol_model_is_explicitly_rejected():
    """Unsupported model routes should fail explicitly instead of silently falling back."""
    request = PricingRequest(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        time_to_maturity=1.0,
        volatility=0.2,
        payoff_type=PayoffTypeSchema.EUROPEAN_CALL,
        model=ModelType.LOCAL_VOL,
    )

    with pytest.raises(HTTPException) as exc:
        await price_option_endpoint(request)
    assert exc.value.status_code == 400
    assert "local volatility surface" in str(exc.value.detail).lower()


@pytest.mark.anyio
async def test_rough_heston_route_requires_params():
    """rough_heston requests must include rough_heston parameter payload."""
    request = PricingRequest(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        time_to_maturity=1.0,
        payoff_type=PayoffTypeSchema.EUROPEAN_CALL,
        model=ModelType.ROUGH_HESTON,
    )

    with pytest.raises(HTTPException) as exc:
        await price_option_endpoint(request)
    assert exc.value.status_code == 400
    assert "rough_heston parameters are required" in str(exc.value.detail)


@pytest.mark.anyio
async def test_batch_error_item_matches_response_schema():
    """Batch endpoint should return schema-valid error items when one option fails."""
    ok_request = PricingRequest(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        time_to_maturity=1.0,
        volatility=0.2,
        payoff_type=PayoffTypeSchema.EUROPEAN_CALL,
        model=ModelType.GBM,
        config={"num_paths": 2000, "num_steps": 30, "seed": 11},
    )
    bad_request = PricingRequest(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        time_to_maturity=1.0,
        volatility=0.2,
        payoff_type=PayoffTypeSchema.BARRIER_UP_OUT_CALL,
        model=ModelType.GBM,
        config={"num_paths": 2000, "num_steps": 30, "seed": 11},
    )

    response = await batch_price_options(BatchPricingRequest(options=[ok_request, bad_request]))

    assert len(response.results) == 2
    assert response.results[0].error is None
    assert response.results[1].error is not None
    assert response.results[1].greeks is None
