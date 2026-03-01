"""FastAPI application for Monte Carlo pricing."""

import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .schemas import (
    PricingRequest, PricingResponse,
    BatchPricingRequest, BatchPricingResponse,
    HealthResponse,
    PayoffTypeSchema, ModelType
)

# Import pricing library
try:
    from mc_pricer import (
        MarketData, HestonParams, JumpParams, SimulationConfig,
        Backend, DiscretizationScheme, VarianceReduction, PayoffType,
        BarrierParams, SABRParams, RoughHestonParams,
        price_option, price_with_greeks, price_american_option_lsm,
        price_sabr_option, price_rough_heston_option,
        CUPY_AVAILABLE, SCIPY_AVAILABLE
    )
    MC_PRICER_AVAILABLE = True
except ImportError as e:
    MC_PRICER_AVAILABLE = False
    CUPY_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Check CUDA
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except:
    CUDA_AVAILABLE = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    print("Monte Carlo Pricing API starting...")
    if not MC_PRICER_AVAILABLE:
        print(f"WARNING: mc_pricer not available")
    yield
    print("Monte Carlo Pricing API shutting down...")


app = FastAPI(
    title="Monte Carlo Option Pricing API",
    description="GPU-accelerated Monte Carlo option pricing with multiple models",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def convert_payoff_type(payoff: PayoffTypeSchema) -> PayoffType:
    """Convert API schema to internal PayoffType."""
    mapping = {
        PayoffTypeSchema.EUROPEAN_CALL: PayoffType.EUROPEAN_CALL,
        PayoffTypeSchema.EUROPEAN_PUT: PayoffType.EUROPEAN_PUT,
        PayoffTypeSchema.ASIAN_CALL: PayoffType.ASIAN_CALL,
        PayoffTypeSchema.ASIAN_PUT: PayoffType.ASIAN_PUT,
        PayoffTypeSchema.ASIAN_GEOM_CALL: PayoffType.ASIAN_GEOM_CALL,
        PayoffTypeSchema.ASIAN_GEOM_PUT: PayoffType.ASIAN_GEOM_PUT,
        PayoffTypeSchema.BARRIER_UP_OUT_CALL: PayoffType.BARRIER_UP_OUT_CALL,
        PayoffTypeSchema.BARRIER_UP_IN_CALL: PayoffType.BARRIER_UP_IN_CALL,
        PayoffTypeSchema.BARRIER_DOWN_OUT_CALL: PayoffType.BARRIER_DOWN_OUT_CALL,
        PayoffTypeSchema.BARRIER_DOWN_IN_CALL: PayoffType.BARRIER_DOWN_IN_CALL,
        PayoffTypeSchema.BARRIER_UP_OUT_PUT: PayoffType.BARRIER_UP_OUT_PUT,
        PayoffTypeSchema.BARRIER_UP_IN_PUT: PayoffType.BARRIER_UP_IN_PUT,
        PayoffTypeSchema.BARRIER_DOWN_OUT_PUT: PayoffType.BARRIER_DOWN_OUT_PUT,
        PayoffTypeSchema.BARRIER_DOWN_IN_PUT: PayoffType.BARRIER_DOWN_IN_PUT,
        PayoffTypeSchema.AMERICAN_CALL: PayoffType.AMERICAN_CALL,
        PayoffTypeSchema.AMERICAN_PUT: PayoffType.AMERICAN_PUT,
    }
    return mapping.get(payoff)


def _is_barrier_payoff(payoff_type: PayoffType) -> bool:
    return payoff_type.value.startswith("barrier_")


def _build_simulation_config(request: PricingRequest) -> SimulationConfig:
    """Build internal simulation config from request config."""
    if not request.config:
        return SimulationConfig()

    if request.config.use_antithetic and request.config.use_control_variate:
        vr = VarianceReduction.ANTITHETIC_CV
    elif request.config.use_antithetic:
        vr = VarianceReduction.ANTITHETIC
    elif request.config.use_control_variate:
        vr = VarianceReduction.CONTROL_VARIATE
    else:
        vr = VarianceReduction.NONE

    return SimulationConfig(
        num_paths=request.config.num_paths,
        num_steps=request.config.num_steps,
        backend=Backend.NUMPY,
        scheme=DiscretizationScheme.QE,
        variance_reduction=vr,
        seed=request.config.seed,
        use_sobol=request.config.use_sobol
    )


def _price_request(request: PricingRequest):
    """Route a pricing request to the appropriate model implementation."""
    market = MarketData(S0=request.spot, r=request.rate, q=request.dividend)
    payoff_type = convert_payoff_type(request.payoff_type)
    if payoff_type is None:
        raise ValueError(f"Unknown payoff type: {request.payoff_type}")

    config = _build_simulation_config(request)

    if request.model == ModelType.SABR:
        if request.sabr is None:
            raise ValueError("sabr parameters are required for model='sabr'")
        if payoff_type not in (PayoffType.EUROPEAN_CALL, PayoffType.EUROPEAN_PUT):
            raise ValueError("SABR API currently supports only european_call/european_put")

        sabr = SABRParams(
            alpha=request.sabr.alpha,
            beta=request.sabr.beta,
            rho=request.sabr.rho,
            nu=request.sabr.nu
        )
        return price_sabr_option(
            F0=request.spot,
            K=request.strike,
            T=request.time_to_maturity,
            r=request.rate,
            sabr=sabr,
            is_call=payoff_type == PayoffType.EUROPEAN_CALL,
            config=config
        )

    if request.model == ModelType.ROUGH_HESTON:
        if request.rough_heston is None:
            raise ValueError("rough_heston parameters are required for model='rough_heston'")
        if _is_barrier_payoff(payoff_type):
            raise ValueError("Barrier payoffs are not supported with rough_heston in this API")

        rough = RoughHestonParams(
            v0=request.rough_heston.v0,
            theta=request.rough_heston.theta,
            lambda_=request.rough_heston.lambda_,
            nu=request.rough_heston.nu,
            rho=request.rough_heston.rho,
            H=request.rough_heston.H
        )
        return price_rough_heston_option(
            market=market,
            K=request.strike,
            T=request.time_to_maturity,
            params=rough,
            payoff_type=payoff_type,
            config=config
        )

    if request.model == ModelType.LOCAL_VOL:
        raise ValueError(
            "model='local_vol' requires a local volatility surface payload, "
            "which is not exposed by this API yet"
        )

    heston = None
    jump = None
    sigma = request.volatility
    barrier_params = None

    if request.model == ModelType.GBM:
        if sigma is None:
            raise ValueError("volatility is required for model='gbm'")

    if request.model in (ModelType.HESTON, ModelType.BATES):
        if request.heston is None:
            raise ValueError(f"heston parameters are required for model='{request.model.value}'")
        heston = HestonParams(
            v0=request.heston.v0,
            kappa=request.heston.kappa,
            theta=request.heston.theta,
            xi=request.heston.xi,
            rho=request.heston.rho
        )
        sigma = None

    if request.model == ModelType.BATES:
        if request.jump is None:
            raise ValueError("jump parameters are required for model='bates'")
        jump = JumpParams(
            lambda_j=request.jump.lambda_j,
            mu_j=request.jump.mu_j,
            sigma_j=request.jump.sigma_j
        )

    if request.barrier:
        barrier_params = BarrierParams(
            barrier=request.barrier.barrier,
            rebate=request.barrier.rebate
        )
    elif _is_barrier_payoff(payoff_type):
        raise ValueError(
            f"barrier parameters are required for payoff_type='{payoff_type.value}'"
        )

    if payoff_type in (PayoffType.AMERICAN_CALL, PayoffType.AMERICAN_PUT):
        if request.compute_greeks:
            raise ValueError("compute_greeks is not supported for American options")
        return price_american_option_lsm(
            market=market,
            K=request.strike,
            T=request.time_to_maturity,
            is_call=payoff_type == PayoffType.AMERICAN_CALL,
            heston=heston,
            jump=jump,
            config=config,
            sigma=sigma
        )

    if request.compute_greeks:
        return price_with_greeks(
            market=market, K=request.strike, T=request.time_to_maturity,
            payoff_type=payoff_type, heston=heston, jump=jump,
            barrier=barrier_params, config=config, sigma=sigma
        )
    return price_option(
        market=market, K=request.strike, T=request.time_to_maturity,
        payoff_type=payoff_type, heston=heston, jump=jump,
        barrier=barrier_params, config=config, sigma=sigma
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if MC_PRICER_AVAILABLE else "degraded",
        version="1.0.0",
        cuda_available=CUDA_AVAILABLE,
        cupy_available=CUPY_AVAILABLE if MC_PRICER_AVAILABLE else False
    )


@app.post("/price", response_model=PricingResponse)
async def price_option_endpoint(request: PricingRequest):
    """Price a single option using Monte Carlo simulation."""
    if not MC_PRICER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Pricing library not available")

    start_time = time.time()

    try:
        result = _price_request(request)

        return PricingResponse(
            price=result.price,
            std_error=result.std_error,
            paths_used=result.paths_used,
            elapsed_time=result.elapsed_time or (time.time() - start_time),
            greeks=result.greeks,
            control_variate_beta=result.control_variate_beta,
            variance_reduction=result.variance_reduction,
            model=request.model.value,
            error=None
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/price/batch", response_model=BatchPricingResponse)
async def batch_price_options(request: BatchPricingRequest):
    """Price multiple options in batch."""
    if not MC_PRICER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Pricing library not available")

    start_time = time.time()
    results = []

    for opt_request in request.options:
        try:
            result = await price_option_endpoint(opt_request)
            results.append(result)
        except HTTPException as e:
            results.append(PricingResponse(
                price=0.0, std_error=0.0, paths_used=0, elapsed_time=0.0,
                variance_reduction="error", model=opt_request.model.value,
                greeks=None,
                error=str(e.detail)
            ))

    return BatchPricingResponse(
        results=results,
        total_time=time.time() - start_time,
        num_options=len(request.options)
    )


@app.get("/models")
async def list_models():
    """List available pricing models."""
    return {
        "models": [m.value for m in ModelType],
        "payoff_types": [p.value for p in PayoffTypeSchema]
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
