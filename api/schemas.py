"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List
from enum import Enum


class PayoffTypeSchema(str, Enum):
    """Available payoff types."""
    EUROPEAN_CALL = "european_call"
    EUROPEAN_PUT = "european_put"
    ASIAN_CALL = "asian_call"
    ASIAN_PUT = "asian_put"
    ASIAN_GEOM_CALL = "asian_geom_call"
    ASIAN_GEOM_PUT = "asian_geom_put"
    BARRIER_UP_OUT_CALL = "barrier_up_out_call"
    BARRIER_UP_IN_CALL = "barrier_up_in_call"
    BARRIER_DOWN_OUT_CALL = "barrier_down_out_call"
    BARRIER_DOWN_IN_CALL = "barrier_down_in_call"
    BARRIER_UP_OUT_PUT = "barrier_up_out_put"
    BARRIER_UP_IN_PUT = "barrier_up_in_put"
    BARRIER_DOWN_OUT_PUT = "barrier_down_out_put"
    BARRIER_DOWN_IN_PUT = "barrier_down_in_put"
    AMERICAN_CALL = "american_call"
    AMERICAN_PUT = "american_put"


class ModelType(str, Enum):
    """Available pricing models."""
    GBM = "gbm"
    HESTON = "heston"
    BATES = "bates"
    SABR = "sabr"
    LOCAL_VOL = "local_vol"
    ROUGH_HESTON = "rough_heston"


class HestonParamsSchema(BaseModel):
    """Heston model parameters."""
    v0: float = Field(..., gt=0, description="Initial variance")
    kappa: float = Field(..., gt=0, description="Mean reversion speed")
    theta: float = Field(..., gt=0, description="Long-term variance")
    xi: float = Field(..., gt=0, description="Vol of vol")
    rho: float = Field(..., ge=-1, le=1, description="Correlation")


class JumpParamsSchema(BaseModel):
    """Jump diffusion parameters."""
    lambda_j: float = Field(0.0, ge=0, description="Jump intensity")
    mu_j: float = Field(0.0, description="Mean log jump size")
    sigma_j: float = Field(0.0, ge=0, description="Std of log jump size")


class SABRParamsSchema(BaseModel):
    """SABR model parameters."""
    alpha: float = Field(..., gt=0, description="Vol of vol")
    beta: float = Field(..., ge=0, le=1, description="CEV exponent")
    rho: float = Field(..., ge=-1, le=1, description="Correlation")
    nu: float = Field(..., gt=0, description="Initial vol")


class RoughHestonParamsSchema(BaseModel):
    """Rough Heston model parameters."""
    v0: float = Field(..., ge=0, description="Initial variance")
    theta: float = Field(..., ge=0, description="Long-term variance")
    lambda_: float = Field(..., gt=0, description="Mean reversion speed")
    nu: float = Field(..., gt=0, description="Vol of vol")
    rho: float = Field(..., ge=-1, le=1, description="Correlation")
    H: float = Field(..., gt=0, lt=0.5, description="Hurst parameter")


class BarrierParamsSchema(BaseModel):
    """Barrier option parameters."""
    barrier: float = Field(..., gt=0, description="Barrier level")
    rebate: float = Field(0.0, ge=0, description="Rebate amount")


class SimulationConfigSchema(BaseModel):
    """Monte Carlo simulation configuration."""
    num_paths: int = Field(100000, gt=0, le=10000000, description="Number of paths")
    num_steps: int = Field(252, gt=0, le=10000, description="Time steps")
    use_antithetic: bool = Field(True, description="Use antithetic variates")
    use_control_variate: bool = Field(True, description="Use control variates")
    use_sobol: bool = Field(False, description="Use Sobol QMC")
    seed: Optional[int] = Field(None, description="Random seed")


class PricingRequest(BaseModel):
    """Request for option pricing."""
    spot: float = Field(..., gt=0, description="Current spot price")
    strike: float = Field(..., gt=0, description="Strike price")
    rate: float = Field(..., description="Risk-free rate (annualized)")
    dividend: float = Field(0.0, ge=0, description="Dividend yield")
    time_to_maturity: float = Field(..., gt=0, le=30, description="Time to expiry (years)")

    payoff_type: PayoffTypeSchema = Field(..., description="Option payoff type")
    model: ModelType = Field(ModelType.GBM, description="Pricing model")

    volatility: Optional[float] = Field(None, gt=0, le=5, description="Volatility for GBM")
    heston: Optional[HestonParamsSchema] = Field(None, description="Heston parameters")
    jump: Optional[JumpParamsSchema] = Field(None, description="Jump parameters")
    sabr: Optional[SABRParamsSchema] = Field(None, description="SABR parameters")
    rough_heston: Optional[RoughHestonParamsSchema] = Field(
        None, description="Rough Heston parameters"
    )
    barrier: Optional[BarrierParamsSchema] = Field(None, description="Barrier parameters")

    config: Optional[SimulationConfigSchema] = Field(None, description="Simulation config")
    compute_greeks: bool = Field(False, description="Compute Greeks")


class PricingResponse(BaseModel):
    """Response from option pricing."""
    price: float = Field(..., description="Option price")
    std_error: float = Field(..., description="Standard error")
    paths_used: int = Field(..., description="Number of paths used")
    elapsed_time: float = Field(..., description="Computation time (seconds)")

    greeks: Optional[Dict[str, float]] = Field(None, description="Greeks if computed")
    control_variate_beta: Optional[float] = Field(None, description="CV beta")
    variance_reduction: str = Field(..., description="Variance reduction used")
    model: str = Field(..., description="Model used")
    error: Optional[str] = Field(None, description="Error message if pricing failed")


class BatchPricingRequest(BaseModel):
    """Request for batch pricing multiple options."""
    options: List[PricingRequest] = Field(..., min_length=1, max_length=100)


class BatchPricingResponse(BaseModel):
    """Response from batch pricing."""
    results: List[PricingResponse]
    total_time: float
    num_options: int


class GreeksRequest(BaseModel):
    """Request for Greeks calculation."""
    spot: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    rate: float
    volatility: float = Field(..., gt=0)
    time_to_maturity: float = Field(..., gt=0)
    payoff_type: PayoffTypeSchema
    config: Optional[SimulationConfigSchema] = None


class GreeksResponse(BaseModel):
    """Response with all Greeks."""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    elapsed_time: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    cuda_available: bool
    cupy_available: bool


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
