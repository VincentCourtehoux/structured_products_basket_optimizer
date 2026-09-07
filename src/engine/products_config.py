"""
Product configuration schemas, presets, and validation utilities.
"""

from dataclasses import dataclass
from typing import Any, Union
from src.products.autocall import AutocallParams, AutocallPricer
from src.products.capital_guaranteed import CapitalGuaranteedParams, CapitalGuaranteedPricer
from src.products.delta_one import DeltaOneParams, DeltaOnePricer

ProductParams = Union[AutocallParams, CapitalGuaranteedParams, DeltaOneParams]
ProductPricer = Union[AutocallPricer, CapitalGuaranteedPricer, DeltaOnePricer]


def _normalize_type(cfg: dict[str, Any]) -> str:
    return str(cfg.get("type", "autocall")).lower().replace("-", "_")


@dataclass(frozen=True)
class ProductSpec:
    """
    One declarative entry per product type. Adding a new product means
    adding one ProductSpec to REGISTRY (plus a small builder function).
    """
    params_cls: type
    pricer_cls: type
    build_params: Any  # Callable[[dict[str, Any]], ProductParams]
    validate_specific: Any  # Callable[[dict[str, Any], str], tuple[bool, str] | None]
    index_field: str  # "index_names" (list) or "index_name" (single str)


def _build_autocall_params(cfg: dict[str, Any]) -> AutocallParams:
    raw_cb = cfg.get("coupon_barrier")
    parsed_cb = float(raw_cb) if raw_cb is not None and str(raw_cb).strip() else None

    return AutocallParams(
        name=str(cfg.get("name", "Autocall")),
        index_names=list(cfg.get("index_names", [])),
        basket_type=str(cfg.get("basket_type", "worst-of")),
        maturity=float(cfg.get("maturity", 8.0)),
        obs_freq=str(cfg.get("obs_freq", "quarterly")),
        autocall_barrier_initial=float(cfg.get("autocall_barrier_initial", 1.0)),
        barrier_step_down=float(cfg.get("barrier_step_down", 0.0)),
        barrier_step_freq=str(cfg.get("barrier_step_freq", "quarterly")),
        first_call_year=float(cfg.get("first_call_year", 1.0)),
        barrier_step_start_year=float(cfg.get("barrier_step_start_year", 1.0)),
        coupon_rate=float(cfg.get("coupon_rate", 0.025)),
        coupon_barrier=parsed_cb,
        memory_coupon=bool(cfg.get("memory_coupon", True)),
        capital_barrier=float(cfg.get("capital_barrier", 0.60)),
        capital_barrier_type=str(cfg.get("capital_barrier_type", "european")),
        notional=float(cfg.get("notional", 1.0)),
    )


def _validate_autocall_specific(cfg: dict[str, Any], name: str) -> tuple[bool, str] | None:
    idx_names = cfg.get("index_names", [])

    if not idx_names:
        return False, f"[{name}] At least one underlying index must be selected for Autocall products."

    if cfg.get("basket_type") == "single" and len(idx_names) != 1:
        return False, f"[{name}] The 'single' basket type requires exactly 1 underlying index."

    if float(cfg.get("capital_barrier", 0.0)) <= 0:
        return False, f"[{name}] The capital barrier must be strictly positive."

    return None


def _build_capital_guaranteed_params(cfg: dict[str, Any]) -> CapitalGuaranteedParams:
    return CapitalGuaranteedParams(
        name=str(cfg.get("name", "Capital Guaranteed")),
        index_name=str(cfg.get("index_name", "")),
        maturity=float(cfg.get("maturity", 2.0)),
        obs_freq=str(cfg.get("obs_freq", "semi-annual")),
        floor=float(cfg.get("floor", 0.04)),
        participation=float(cfg.get("participation", 1.15)),
        lookback_type=str(cfg.get("lookback_type", "max")),
        notional=float(cfg.get("notional", 1.0)),
    )


def _build_delta_one_params(cfg: dict[str, Any]) -> DeltaOneParams:
    return DeltaOneParams(
        name=str(cfg.get("name", "Delta One")),
        index_name=str(cfg.get("index_name", "")),
        maturity=float(cfg.get("maturity", 2.0)),
        fixed_coupon=float(cfg.get("fixed_coupon", 0.06)),
        notional=float(cfg.get("notional", 1.0)),
    )


def _no_specific_validation(cfg: dict[str, Any], name: str) -> tuple[bool, str] | None:
    return None


REGISTRY: dict[str, ProductSpec] = {
    "autocall": ProductSpec(
        params_cls=AutocallParams,
        pricer_cls=AutocallPricer,
        build_params=_build_autocall_params,
        validate_specific=_validate_autocall_specific,
        index_field="index_names",
    ),
    "capital_guaranteed": ProductSpec(
        params_cls=CapitalGuaranteedParams,
        pricer_cls=CapitalGuaranteedPricer,
        build_params=_build_capital_guaranteed_params,
        validate_specific=_no_specific_validation,
        index_field="index_name",
    ),
    "delta_one": ProductSpec(
        params_cls=DeltaOneParams,
        pricer_cls=DeltaOnePricer,
        build_params=_build_delta_one_params,
        validate_specific=_no_specific_validation,
        index_field="index_name",
    ),
}


class ProductFactory:
    """
    Factory class for generating structured product parameters, pricers, and presets.
    Handles validation against available market indices.
    """

    def __init__(self, available_indices: list[str] | None = None):
        self.available_indices = available_indices or ["SX5E", "SPX"]

    def get_preset(self, preset_key: str, index: str | None = None) -> dict[str, Any]:
        """
        Retrieves preset product configuration dictionaries.
        """
        target_index = index if index is not None else self.available_indices[0]

        presets = {
            "p1": {
                "name": "P1 - Step-down Autocall",
                "type": "autocall",
                "index_names": [target_index],
                "basket_type": "single",
                "maturity": 8.0,
                "obs_freq": "quarterly",
                "autocall_barrier_initial": 1.00,
                "barrier_step_down": 0.01,
                "barrier_step_freq": "quarterly",
                "first_call_year": 1.0,
                "barrier_step_start_year": 1.0,
                "coupon_rate": 0.02,
                "coupon_barrier": None,
                "memory_coupon": True,
                "capital_barrier": 0.50,
                "capital_barrier_type": "european",
                "notional": 1.0,
            },
            "p2": {
                "name": "P2 - Phoenix Autocall",
                "type": "autocall",
                "index_names": [target_index],
                "basket_type": "single",
                "maturity": 5.0,
                "obs_freq": "semi-annual",
                "autocall_barrier_initial": 1.00,
                "barrier_step_down": 0.0,
                "barrier_step_freq": "semi-annual",
                "first_call_year": 1.0,
                "barrier_step_start_year": 1.0,
                "coupon_rate": 0.04,
                "coupon_barrier": 0.75,
                "memory_coupon": True,
                "capital_barrier": 0.60,
                "capital_barrier_type": "european",
                "notional": 1.0,
            },
            "p3": {
                "name": "P3 - Capital Guaranteed Asian",
                "type": "capital_guaranteed",
                "index_name": target_index,
                "maturity": 5.0,
                "obs_freq": "monthly",
                "floor": 0.00,
                "participation": 0.70,
                "lookback_type": "avg",
                "notional": 1.0,
            },
            "p4": {
                "name": "P4 - Delta 1 Tracker 36M",
                "type": "delta_one",
                "index_name": target_index,
                "maturity": 3.0,
                "fixed_coupon": 0.035,
                "notional": 1.0,
            },
        }

        return presets.get(preset_key.lower(), presets["p1"])

    def create_params(self, cfg: dict[str, Any]) -> ProductParams:
        """
        Instantiates the appropriate Params dataclass from a configuration dictionary.
        """
        ptype = _normalize_type(cfg)
        spec = REGISTRY.get(ptype)

        if spec is None:
            raise ValueError(f"Unknown product type: {ptype}")

        return spec.build_params(cfg)

    def create_pricer(self, cfg: dict[str, Any]) -> ProductPricer:
        """
        Instantiates a product pricer engine from a configuration dictionary.
        """
        ptype = _normalize_type(cfg)
        spec = REGISTRY.get(ptype)

        if spec is None:
            raise ValueError(f"Unknown product type: {ptype}")

        params = spec.build_params(cfg)
        return spec.pricer_cls(params)

    def validate_config(self, cfg: dict[str, Any]) -> tuple[bool, str]:
        """
        Validates product configuration logistics and index availability.
        """
        ptype = _normalize_type(cfg)
        name = cfg.get("name", "Unnamed")
        spec = REGISTRY.get(ptype)

        if spec is None:
            return False, f"Unrecognized product type: {ptype}"

        if not cfg.get("name"):
            return False, "The product must have a valid name."

        if float(cfg.get("maturity", 0.0)) <= 0:
            return False, f"[{name}] Maturity must be strictly positive."

        # Product-specific rules (basket size, positive barrier, etc.)
        specific_result = spec.validate_specific(cfg, name)
        if specific_result is not None:
            return specific_result

        # Index availability, generic across single- and multi-index products
        if spec.index_field == "index_names":
            indices_to_check = cfg.get("index_names", [])
        else:
            idx_name = cfg.get("index_name", "")
            if not idx_name:
                return False, f"[{name}] An underlying index must be selected."
            indices_to_check = [idx_name]

        for idx in indices_to_check:
            if idx not in self.available_indices:
                return False, f"[{name}] The index '{idx}' is not available in the current simulation environment."

        return True, "Valid"