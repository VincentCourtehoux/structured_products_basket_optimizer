"""
Product configuration schemas, presets, and validation utilities.
"""

from typing import Any, Union
from src.products.autocall import AutocallParams, AutocallPricer
from src.products.capital_guaranteed import CapitalGuaranteedParams, CapitalGuaranteedPricer
from src.products.delta_one import DeltaOneParams, DeltaOnePricer


def get_default_preset(
    preset_key: str, 
    index: str | None = None, 
    available_indices: list[str] | None = None
) -> dict[str, Any]:
    """
    Retrieves preset product configuration dictionaries.
    Percentages are stored as decimal fractions (e.g., 0.025 for 2.5%).
    
    Args:
        preset_key: The identifier for the preset (e.g., 'p1', 'p2').
        available_indices: List of available indices to set as default underlyings.
        
    Returns:
        A dictionary containing the requested product configuration.
    """
    default_indices = available_indices if available_indices else ["SX5E", "SPX"]
    index = index if index is not None else default_indices[0]

    presets = {
        "p1": {
            "name": "P1 - Step-down Autocall",
            "type": "autocall",
            "index_names": [index],
            "basket_type": "single",
            "maturity": 8.0,
            "obs_freq": "quarterly",
            "autocall_barrier_initial": 1.00,
            "barrier_step_down": 0.005,
            "barrier_step_freq": "quarterly",
            "first_call_year": 1.0,
            "barrier_step_start_year": 1.25,
            "coupon_rate": 0.025,
            "coupon_barrier": None,
            "memory_coupon": True,
            "capital_barrier": 0.60,
            "capital_barrier_type": "european",
            "notional": 1.0,
        },
        "p2": {
            "name": "P2 - ATM Autocall",
            "type": "autocall",
            "index_names": [index],
            "basket_type": "single",
            "maturity": 8.0,
            "obs_freq": "annual",
            "autocall_barrier_initial": 1.00,
            "barrier_step_down": 0.0,
            "barrier_step_freq": "annual",
            "first_call_year": 1.0,
            "barrier_step_start_year": 1.0,
            "coupon_rate": 0.11,
            "coupon_barrier": None,
            "memory_coupon": True,
            "capital_barrier": 0.50,
            "capital_barrier_type": "european",
            "notional": 1.0,
        },
        "p3": {
            "name": "P3 - Capital Guaranteed Lookback",
            "type": "capital_guaranteed",
            "index_name": index,
            "maturity": 2.0,
            "obs_freq": "semi-annual",
            "floor": 0.04,
            "participation": 1.15,
            "lookback_type": "avg",
            "notional": 1.0,
        },
        "p4": {
            "name": "P4 - Delta 1 SCORED 24M",
            "type": "delta_one",
            "index_name": index,
            "maturity": 2.0,
            "fixed_coupon": 0.06,
            "notional": 1.0,
        },
    }
    
    return presets.get(preset_key.lower(), presets["p1"])


def create_params_from_config(cfg: dict[str, Any]) -> Union[AutocallParams, CapitalGuaranteedParams, DeltaOneParams]:
    """
    Instantiates the appropriate Params dataclass from a configuration dictionary.
    
    Args:
        cfg: The product configuration dictionary.
        
    Returns:
        An instance of the corresponding product Params dataclass.
    """
    ptype = str(cfg.get("type", "autocall")).lower()
    
    if ptype == "autocall":
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
        
    elif ptype in ("capital_guaranteed", "capital-guaranteed"):
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
        
    elif ptype in ("delta_one", "delta-one"):
        return DeltaOneParams(
            name=str(cfg.get("name", "Delta One")),
            index_name=str(cfg.get("index_name", "")),
            maturity=float(cfg.get("maturity", 2.0)),
            fixed_coupon=float(cfg.get("fixed_coupon", 0.06)),
            notional=float(cfg.get("notional", 1.0)),
        )
        
    raise ValueError(f"Unknown product type: {ptype}")


def create_pricer_from_config(cfg: dict[str, Any]) -> Union[AutocallPricer, CapitalGuaranteedPricer, DeltaOnePricer]:
    """
    Instantiates a product pricer engine from a configuration dictionary.
    
    Args:
        cfg: The product configuration dictionary.
        
    Returns:
        The instantiated pricer object corresponding to the product type.
    """
    params = create_params_from_config(cfg)
    
    if isinstance(params, AutocallParams):
        return AutocallPricer(params)
    elif isinstance(params, CapitalGuaranteedParams):
        return CapitalGuaranteedPricer(params)
    elif isinstance(params, DeltaOneParams):
        return DeltaOnePricer(params)
        
    raise ValueError(f"Unsupported product class mapping: {type(params)}")


def validate_product_config(cfg: dict[str, Any], available_indices: list[str] | None = None) -> tuple[bool, str]:
    """
    Validates that a product configuration has the required fields, logical parameters, 
    and valid underlying indices.
    
    Args:
        cfg: The product configuration dictionary.
        available_indices: List of valid market indices to cross-check against.
        
    Returns:
        A tuple (is_valid, status_message).
    """
    ptype = str(cfg.get("type", "")).lower()
    name = cfg.get("name", "Unnamed")

    if not cfg.get("name"):
        return False, "The product must have a valid name."

    # Common checks across all product types
    if float(cfg.get("maturity", 0.0)) <= 0:
        return False, f"[{name}] Maturity must be strictly positive."

    # Index extraction based on product type
    if ptype == "autocall":
        idx_names = cfg.get("index_names", [])
        
        if not idx_names:
            return False, f"[{name}] At least one underlying index must be selected for Autocall products."
            
        if cfg.get("basket_type") == "single" and len(idx_names) != 1:
            return False, f"[{name}] The 'single' basket type requires exactly 1 underlying index."
            
        if float(cfg.get("capital_barrier", 0.0)) <= 0:
            return False, f"[{name}] The capital barrier must be strictly positive."
            
        indices_to_check = idx_names
        
    elif ptype in ("capital_guaranteed", "capital-guaranteed", "delta_one", "delta-one"):
        idx_name = cfg.get("index_name", "")
        
        if not idx_name:
            return False, f"[{name}] An underlying index must be selected."
            
        indices_to_check = [idx_name]
        
    else:
        return False, f"Unrecognized product type: {ptype}"

    # Global index validation against available simulation data
    if available_indices:
        for idx in indices_to_check:
            if idx not in available_indices:
                return False, f"[{name}] The index '{idx}' is not available in the current simulation environment."

    return True, "Valid"