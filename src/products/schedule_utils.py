"""
Shared utilities for structured product schedules and index resolution.
"""

import numpy as np

FREQ_TO_N_PER_YEAR: dict[str, int] = {
    "monthly": 12,
    "quarterly": 4,
    "semi-annual": 2,
    "annual": 1,
}


def n_per_year(obs_freq: str, default_freq: str = "quarterly") -> int:
    """Resolves an observation frequency string to periods-per-year, with a fallback."""
    return FREQ_TO_N_PER_YEAR.get(str(obs_freq).lower(), FREQ_TO_N_PER_YEAR[default_freq])


def build_observation_schedule(
    maturity: float, obs_freq: str, default_freq: str = "quarterly"
) -> list[float]:
    """
    Builds a regular observation date schedule in year fractions,
    e.g. maturity=2.0, obs_freq="semi-annual" -> [0.5, 1.0, 1.5, 2.0].

    Used by every Params.__post_init__ that doesn't receive explicit
    observation_dates. Mathematically identical to the previous
    per-class implementations.
    """
    n_py = n_per_year(obs_freq, default_freq=default_freq)
    dt_step = 1.0 / n_py
    n_obs = int(round(maturity * n_py))
    return [round((i + 1) * dt_step, 6) for i in range(n_obs)]


def map_dates_to_steps(obs_dates: list[float], dt: float, n_path_steps: int) -> list[int]:
    """
    Maps year-fraction observation dates onto simulation step indices,
    clamped to the last available step.
    """
    return [min(int(round(t / dt)), n_path_steps - 1) for t in obs_dates]


def resolve_index(name: str, index_names: list[str]) -> int:
    """Resolves a single underlying name to its column index in the paths array."""
    if name not in index_names:
        raise ValueError(f"Underlying index '{name}' not found in simulation paths.")
    return index_names.index(name)


def resolve_indices(names: list[str], index_names: list[str]) -> list[int]:
    """Resolves multiple underlying names to their column indices in the paths array."""
    matched_idx = []
    for name in names:
        matched_idx.append(resolve_index(name, index_names))
    return matched_idx


def relative_paths(paths: np.ndarray, matched_idx: list[int]) -> np.ndarray:
    """
    Extracts the sub-paths for matched_idx and normalizes by S(0).
    Shape in: (n_sim, n_path_steps, n_assets) -> shape out: (n_sim, n_path_steps, len(matched_idx))
    """
    s0 = paths[:, 0:1, matched_idx]
    return paths[:, :, matched_idx] / s0