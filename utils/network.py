"""Projection of the SHAP attributions on the 18 network templates.

Shared by ``export_R_data.py``, ``visualization/base.py`` and the tools, so that every
figure uses the same normalisation.
"""
import numpy as np
import pandas as pd

NORMALIZATIONS = ("block", "timepoint")


def load_templates(path):
    """Network templates of ``external/schaeffer_subcortical_smith_templates.csv``.

    Returns (network names, ROI names, weights) with weights of shape (18, 481), each
    template L2-normalised over the ROIs.
    """
    network = pd.read_csv(path)
    weights = network.iloc[:, 1:].to_numpy(dtype=float)
    weights = weights / np.sqrt(np.sum(weights ** 2, axis=1, keepdims=True))
    return list(network.iloc[:, 0]), list(network.columns[1:]), weights


def normalize_shap(shap_values, normalization="block"):
    """L2-normalise SHAP maps of shape (..., T, R, C) (T time points, R ROIs, C conditions).

    block      each block's map (T x R) of each condition is divided by its norm over time and
               ROIs. The norm of the map at time t is then the share of the block's attribution
               carried by that second, so the projection keeps how much each second contributes
               to the prediction (the explained output is the last time point of the block).
    timepoint  each map (R) is divided by its own norm over the ROIs (previous version): every
               second gets a unit map, and the projection is a cosine similarity whose time
               course only reflects changes of the spatial pattern, not of the amount.
    """
    if normalization == "block":
        axes = (-3, -2)
    elif normalization == "timepoint":
        axes = (-2,)
    else:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}, not {normalization!r}")
    norm = np.sqrt(np.sum(shap_values ** 2, axis=axes, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 0, shap_values / norm, np.nan)


def network_projection(shap_values, weights, normalization="block"):
    """(..., T, 481, C) -> (..., T, 18, C): normalisation (see ``normalize_shap``), then dot
    product of each map with each L2-normalised template along the ROI axis."""
    return np.einsum("...trc,kr->...tkc", normalize_shap(shap_values, normalization), weights)
