"""Projection of the SHAP attributions on the 18 network templates.

Shared by ``export_R_data.py``, ``visualization/base.py`` and the tools, so that every
figure uses the same normalisation.
"""
import numpy as np
import pandas as pd

NORMALIZATIONS = ("block", "block_with_last", "timepoint")

# 1-based second of the explained output (prediction at the last time point of the 25 s window).
# Its own input carries a large attribution by construction and is left out of every analysis,
# so it is also left out of the block norm (see normalize_shap).
LAST_SECOND = 25


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

    block            each block's map of each condition is divided by its norm over ROIs and the
                     seconds analysed (1 to 24: the last second, LAST_SECOND, is left out of the
                     norm as of every analysis). The norm of the map at second t is then the share
                     of the block's attribution carried by that second, so the projection keeps
                     how much each second contributes to the prediction.
    block_with_last  same, with the last second in the norm (exports of November 16, 2026): the
                     last second, which carries the largest attribution by construction, then
                     sets most of the norm, so the other seconds of a block are scaled by how
                     dominant its last second is.
    timepoint        each map (R) is divided by its own norm over the ROIs (the paper's Methods,
                     "L2 regularized ROI importance"): every second gets a unit map, and the
                     projection is a cosine similarity whose time course only reflects changes
                     of the spatial pattern, not of the amount.

    The three differ by a positive factor common to all networks of a block and second: the
    order of the networks at each second (shap_rank in R/Article.Rmd), hence the selection of
    the networks, is the same; only the amounts (curves) differ.
    """
    if normalization == "block":
        kept = shap_values[..., :LAST_SECOND - 1, :, :]
        norm = np.sqrt(np.sum(kept ** 2, axis=(-3, -2), keepdims=True))
    elif normalization == "block_with_last":
        norm = np.sqrt(np.sum(shap_values ** 2, axis=(-3, -2), keepdims=True))
    elif normalization == "timepoint":
        norm = np.sqrt(np.sum(shap_values ** 2, axis=-2, keepdims=True))
    else:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}, not {normalization!r}")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 0, shap_values / norm, np.nan)


def network_projection(shap_values, weights, normalization="block"):
    """(..., T, 481, C) -> (..., T, 18, C): normalisation (see ``normalize_shap``), then dot
    product of each map with each L2-normalised template along the ROI axis."""
    return np.einsum("...trc,kr->...tkc", normalize_shap(shap_values, normalization), weights)
