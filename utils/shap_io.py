"""Reading the SHAP files of ``main_shap.py`` with a checked axis order.

``shap.DeepExplainer.shap_values`` returns, for a model with 3 outputs:

* shap >= 0.45: one array (blocks, 25 TR, 481 ROIs, 3 conditions);
* shap <  0.45: a list of 3 arrays (blocks, 25, 481), which ``np.savez`` stores as one array
  (3 conditions, blocks, 25, 481).

Every reader used ``reshape(-1, 25, 481, 3)``, which is right for the first layout only: on the
second it silently mixes conditions, blocks, time points and ROIs (no error, same number of
values). ``as_block_time_roi_condition`` returns the first layout for both, and refuses any other
shape.
"""
import numpy as np

T, R, C = 25, 481, 3


def as_block_time_roi_condition(values):
    """(blocks, 25, 481, 3), from either layout written by ``main_shap.py``."""
    values = np.asarray(values)
    if values.ndim == 4 and values.shape[1:] == (T, R, C):
        return values
    if values.ndim == 4 and values.shape[0] == C and values.shape[2:] == (T, R):
        return np.moveaxis(values, 0, -1)          # list of 3 condition arrays (shap < 0.45)
    raise ValueError(f"SHAP array of shape {values.shape}: expected (blocks, {T}, {R}, {C}) "
                     f"or ({C}, blocks, {T}, {R})")


def load_shap(path):
    """(shap (blocks, 25, 481, 3), id1, id2) of one ``shap-*.npz`` file."""
    z = np.load(path)
    return (as_block_time_roi_condition(z["shap"]),
            np.asarray(z["id1"]).ravel(), np.asarray(z["id2"]).ravel())


def layout(path):
    """'blocks x time x ROI x condition' or 'condition x blocks x time x ROI (shap < 0.45)'."""
    shape = np.load(path, mmap_mode="r")["shap"].shape
    if len(shape) == 4 and shape[1:] == (T, R, C):
        return f"{shape}: blocks x time x ROI x condition"
    if len(shape) == 4 and shape[0] == C and shape[2:] == (T, R):
        return f"{shape}: condition x blocks x time x ROI (shap < 0.45, reordered when read)"
    return f"{shape}: unknown"
