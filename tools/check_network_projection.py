"""Which projection produced the shap_network_* columns of an existing export?

Recomputes the 54 network columns from the 1443 shap_roi_* columns of the same rows, in three ways:
  block     : the map of each block and condition (25 time points x 481 ROIs) is L2-normalised
              over time and ROIs and projected on the L2-normalised templates along the ROI axis
              (current exports, utils/network.py)
  timepoint : each (block, time, condition) map is L2-normalised over the 481 ROIs, same
              projection (exports before the block normalisation)
  notebook  : per-time-point normalisation, then np.matmul(x.reshape(-1, 481), W.T).reshape(-1, 25, 18, 3)
              as in the first versions of visualization/base.py and results_article.ipynb, which
              mixes ROIs and conditions
and reports which one matches the stored columns. The rows of an export are ordered block by
block (25 rows per block), so whole blocks are read. Nothing leaves the machine.

Usage:
    python tools/check_network_projection.py <export_*_block_with_shap.csv> external/schaeffer_subcortical_smith_templates.csv [n_rows]
"""
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.network import load_templates, network_projection  # noqa: E402

export_file, template_file = sys.argv[1], sys.argv[2]
T = 25                                                         # time points per block
n_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 500
n_rows = max(T, n_rows // T * T)                               # whole blocks
conditions = ["worry", "neutral", "reappraisal"]

net_names, _, W = load_templates(template_file)

df = pd.read_csv(export_file, nrows=n_rows)
if len(df) % T or ("timepoint" in df and not (df["timepoint"].to_numpy().reshape(-1, T) == np.arange(T)).all()):
    sys.exit(f"The first {len(df)} rows are not whole blocks of {T} time points")
roi_cols = {c: [col for col in df.columns if col.startswith(f"shap_roi_{c}_")] for c in conditions}
if any(len(v) != 481 for v in roi_cols.values()):
    sys.exit(f"Expected 481 shap_roi_<condition>_* columns per condition, found {[len(v) for v in roi_cols.values()]}")
missing = [f"shap_network_{c}_{n}" for c in conditions for n in net_names if f"shap_network_{c}_{n}" not in df.columns]
if missing:
    sys.exit(f"Network columns not found (first ones): {missing[:3]}")

x = np.stack([df[roi_cols[c]].to_numpy(dtype=float) for c in conditions], axis=-1)        # (rows, 481, 3)
stored = np.stack([df[[f"shap_network_{c}_{n}" for n in net_names]].to_numpy(dtype=float)
                   for c in conditions], axis=-1)                                          # (rows, 18, 3)
x_norm = x / np.sqrt(np.sum(x ** 2, axis=1, keepdims=True))
blocks = x.reshape(-1, T, 481, 3)
candidates = {
    "block (L2 over time and ROIs, ROI axis)": network_projection(blocks, W, "block").reshape(-1, 18, 3),
    "timepoint (L2 over ROIs, ROI axis)": network_projection(blocks, W, "timepoint").reshape(-1, 18, 3),
    "notebook (reshape(-1, 481) on (..., 481, 3))": np.matmul(x_norm.reshape(-1, 481), W.T).reshape(-1, 18, 3),
}
print(f"{len(df)} rows of {export_file}")
for name, value in candidates.items():
    ok = np.isfinite(value) & np.isfinite(stored)
    diff = np.abs(value[ok] - stored[ok]).max()
    corr = np.corrcoef(value[ok], stored[ok])[0, 1]
    print(f"  {name:46s} max |difference| = {diff:.2e}   correlation = {corr:.4f}")
best = min(candidates, key=lambda k: np.nanmax(np.abs(candidates[k] - stored)))
if np.nanmax(np.abs(candidates[best] - stored)) < 1e-6:
    print(f"-> The stored shap_network_* columns were computed with the {best.split(' ')[0].upper()} method.")
else:
    print("-> Neither method reproduces the stored columns exactly: another normalisation was used.")
