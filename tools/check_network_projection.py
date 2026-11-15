"""Which projection produced the shap_network_* columns of an existing export?

Recomputes the 54 network columns from the 1443 shap_roi_* columns of the same rows, in two ways:
  correct  : each (block, time, condition) map is L2-normalised over the 481 ROIs and projected
             on the L2-normalised templates (projection on the ROI axis)
  notebook : same normalisation, then np.matmul(x.reshape(-1, 481), W.T).reshape(-1, 25, 18, 3)
             as in visualization/base.py (generate_network_heatmap) and results_article.ipynb,
             which mixes ROIs and conditions
and reports which one matches the stored columns. Nothing leaves the machine.

Usage:
    python tools/check_network_projection.py <export_*_block_with_shap.csv> external/schaeffer_subcortical_smith_templates.csv [n_rows]
"""
import sys
import numpy as np
import pandas as pd

export_file, template_file = sys.argv[1], sys.argv[2]
n_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 500
conditions = ["worry", "neutral", "reappraisal"]

network = pd.read_csv(template_file)
net_names = list(network.iloc[:, 0])
W = network.iloc[:, 1:].to_numpy(dtype=float)
W = W / np.sqrt(np.sum(W ** 2, axis=1, keepdims=True))

df = pd.read_csv(export_file, nrows=n_rows)
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
candidates = {
    "correct (projection on the ROI axis)": np.einsum("nrc,kr->nkc", x_norm, W),
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
