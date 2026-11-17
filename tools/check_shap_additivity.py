"""Do the saved DeepExplainer attributions add up to the predictions they explain?

SHAP values satisfy, for each block and condition c,

    sum over time points and ROIs of shap[block, :, :, c]  =  f_c(block) - E_background[f_c]

where f_c is the prediction at the last time point of the 25 s block window (main_shap.py) and
the background is the set of training blocks. ``main_shap.py`` ran DeepExplainer with
``check_additivity=False`` (no rule for LSTM and LayerNorm modules), so this property was not
checked. This tool checks it afterwards, without the model or the window caches: it only reads

    config.SHAP_DIR         shap-lstmv4vf_pop-<pop>_batch-*.npz   (attributions)
    config.PREDICTIONS_DIR  <name>/<checkpoint>_sub-true_type-block_step-{train,<pop>}_n-all.pt
                            (predictions of the blocks explained and of the training blocks)

For each condition it reports the correlation and the regression of the sum of the SHAP values
on f_c - E[f_c], and the error relative to |f_c - E[f_c]|. A constant offset only means that the
background mean differs from the mean of the training predictions; a low correlation or a slope
far from 1 means that the attributions do not decompose the predictions (their amounts are then
unreliable, the spatial patterns less affected).

Usage::

    python tools/check_shap_additivity.py                 # test = RAW external validation set
    python tools/check_shap_additivity.py --pop val       # val = FINA test set
"""
import glob
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from config import CHECKPOINT, PREDICTIONS_DIR, SHAP_DIR  # noqa: E402
from utils.shap_io import load_shap  # noqa: E402

CONDITIONS = ["Worry", "Neutral", "Reappraisal"]


def norm_id(values):
    """Identifiers as strings without prefix or decimal part (as export_R_data.py)."""
    s = pd.Series(values).astype(str).str.replace("RAW_", "", regex=False).str.strip()
    return s.str.replace(r"\.0$", "", regex=True).to_numpy()


def block_keys(id1, id2):
    """participant + index of the block within the participant (order of the blocks)."""
    key = pd.Series(norm_id(id1)) + "_" + pd.Series(norm_id(id2))
    return (key + "_" + key.groupby(key).cumcount().astype(str)).to_numpy()


def load_predictions(name, checkpoint, step):
    import torch
    path = os.path.join(PREDICTIONS_DIR, name, f"{Path(checkpoint).stem}_sub-true_type-block_step-{step}_n-all.pt")
    if not os.path.exists(path):
        sys.exit(f"Not found: {path}")
    data, prediction, labels, id1, id2 = torch.load(path, weights_only=False)[:5]   # local file
    return np.asarray(prediction, dtype=float), np.asarray(labels, dtype=float), id1, id2


def load_shap_sums(pop, shap_dir):
    """Sum of the SHAP values of each block over time and ROIs, read batch by batch: (blocks, 3)."""
    files = sorted(glob.glob(os.path.join(shap_dir, f"shap-lstmv4vf_pop-{pop}_batch-*.npz")),
                   key=lambda f: int(f.rsplit("-", 1)[1].split(".")[0]))
    if not files:
        sys.exit(f"No SHAP file shap-lstmv4vf_pop-{pop}_batch-*.npz in {shap_dir}")
    sums, id1, id2 = [], [], []
    for f in files:
        shap, i1, i2 = load_shap(f)
        sums.append(shap.sum(axis=(1, 2)))
        id1.extend(i1)
        id2.extend(i2)
    return np.vstack(sums), np.array(id1), np.array(id2)


def report(shap_sum, f, expected, true_class):
    delta = f - expected                                   # what the attributions should add up to
    rows = []
    for c, name in enumerate(CONDITIONS):
        for subset, mask in (("all blocks", np.ones(len(f), bool)), (f"{name} blocks", true_class == c)):
            x, y = delta[mask, c], shap_sum[mask, c]
            slope, intercept = np.polyfit(x, y, 1)
            rows.append({
                "condition": name, "blocks": subset, "n": int(mask.sum()),
                "r": np.corrcoef(x, y)[0, 1], "slope": slope, "intercept": intercept,
                "median |sum - delta| / median |delta|": np.median(np.abs(y - x)) / np.median(np.abs(x)),
            })
    return pd.DataFrame(rows)


def main():
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pop", default="test", choices=["test", "val"])
    parser.add_argument("--name", default="lstmv4vlt", help="predictions/<name>/ folder")
    parser.add_argument("--checkpoint", default=CHECKPOINT, help="only its file name is used (prediction file names)")
    parser.add_argument("--shap_dir", default=SHAP_DIR)
    parser.add_argument("--out", default=None, help="optional CSV of the table")
    args = parser.parse_args()

    shap_sum, sid1, sid2 = load_shap_sums(args.pop, args.shap_dir)
    pred, labels, pid1, pid2 = load_predictions(args.name, args.checkpoint, args.pop)
    train_pred = load_predictions(args.name, args.checkpoint, "train")[0]
    expected = train_pred[:, -1, :].mean(axis=0)            # DeepExplainer: mean output on the background

    # align the predictions on the SHAP blocks (participant + block index), as export_R_data.py
    position = {k: i for i, k in enumerate(block_keys(pid1, pid2))}
    skeys = block_keys(sid1, sid2)
    missing = [k for k in skeys if k not in position]
    if missing:
        sys.exit(f"{len(missing)} SHAP blocks have no prediction (e.g. {missing[:3]})")
    idx = np.array([position[k] for k in skeys])
    f = pred[idx, -1, :]                                     # explained output: last time point
    true_class = labels[idx].sum(axis=1).argmax(axis=-1)    # condition of the block

    print(f"{len(skeys)} blocks of population {args.pop}; E[f] on {len(train_pred)} training blocks = "
          + ", ".join(f"{c} {e:.3f}" for c, e in zip(CONDITIONS, expected)))
    table = report(shap_sum, f, expected, true_class)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(table.to_string(index=False))
    ok = table[table["blocks"] == "all blocks"]
    if (ok["r"] > 0.95).all() and ((ok["slope"] - 1).abs() < 0.1).all():
        print("-> The attributions add up to the predictions (r > 0.95, slope within 10 % of 1).")
    else:
        print("-> The attributions do not add up to the predictions: their amounts are approximate "
              "(report it as a limitation; the spatial patterns can still be compared).")
    if args.out:
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
