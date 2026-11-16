"""Compare the saved DeepExplainer attributions with a second explainer on a few participants.

``shap.DeepExplainer`` has no rule for the LSTM and LayerNorm modules of the model. This tool
recomputes the attributions of the same blocks with ``shap.GradientExplainer`` (expected
gradients, which only needs the gradients of the model) and compares the two:

* ROI level: correlation of the two maps (25 time points x 481 ROIs) of each block, for the
  condition of the block;
* network level: block-normalised projection on the 18 templates (utils/network.py), then the
  Spearman correlation of the 18 networks at each second, and of the mean of each network
  over the blocks and seconds of each condition (the ordering of the figures).

No window cache is needed: the inputs of the blocks are stored in the block prediction files
(first element of ``<checkpoint>_sub-true_type-block_step-*_n-all.pt``). It needs the
checkpoint, the block predictions of the population and of the training set (background),
and the SHAP files of the population.

Usage::

    python tools/compare_explainers.py --pop test --participants 3 --background 200
"""
import glob
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from config import CHECKPOINT, PREDICTIONS_DIR, SHAP_DIR  # noqa: E402
from tools.check_shap_additivity import block_keys  # noqa: E402
from utils.network import load_templates, network_projection  # noqa: E402

CONDITIONS = ["Worry", "Neutral", "Reappraisal"]
TEMPLATES = os.path.join(REPO, "external", "schaeffer_subcortical_smith_templates.csv")


def load_block_file(name, checkpoint, step):
    import torch
    path = os.path.join(PREDICTIONS_DIR, name, f"{Path(checkpoint).stem}_sub-true_type-block_step-{step}_n-all.pt")
    if not os.path.exists(path):
        sys.exit(f"Not found: {path}")
    data, prediction, labels, id1, id2 = torch.load(path, weights_only=False)[:5]   # local file
    return data, np.asarray(labels, dtype=float), id1, id2


def load_shap(pop, shap_dir, n_files):
    files = sorted(glob.glob(os.path.join(shap_dir, f"shap-lstmv4vf_pop-{pop}_batch-*.npz")),
                   key=lambda f: int(f.rsplit("-", 1)[1].split(".")[0]))[:n_files]
    if not files:
        sys.exit(f"No SHAP file shap-lstmv4vf_pop-{pop}_batch-*.npz in {shap_dir}")
    shap, id1, id2 = [], [], []
    for f in files:
        z = np.load(f)
        shap.append(z["shap"].reshape(-1, 25, 481, 3))
        id1.extend(np.asarray(z["id1"]).ravel())
        id2.extend(np.asarray(z["id2"]).ravel())
    return np.concatenate(shap), np.array(id1), np.array(id2)


def compare(deep, other, true_class, weights):
    """Agreement of two attribution arrays (blocks, 25, 481, 3) for the condition of each block."""
    b = np.arange(len(deep))
    d, o = deep[b, ..., true_class], other[b, ..., true_class]           # (blocks, 25, 481)
    roi_r = np.array([np.corrcoef(x.ravel(), y.ravel())[0, 1] for x, y in zip(d, o)])
    nd = np.abs(network_projection(deep, weights))[b, ..., true_class]   # (blocks, 25, 18)
    no = np.abs(network_projection(other, weights))[b, ..., true_class]
    second_rho = np.array([spearmanr(x, y)[0] for x, y in zip(nd.reshape(-1, 18), no.reshape(-1, 18))])
    rows = [{"level": "ROI map of each block (Pearson r)", "condition": "all", "median": np.nanmedian(roi_r),
             "q25": np.nanquantile(roi_r, .25), "q75": np.nanquantile(roi_r, .75)},
            {"level": "18 networks at each second (Spearman rho)", "condition": "all", "median": np.nanmedian(second_rho),
             "q25": np.nanquantile(second_rho, .25), "q75": np.nanquantile(second_rho, .75)}]
    for c, name in enumerate(CONDITIONS):
        m = true_class == c
        if m.sum():
            rho = spearmanr(nd[m].mean(axis=(0, 1)), no[m].mean(axis=(0, 1)))[0]
            rows.append({"level": "order of the 18 networks (Spearman rho)", "condition": name,
                         "median": rho, "q25": np.nan, "q75": np.nan})
    return pd.DataFrame(rows)


def main():
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pop", default="test", choices=["test", "val"])
    parser.add_argument("--participants", type=int, default=3, help="participants explained (34 blocks each)")
    parser.add_argument("--background", type=int, default=200, help="training blocks drawn as background")
    parser.add_argument("--nsamples", type=int, default=200, help="samples of GradientExplainer per block")
    parser.add_argument("--name", default="lstmv4vlt")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--shap_dir", default=SHAP_DIR)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", default=None, help="optional .npz with the new attributions")
    args = parser.parse_args()

    import shap as shap_lib
    import torch
    from models.lstm_v4 import LSTMClassifier

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # saved attributions of the first participants (one SHAP file = 3 participants x 34 blocks)
    deep, sid1, sid2 = load_shap(args.pop, args.shap_dir, n_files=int(np.ceil(args.participants / 3)))
    keep = np.isin(sid1, pd.unique(sid1)[:args.participants])
    deep, sid1, sid2 = deep[keep], sid1[keep], sid2[keep]

    # inputs of the same blocks (block prediction file), aligned as export_R_data.py
    data, labels, pid1, pid2 = load_block_file(args.name, args.checkpoint, args.pop)
    position = {k: i for i, k in enumerate(block_keys(pid1, pid2))}
    skeys = block_keys(sid1, sid2)
    missing = [k for k in skeys if k not in position]
    if missing:
        sys.exit(f"{len(missing)} SHAP blocks have no input (e.g. {missing[:3]})")
    idx = np.array([position[k] for k in skeys])
    x = torch.as_tensor(np.asarray(data)[idx], dtype=torch.float32)
    true_class = labels[idx].sum(axis=1).argmax(axis=-1)

    train_data = np.asarray(load_block_file(args.name, args.checkpoint, "train")[0])
    background = torch.as_tensor(train_data[rng.choice(len(train_data), min(args.background, len(train_data)), replace=False)],
                                 dtype=torch.float32)

    model = LSTMClassifier.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()

    class LastTimePoint(torch.nn.Module):                    # same explained output as main_shap.py
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, inp):
            return self.m(inp)[:, -1, :]

    explainer = shap_lib.GradientExplainer(LastTimePoint(model), background)
    print(f"GradientExplainer on {len(x)} blocks ({args.participants} participants), background {len(background)} blocks")
    other = np.asarray(explainer.shap_values(x, nsamples=args.nsamples, rseed=args.seed)).reshape(-1, 25, 481, 3)

    weights = load_templates(TEMPLATES)[2]
    table = compare(deep, other, true_class, weights)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(table.to_string(index=False))
    if args.out:
        np.savez_compressed(args.out, shap=other, id1=sid1, id2=sid2)


if __name__ == "__main__":
    main()
