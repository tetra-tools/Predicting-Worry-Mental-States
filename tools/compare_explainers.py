"""Recompute the attributions of one SHAP batch with GradientExplainer and compare with DeepExplainer.

``shap.DeepExplainer`` has no rule for the LSTM and LayerNorm modules of the model (its
attributions do not add up to the predictions, tools/check_shap_additivity.py).
``shap.GradientExplainer`` (expected gradients) only needs the gradients of the model. This
tool recomputes, for one batch of ``main_shap.py`` (3 participants x 34 blocks), the
attributions of the same blocks for the same explained output (prediction at the last
second), with the same background (all training blocks by default), and compares the two:

* additivity of each explainer on these blocks (sum vs prediction - background mean);
* ROI maps of each block (Pearson r, condition of the block);
* the 18 networks at each second (Spearman rho of the block-normalised projections);
* the order of the 18 networks per condition (Spearman rho of their means);
* optional: convergence with the number of samples (--convergence 50,100,400) and
  repeatability across seeds (--repeats 1): the same metrics, GradientExplainer against itself.

The GradientExplainer attributions are written with the file name of main_shap.py
(``<out>/shap-lstmv4vf_pop-<pop>_batch-<b>.npz``), so the other analyses can be rerun on them
with ``--shap_dir <out>`` (phase_protocol.py, tools/check_shap_additivity.py). No window cache
is needed: the inputs are stored in the block prediction files. Needs the checkpoint, the block
predictions of the population and of the training set, and the SHAP files of the population.

Usage::

    python tools/compare_explainers.py --pop test --batch_idx 0 --out results/gradient_explainer \\
        --convergence 50,100,400 --repeats 1          # one batch (a SLURM array runs them all)
    python tools/compare_explainers.py --summarize results/gradient_explainer   # all batches
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
from utils.shap_io import as_block_time_roi_condition, load_shap  # noqa: E402
from utils.network import load_templates, network_projection  # noqa: E402

CONDITIONS = ["Worry", "Neutral", "Reappraisal"]
TEMPLATES = os.path.join(REPO, "external", "schaeffer_subcortical_smith_templates.csv")
POPULATION_LABELS = {"val": "FINA: test set", "test": "RAW: external validation set"}


def load_block_file(name, checkpoint, step):
    import torch
    path = os.path.join(PREDICTIONS_DIR, name, f"{Path(checkpoint).stem}_sub-true_type-block_step-{step}_n-all.pt")
    if not os.path.exists(path):
        sys.exit(f"Not found: {path}")
    data, prediction, labels, id1, id2 = torch.load(path, weights_only=False)[:5]   # local file
    return np.asarray(data, dtype=np.float32), np.asarray(prediction, dtype=float), np.asarray(labels, dtype=float), id1, id2


def as_blocks(values, n):
    """SHAP output -> (n, 25, 481, 3) whatever the shap version (list of 3 arrays or one array)."""
    if isinstance(values, list):
        return np.stack([np.asarray(v) for v in values], axis=-1).reshape(n, 25, 481, 3)
    values = np.asarray(values)
    if values.shape[-1] == 3 and values.shape[0] == n:
        return values.reshape(n, 25, 481, 3)
    if values.shape[0] == 3:
        return np.moveaxis(values, 0, -1).reshape(n, 25, 481, 3)
    raise ValueError(f"Unexpected SHAP shape {values.shape}")


def additivity(shap, f, expected, true_class):
    """r and slope of the sum of the attributions on f - E[f], for the condition of each block."""
    b = np.arange(len(shap))
    s = shap.sum(axis=(1, 2))[b, true_class]
    d = (f - expected)[b, true_class]
    return np.corrcoef(d, s)[0, 1], np.polyfit(d, s, 1)[0]


def compare(a, b, true_class, weights):
    """Agreement of two attribution arrays (blocks, 25, 481, 3) for the condition of each block."""
    i = np.arange(len(a))
    da, db = a[i, ..., true_class], b[i, ..., true_class]                  # (blocks, 25, 481)
    roi_r = np.array([np.corrcoef(x.ravel(), y.ravel())[0, 1] for x, y in zip(da, db)])
    na = np.abs(network_projection(a, weights))[i, ..., true_class]         # (blocks, 25, 18)
    nb = np.abs(network_projection(b, weights))[i, ..., true_class]
    second = np.array([spearmanr(x, y)[0] for x, y in zip(na.reshape(-1, 18), nb.reshape(-1, 18))])
    rows = {"roi_map_r": np.nanmedian(roi_r), "networks_per_second_rho": np.nanmedian(second)}
    for c, name in enumerate(CONDITIONS):
        m = true_class == c
        if m.sum():
            rows[f"network_order_rho_{name}"] = spearmanr(na[m].mean(axis=(0, 1)), nb[m].mean(axis=(0, 1)))[0]
            # time course of the share of attribution (seconds 1-24, all networks): same phases?
            ta = (na[m].sum(axis=2) / na[m].sum(axis=(1, 2))[:, None]).mean(axis=0)[:24]
            tb = (nb[m].sum(axis=2) / nb[m].sum(axis=(1, 2))[:, None]).mean(axis=0)[:24]
            rows[f"time_course_r_{name}"] = np.corrcoef(ta, tb)[0, 1]
    return rows


def summarize(folder):
    files = sorted(glob.glob(os.path.join(folder, "compare_pop-*_batch-*.csv")))
    if not files:
        sys.exit(f"No compare_pop-*_batch-*.csv in {folder}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    metrics = [c for c in df.columns if c not in ("pop", "batch", "comparison", "n_blocks")]
    out = df.groupby(["pop", "comparison"])[metrics].median()
    out.insert(0, "batches", df.groupby(["pop", "comparison"]).size())
    out.to_csv(os.path.join(folder, "compare_summary.csv"))
    with pd.option_context("display.width", 250, "display.precision", 3, "display.max_columns", 30):
        print(out.T)
    print(f"Summary of {len(files)} batches (median across batches) in {os.path.join(folder, 'compare_summary.csv')}")


def main():
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pop", default="test", choices=["test", "val"],
                        help="test = RAW external validation set, val = FINA test set")
    parser.add_argument("--batch_idx", type=int, default=0, help="SHAP batch of main_shap.py (3 participants)")
    parser.add_argument("--background", type=int, default=0,
                        help="training blocks used as background (0 = all, as main_shap.py)")
    parser.add_argument("--nsamples", type=int, default=200, help="samples of GradientExplainer per block")
    parser.add_argument("--convergence", default="", help="other numbers of samples to compare, e.g. 50,100,400")
    parser.add_argument("--repeats", type=int, default=0, help="extra runs with other seeds (repeatability)")
    parser.add_argument("--name", default="lstmv4vlt")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--shap_dir", default=SHAP_DIR, help="DeepExplainer SHAP files")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", default=os.path.join(REPO, "results", "gradient_explainer"))
    parser.add_argument("--summarize", default=None, metavar="FOLDER", help="only summarise the batches of FOLDER")
    parser.add_argument("--other_dir", default=None, metavar="FOLDER",
                        help="compare DeepExplainer with the attributions already in FOLDER (same file names as "
                             "main_shap.py, e.g. tools/expected_gradients_all_baselines.py --merge) instead of "
                             "computing GradientExplainer; --nsamples, --convergence and --repeats are ignored")
    parser.add_argument("--other_name", default="ExpectedGradients all baselines", help="label of --other_dir")
    args = parser.parse_args()
    if args.summarize:
        return summarize(args.summarize)

    import torch

    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    torch.set_num_threads(threads)
    os.makedirs(args.out, exist_ok=True)

    # DeepExplainer attributions of the batch
    path = os.path.join(args.shap_dir, f"shap-lstmv4vf_pop-{args.pop}_batch-{args.batch_idx}.npz")
    if not os.path.exists(path):
        sys.exit(f"Not found: {path}")
    deep, sid1, sid2 = load_shap(path)
    deep = deep.astype(np.float64)

    # inputs and predictions of the same blocks, aligned as export_R_data.py
    data, pred, labels, pid1, pid2 = load_block_file(args.name, args.checkpoint, args.pop)
    position = {k: i for i, k in enumerate(block_keys(pid1, pid2))}
    skeys = block_keys(sid1, sid2)
    # keys count blocks within a participant over the whole population: a batch holds whole participants
    missing = [k for k in skeys if k not in position]
    if missing:
        sys.exit(f"{len(missing)} SHAP blocks have no input (e.g. {missing[:3]})")
    idx = np.array([position[k] for k in skeys])
    x = torch.as_tensor(data[idx])
    f = pred[idx, -1, :]
    true_class = labels[idx].sum(axis=1).argmax(axis=-1)

    train_data, train_pred = load_block_file(args.name, args.checkpoint, "train")[:2]
    rng = np.random.default_rng(args.seed)
    keep = (np.arange(len(train_data)) if args.background <= 0 or args.background >= len(train_data)
            else rng.choice(len(train_data), args.background, replace=False))
    background = torch.as_tensor(train_data[keep])
    expected = train_pred[keep, -1, :].mean(axis=0)                       # mean output on the background
    weights = load_templates(TEMPLATES)[2]

    if args.other_dir:                                                    # precomputed attributions
        other_path = os.path.join(args.other_dir, f"shap-lstmv4vf_pop-{args.pop}_batch-{args.batch_idx}.npz")
        if not os.path.exists(other_path):
            sys.exit(f"Not found: {other_path}")
        zo = np.load(other_path)
        if not (np.array_equal(np.asarray(zo["id1"]).ravel().astype(str), sid1.astype(str))
                and np.array_equal(np.asarray(zo["id2"]).ravel().astype(str), sid2.astype(str))):
            sys.exit(f"{other_path}: blocks not in the order of the DeepExplainer file")
        other = as_block_time_roi_condition(zo["shap"]).astype(np.float64)
        rows = []
        for name, values in (("DeepExplainer", deep), (args.other_name, other)):
            r, slope = additivity(values, f, expected, true_class)
            rows.append({"comparison": f"additivity {name}", "additivity_r": r, "additivity_slope": slope})
        rows.append({"comparison": f"DeepExplainer vs {args.other_name}", **compare(deep, other, true_class, weights)})
        table = pd.DataFrame(rows)
        table.insert(0, "n_blocks", len(x))
        table.insert(0, "batch", args.batch_idx)
        table.insert(0, "pop", args.pop)
        out_csv = os.path.join(args.out, f"compare_pop-{args.pop}_batch-{args.batch_idx}.csv")
        table.to_csv(out_csv, index=False)
        with pd.option_context("display.width", 250, "display.precision", 3, "display.max_columns", 30):
            print(table.drop(columns=["pop", "batch", "n_blocks"]).set_index("comparison").T)
        print(f"Written: {out_csv}")
        return

    import shap as shap_lib
    from models.lstm_v4 import LSTMClassifier

    model = LSTMClassifier.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()

    class LastTimePoint(torch.nn.Module):                                # same explained output as main_shap.py
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, inp):
            return self.m(inp)[:, -1, :]

    explainer = shap_lib.GradientExplainer(LastTimePoint(model), background)

    def run(nsamples, seed):
        torch.manual_seed(seed)
        return as_blocks(explainer.shap_values(x, nsamples=nsamples, rseed=seed), len(x))

    print(f"{POPULATION_LABELS[args.pop]}, batch {args.batch_idx}: {len(x)} blocks, background {len(background)} "
          f"training blocks, {args.nsamples} samples per block, {threads} threads")
    grad = run(args.nsamples, args.seed)
    np.savez_compressed(os.path.join(args.out, f"shap-lstmv4vf_pop-{args.pop}_batch-{args.batch_idx}.npz"),
                        shap=grad.astype(np.float32), id1=sid1, id2=sid2)

    rows = []
    for name, values in (("DeepExplainer", deep), (f"GradientExplainer n={args.nsamples}", grad)):
        r, slope = additivity(values, f, expected, true_class)
        rows.append({"comparison": f"additivity {name}", "additivity_r": r, "additivity_slope": slope})
    rows.append({"comparison": "DeepExplainer vs GradientExplainer", **compare(deep, grad, true_class, weights)})
    for k in [int(v) for v in args.convergence.split(",") if v.strip()]:
        other = run(k, args.seed + 1)
        r, slope = additivity(other, f, expected, true_class)
        rows.append({"comparison": f"convergence n={k} vs n={args.nsamples}", "additivity_r": r,
                     "additivity_slope": slope, **compare(other, grad, true_class, weights)})
    for rep in range(1, args.repeats + 1):
        other = run(args.nsamples, args.seed + 100 * rep)
        rows.append({"comparison": f"repeat seed {args.seed + 100 * rep} vs {args.seed}",
                     **compare(other, grad, true_class, weights)})
    table = pd.DataFrame(rows)
    table.insert(0, "n_blocks", len(x))
    table.insert(0, "batch", args.batch_idx)
    table.insert(0, "pop", args.pop)
    out_csv = os.path.join(args.out, f"compare_pop-{args.pop}_batch-{args.batch_idx}.csv")
    table.to_csv(out_csv, index=False)
    with pd.option_context("display.width", 250, "display.precision", 3, "display.max_columns", 30):
        print(table.drop(columns=["pop", "batch", "n_blocks"]).set_index("comparison").T)
    print(f"Written: {out_csv} and the GradientExplainer attributions (same file name as main_shap.py)")


if __name__ == "__main__":
    main()
