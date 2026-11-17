"""Deterministic expected gradients: integrated gradients averaged over every training block.

Expected gradients (the estimator of ``shap.GradientExplainer``) average, over baselines x' drawn
from the background and points t ~ U(0, 1), (x - x') * grad f(x' + t (x - x')). This tool removes
both random draws:

* every training block (the background of main_shap.py, 3,128 blocks) is used once as baseline;
* the integral over t is computed on a fixed grid (``--method``, ``--n_steps``) with Captum's
  ``IntegratedGradients``. Default: midpoint rule, 64 points. The model's ReLU layers make the
  gradient jump along the path, so a smooth rule (Gauss-Legendre) gains little; on the model
  architecture the midpoint rule's completeness error per baseline fell to a median of 2.6 % at
  64 points and 1.7 % at 128 (errors partly cancel across baselines).

The attribution of a block is then mean_b IG(x; x'_b), the same at every run. It adds up to
f(x) - mean_b f(x'_b) up to the quadrature error (tools/check_shap_additivity.py: r and slope
close to 1). Explained output: the prediction at the last second, as main_shap.py.

The work of one SHAP batch (3 participants x 34 blocks) is split by baseline chunk:

    python tools/expected_gradients_all_baselines.py --pop test --batch_idx 0 --chunk 3 --n_chunks 16
        -> <out>/partial/pop-test_batch-0_chunk-3of16.npz   (sum over the baselines of the chunk)
    python tools/expected_gradients_all_baselines.py --merge
        -> <out>/shap-lstmv4vf_pop-<pop>_batch-<b>.npz     (mean over all baselines; file name of
           main_shap.py, so phase_protocol.py, check_shap_additivity.py and compare_explainers.py
           run on it with --shap_dir / --other_dir)

Needs the checkpoint, the block predictions of the population and of the training set (inputs),
the DeepExplainer SHAP file of the batch (block order and identifiers), and captum (>= 0.7).
"""
import glob
import os
import re
import sys
from argparse import ArgumentParser

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from config import CHECKPOINT, SHAP_DIR  # noqa: E402
from tools.check_shap_additivity import block_keys  # noqa: E402
from tools.compare_explainers import load_block_file  # noqa: E402

POPULATION_LABELS = {"val": "FINA: test set", "test": "RAW: external validation set"}


def chunk_slice(n, chunk, n_chunks):
    """Contiguous slice of the n baselines for chunk ``chunk`` of ``n_chunks``."""
    bounds = np.linspace(0, n, n_chunks + 1).round().astype(int)
    return slice(bounds[chunk], bounds[chunk + 1])


def compute(args):
    import torch
    from captum.attr import IntegratedGradients
    from models.lstm_v4 import LSTMClassifier

    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    torch.set_num_threads(threads)

    # blocks of the batch, in the order of the DeepExplainer SHAP file
    path = os.path.join(args.shap_dir, f"shap-lstmv4vf_pop-{args.pop}_batch-{args.batch_idx}.npz")
    if not os.path.exists(path):
        sys.exit(f"Not found: {path}")
    z = np.load(path)
    sid1, sid2 = np.asarray(z["id1"]).ravel(), np.asarray(z["id2"]).ravel()
    data, _, _, pid1, pid2 = load_block_file(args.name, args.checkpoint, args.pop)
    position = {k: i for i, k in enumerate(block_keys(pid1, pid2))}
    skeys = block_keys(sid1, sid2)
    missing = [k for k in skeys if k not in position]
    if missing:
        sys.exit(f"{len(missing)} SHAP blocks have no input (e.g. {missing[:3]})")
    x = torch.as_tensor(data[[position[k] for k in skeys]])

    background = torch.as_tensor(load_block_file(args.name, args.checkpoint, "train")[0])
    sl = chunk_slice(len(background), args.chunk, args.n_chunks)
    baselines = background[sl]

    model = LSTMClassifier.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()

    def last_time_point(inp):                       # same explained output as main_shap.py
        return model(inp)[:, -1, :]

    ig = IntegratedGradients(last_time_point)
    total = np.zeros(tuple(x.shape) + (3,))
    print(f"{POPULATION_LABELS[args.pop]}, batch {args.batch_idx}: {len(x)} blocks x baselines "
          f"{sl.start}-{sl.stop - 1} ({len(baselines)} of {len(background)}), {args.method} with {args.n_steps} "
          f"points, {threads} threads", flush=True)
    for i, b in enumerate(baselines):
        for c in range(3):
            attr = ig.attribute(x, baselines=b.unsqueeze(0), target=c, n_steps=args.n_steps,
                                method=args.method, internal_batch_size=args.internal_batch_size)
            total[..., c] += attr.detach().double().numpy()
        if (i + 1) % 10 == 0 or i + 1 == len(baselines):
            print(f"  {i + 1}/{len(baselines)} baselines", flush=True)

    out = os.path.join(args.out, "partial")
    os.makedirs(out, exist_ok=True)
    name = os.path.join(out, f"pop-{args.pop}_batch-{args.batch_idx}_chunk-{args.chunk}of{args.n_chunks}.npz")
    np.savez_compressed(name, sum=total, n_baselines=len(baselines), first=sl.start, stop=sl.stop,
                        n_background=len(background), n_steps=args.n_steps, method=args.method, id1=sid1, id2=sid2)
    print(f"Written: {name}")


def merge(args):
    files = glob.glob(os.path.join(args.out, "partial", "pop-*_batch-*_chunk-*of*.npz"))
    if not files:
        sys.exit(f"No partial file in {os.path.join(args.out, 'partial')}")
    groups = {}
    for f in files:
        pop, batch, chunk, n_chunks = re.search(r"pop-(\w+)_batch-(\d+)_chunk-(\d+)of(\d+)\.npz", f).groups()
        groups.setdefault((pop, int(batch), int(n_chunks)), {})[int(chunk)] = f
    incomplete = []
    for (pop, batch, n_chunks), chunks in sorted(groups.items()):
        if sorted(chunks) != list(range(n_chunks)):
            incomplete.append(f"{pop} batch {batch}: chunks {sorted(set(range(n_chunks)) - set(chunks))} missing")
            continue
        parts = [np.load(chunks[c]) for c in range(n_chunks)]
        settings = {(int(p["n_steps"]), str(p["method"])) for p in parts}
        if len(settings) > 1:
            incomplete.append(f"{pop} batch {batch}: chunks computed with different grids {sorted(settings)}")
            continue
        n_background = int(parts[0]["n_background"])
        covered = sorted((int(p["first"]), int(p["stop"])) for p in parts)
        if covered[0][0] != 0 or covered[-1][1] != n_background or any(a[1] != b[0] for a, b in zip(covered, covered[1:])):
            incomplete.append(f"{pop} batch {batch}: baselines not covered exactly once")
            continue
        n = sum(int(p["n_baselines"]) for p in parts)
        mean = sum(p["sum"] for p in parts) / n
        out = os.path.join(args.out, f"shap-lstmv4vf_pop-{pop}_batch-{batch}.npz")
        n_steps, method = settings.pop()
        np.savez_compressed(out, shap=mean.astype(np.float32), id1=parts[0]["id1"], id2=parts[0]["id2"],
                            n_baselines=n, n_steps=n_steps, method=method)
        print(f"{POPULATION_LABELS.get(pop, pop)} batch {batch}: {n} baselines, {method} {n_steps} points -> {out}")
    if incomplete:
        print("Not merged:\n  " + "\n  ".join(incomplete))
        sys.exit(1)


def main():
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pop", default="test", choices=["test", "val"],
                        help="test = RAW external validation set, val = FINA test set")
    parser.add_argument("--batch_idx", type=int, default=0, help="SHAP batch of main_shap.py (3 participants)")
    parser.add_argument("--chunk", type=int, default=0, help="chunk of baselines computed by this run")
    parser.add_argument("--n_chunks", type=int, default=1, help="number of baseline chunks")
    parser.add_argument("--n_steps", type=int, default=64, help="grid points over t")
    parser.add_argument("--method", default="riemann_middle",
                        choices=["riemann_middle", "riemann_trapezoid", "riemann_left", "riemann_right", "gausslegendre"],
                        help="quadrature rule of Captum IntegratedGradients")
    parser.add_argument("--internal_batch_size", type=int, default=1024, help="scaled inputs per forward pass")
    parser.add_argument("--name", default="lstmv4vlt")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--shap_dir", default=SHAP_DIR, help="DeepExplainer SHAP files (block order)")
    parser.add_argument("--out", default=os.path.join(REPO, "results", "expected_gradients_all"))
    parser.add_argument("--merge", action="store_true", help="merge the partial files of --out")
    args = parser.parse_args()
    if args.merge:
        return merge(args)
    if not 0 <= args.chunk < args.n_chunks:
        sys.exit("--chunk must be in [0, --n_chunks)")
    compute(args)


if __name__ == "__main__":
    main()
