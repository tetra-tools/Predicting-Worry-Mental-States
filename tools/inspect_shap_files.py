"""Provenance of the 38 SHAP files: date, axis layout, blocks and participants of each file.

History of the files (JYC-HK-wfMRIWorryPrediction):

* val  (FINA test set, batches 0-7): run_shap.py of 2024-12-30 (commits b7f8139 to 1807b1d),
  one job looping over the 8 validation batches; written as ``shap-lstmv3vf_batch-<i>.npz``
  (the name says v3, the model is the published lstm_v4 checkpoint), renamed afterwards to
  ``shap-lstmv4vf_pop-val_batch-<i>.npz``.
* test (RAW external validation set, batches 0-29): run_shap.py of 2025-02-05 (6abbb7d),
  SLURM array ``--array=1-30``: tasks 1-29 wrote batches 1-29, task 30 found no batch 30, and
  batch 0 was not written by this array (another run made it).

Both runs: published checkpoint, output = prediction at the last second of the 25 s block,
background = all training blocks (92 x 34), batches of 102 blocks (3 participants),
check_additivity=False, environment lstmstate2 whose requirements pinned shap>=0.46 (one array
(blocks, 25, 481, 3)). This tool checks what the files actually contain:

    python tools/inspect_shap_files.py            # config.SHAP_DIR
    python tools/inspect_shap_files.py --shap_dir /path/to/shap
"""
import argparse
import datetime
import glob
import os
import sys

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from config import SHAP_DIR  # noqa: E402
from utils.shap_io import load_shap, layout  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shap_dir", default=SHAP_DIR)
    args = parser.parse_args()
    try:
        import shap
        print(f"shap installed in this environment: {shap.__version__} "
              f"({'one array per call' if tuple(map(int, shap.__version__.split('.')[:2])) >= (0, 45) else 'list of arrays per output'})")
    except ImportError:
        print("shap is not installed in this environment")

    for pop, n in (("val", 8), ("test", 30)):
        print(f"\n{pop}: expected batches 0-{n - 1}")
        seen = set()
        for i in range(n):
            path = os.path.join(args.shap_dir, f"shap-lstmv4vf_pop-{pop}_batch-{i}.npz")
            if not os.path.exists(path):
                print(f"  batch {i:2d}: MISSING")
                continue
            values, id1, id2 = load_shap(path)
            participants = list(dict.fromkeys((str(a), str(b)) for a, b in zip(id1, id2)))
            overlap = seen & set(participants)
            seen |= set(participants)
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
            print(f"  batch {i:2d}: {mtime}  {layout(path).split(':')[0]:>22}  {values.shape[0]:3d} blocks, "
                  f"{len(participants)} participants ({', '.join(p[0] for p in participants)}), "
                  f"NaN {int(np.isnan(values).sum())}, mean |SHAP| {np.abs(values).mean():.3g}"
                  + (f"  ALSO IN AN EARLIER BATCH: {sorted(overlap)}" if overlap else ""))
        extra = sorted(set(glob.glob(os.path.join(args.shap_dir, f"shap-lstmv4vf_pop-{pop}_batch-*.npz")))
                       - {os.path.join(args.shap_dir, f"shap-lstmv4vf_pop-{pop}_batch-{i}.npz") for i in range(n)})
        if extra:
            print(f"  unexpected files: {extra}")
        print(f"  {len(seen)} participants in total")
    print("\nCheck: one date per population (batch 0 of test may differ: it was not written by the"
          "\narray job), 102 blocks per batch (34 for the last test batch), no participant twice,"
          "\nlayout 'blocks x time x ROI x condition', and a mean |SHAP| of the same order in every batch.")


if __name__ == "__main__":
    main()
