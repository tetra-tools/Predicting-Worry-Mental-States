"""Are the 481 ROIs of the time series in the order of the network templates?

The SHAP value of ROI i is the attribution of column i of ``schaeffer_subcortical_timeseries.csv``
(``TimeSeriesDataset`` keeps every column but the last 3, in file order), and the network
projection weights it with column i + 1 of ``external/schaeffer_subcortical_smith_templates.csv``.
The two files must therefore list the same ROIs in the same order. This tool compares the
header of one time-series file (cohort 1 and cohort 2) with the template columns.

Usage::

    python tools/check_roi_order.py                       # first file found under config.DATA_ROOT
    python tools/check_roi_order.py path/to/schaeffer_subcortical_timeseries.csv
"""
import glob
import os
import re
import sys

import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from config import DATA_ROOT  # noqa: E402

TEMPLATES = os.path.join(REPO, "external", "schaeffer_subcortical_smith_templates.csv")


def key(name):
    """Comparable form of a ROI name: lower case, without a leading 'r' before an AAL name,
    without separators (e.g. 'rThal_AV_L' -> 'thalavl', 'schaefer001' -> 'schaefer1')."""
    name = str(name).strip()
    name = re.sub(r"^r(?=[A-Z])", "", name)
    name = re.sub(r"[^0-9a-zA-Z]", "", name).lower()
    return re.sub(r"0+(\d)", r"\1", name)


def compare(path, rois):
    header = list(pd.read_csv(path, nrows=0).columns)
    series = header[:-3]
    print(f"\n{path}\n  {len(header)} columns: {len(series)} ROIs + task regressors {header[-3:]}")
    if len(series) != len(rois):
        print(f"  MISMATCH: {len(series)} ROI columns, {len(rois)} template columns")
        return False
    exact = sum(a == b for a, b in zip(series, rois))
    same = [key(a) == key(b) for a, b in zip(series, rois)]
    print(f"  identical names: {exact}/{len(rois)}, same ROI after normalising the names: {sum(same)}/{len(rois)}")
    if all(same):
        print("  OK: same ROIs in the same order")
        return True
    bad = [i for i, s in enumerate(same) if not s]
    print(f"  {len(bad)} positions differ, first ones (position: time series / template):")
    for i in bad[:15]:
        print(f"    {i + 1}: {series[i]} / {rois[i]}")
    moved = sum(key(a) in {key(b) for b in rois} for a in series)
    print(f"  {moved}/{len(rois)} time-series ROIs exist in the templates under another position "
          f"(same set, other order)" if moved == len(rois) else
          f"  {moved}/{len(rois)} time-series ROIs found in the templates (names differ: check them by hand)")
    return False


def main():
    rois = list(pd.read_csv(TEMPLATES, nrows=0).columns[1:])
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = []
        for pattern in ("FINA/Public/Analysis/data/*/*/step02_WorryInduction/schaeffer_subcortical_timeseries.csv",
                        "RAW/Public/Analysis/data/*/*/step05_worry/schaeffer_subcortical_timeseries.csv"):
            found = sorted(glob.glob(os.path.join(DATA_ROOT, pattern)))
            paths += found[:1]
        if not paths:
            sys.exit(f"No time-series file under {DATA_ROOT}: give a path")
    ok = all([compare(p, rois) for p in paths])
    print("\nAll files in the order of the templates." if ok else
          "\nAt least one file is NOT in the order of the templates: the network projection would be wrong.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
