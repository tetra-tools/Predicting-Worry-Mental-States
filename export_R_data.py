"""Write the per-time-point exports read by R/Article.Rmd.

For each population (train / val / test) and window configuration (training / block / full)::

    R/data/export_<split>_<type>.csv              one row per time point of each window
    R/data/export_<split>_block_with_shap.csv     val and test only, with the SHAP values

Columns: split, id1, id2, block_id, timepoint, label_<condition>, pred_<condition>,
rating (median rating of the window), pswq_total, rsq_total, hars_score, age, sex,
madrs_total, hamd_total, education, common_id (participant enrolled in both cohorts,
NA otherwise); with SHAP: rate (= rating), shap_roi_<condition>_<ROI> (481 x 3) and
shap_network_<condition>_<network> (18 x 3).

Network attributions (Methods: "dot product between L2 regularized ROI importance and
reference networks"): each SHAP map (block, time point, condition) is L2-normalised over
the 481 ROIs and projected on the L2-normalised templates of
external/schaeffer_subcortical_smith_templates.csv, along the ROI axis.

Inputs (paths in config.py, see readme.md, "Re-running the analyses of the paper"): the
checkpoint, the SHAP files of main_shap.py, the cached predictions, common_file.csv and the
clinical files. An export whose prediction file is missing is skipped (unless
WORRY_RECOMPUTE_PREDICTIONS=1, which recomputes it from the window caches).

Example::

    python export_R_data.py                       # checkpoint: config.CHECKPOINT
"""
import os
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from config import CHECKPOINT, COMMON_FILE, FINA_CLINICAL_CSV, RAW_CLINICAL_CSV  # noqa: E402

CONDITIONS = ["worry", "neutral", "reappraisal"]
CLINICAL = ["pswq_total", "rsq_total", "hars_score", "age", "sex", "madrs_total", "hamd_total", "education"]
TEMPLATES = os.path.join(REPO, "external", "schaeffer_subcortical_smith_templates.csv")


def norm_id(values):
    """Identifiers as strings without prefix or decimal part ('RAW_1234', 1234.0 -> '1234')."""
    s = pd.Series(values).astype(str).str.replace("RAW_", "", regex=False).str.strip()
    return s.str.replace(r"\.0$", "", regex=True)


def load_clinical():
    """Clinical variables of each cohort, indexed by the identifier used for the join."""
    fina = pd.read_csv(FINA_CLINICAL_CSV)
    fina["age"] = fina["age_at_mrscan"]
    fina["hamd_total"] = np.nan                      # not collected in cohort 1
    fina["key"] = norm_id(fina["Vault_ScanID"])     # joined on id2 (scan identifier)
    raw = pd.read_csv(RAW_CLINICAL_CSV)
    raw["sex"] = raw["gender"]
    raw["madrs_total"] = np.nan                      # not collected in cohort 2
    if "education" not in raw:
        raw["education"] = np.nan
    raw["key"] = norm_id(raw["raw_id"])             # joined on id1 (participant identifier)
    for df in (fina, raw):
        for col in CLINICAL:
            if col not in df:
                df[col] = np.nan
    return (fina.drop_duplicates("key").set_index("key")[CLINICAL],
            raw.drop_duplicates("key").set_index("key")[CLINICAL])


def load_common():
    """Pairs (cohort-2 participant, cohort-1 participant) of common_file.csv."""
    common = pd.read_csv(COMMON_FILE)
    common = common[common["raw_id"].notna() & common["mr_studyid"].notna()]
    return pd.DataFrame({"raw": norm_id(common["raw_id"]).values, "fina": norm_id(common["mr_studyid"]).values})


def network_projection(shap_values, weights):
    """(samples, 25, 481, 3) -> (samples, 25, 18, 3): L2 over ROIs, then dot product with each template."""
    shap_norm = shap_values / np.sqrt(np.sum(shap_values ** 2, axis=2, keepdims=True))
    return np.einsum("ntrc,kr->ntkc", shap_norm, weights)


def export(v, split, kind, clinical, common, out_dir, templates=None):
    data, prediction, labels, id1, id2, rates, _ = v._get_predictions(split, kind)
    prediction, labels = prediction.numpy(), labels.numpy()
    n, t = prediction.shape[:2]
    id1, id2 = norm_id(id1).values, norm_id(id2).values
    rating = np.array([np.median(np.asarray(r, dtype=float)) if len(r) else np.nan for r in rates])
    window = pd.Series(id1 + "_" + id2).groupby(id1 + "_" + id2).cumcount().values   # window index within participant
    df = pd.DataFrame({
        "split": split,
        "id1": np.repeat(id1, t), "id2": np.repeat(id2, t),
        "block_id": np.repeat(window, t), "timepoint": np.tile(np.arange(t), n),
        **{f"label_{c}": labels[..., k].ravel() for k, c in enumerate(CONDITIONS)},
        **{f"pred_{c}": prediction[..., k].ravel() for k, c in enumerate(CONDITIONS)},
        "rating": np.repeat(rating, t),
    })
    fina, raw = clinical
    if split == "test":
        # participant identifier: id1 (id2 if the clinical file uses it, as TimeSeriesDataset does)
        df = df.join(raw, on="id1").combine_first(df[["id2"]].join(raw, on="id2").drop(columns="id2"))[df.columns.tolist() + CLINICAL]
        df["common_id"] = df["id1"].map(dict(zip(common["raw"], common["fina"])))
    else:
        df = df.join(fina, on="id2")
        df["common_id"] = df["id1"].map(dict(zip(common["fina"], common["raw"])))
    df.to_csv(os.path.join(out_dir, f"export_{split}_{kind}.csv"), index=False)
    print(f"export_{split}_{kind}.csv: {len(df)} rows")

    if kind == "block" and split in ("val", "test"):
        shap = v.shap_vals if split == "val" else v.shap_test                       # (blocks, 25, 481, 3)
        sid1 = norm_id(v.shap_vals_id1 if split == "val" else v.shap_test_id1).values
        sid2 = norm_id(v.shap_vals_id2 if split == "val" else v.shap_test_id2).values
        # align the SHAP blocks on the predictions (participant + block index within participant)
        skey = pd.Series(sid1 + "_" + sid2); skey = skey + "_" + skey.groupby(skey).cumcount().astype(str)
        pkey = pd.Series(id1 + "_" + id2) + "_" + pd.Series(window).astype(str)
        position = dict(zip(skey, range(len(skey))))
        missing = [k for k in pkey if k not in position]
        if missing:
            raise ValueError(f"{len(missing)} {split} blocks have no SHAP values (e.g. {missing[:3]})")
        shap = shap[[position[k] for k in pkey]]
        roi_names, weights, network_names = templates
        network = network_projection(shap, weights)
        cols = {"rate": df["rating"].to_numpy()}
        for k, c in enumerate(CONDITIONS):
            for r, roi in enumerate(roi_names):
                cols[f"shap_roi_{c}_{roi}"] = shap[:, :, r, k].ravel()
        for k, c in enumerate(CONDITIONS):
            for j, net in enumerate(network_names):
                cols[f"shap_network_{c}_{net}"] = network[:, :, j, k].ravel()
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        df.to_csv(os.path.join(out_dir, f"export_{split}_block_with_shap.csv"), index=False)
        print(f"export_{split}_block_with_shap.csv: {len(df)} rows, {df.shape[1]} columns")


def main():
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--name", default="lstmv4vlt", help="results/<name>/ folder (heatmaps)")
    parser.add_argument("--out", default=os.path.join(REPO, "R", "data"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from visualization import Visualization
    v = Visualization(args.checkpoint, args.name)

    network = pd.read_csv(TEMPLATES)
    weights = network.iloc[:, 1:].to_numpy(dtype=float)
    weights = weights / np.sqrt(np.sum(weights ** 2, axis=1, keepdims=True))
    templates = (list(network.columns[1:]), weights, list(network.iloc[:, 0]))

    clinical, common = load_clinical(), load_common()
    skipped = []
    for split in ["train", "val", "test"]:
        for kind in ["training", "block", "full"]:
            try:
                export(v, split, kind, clinical, common, args.out, templates)
            except FileNotFoundError as error:   # prediction file missing (and not recomputed)
                skipped.append(f"export_{split}_{kind}.csv")
                print(f"Skipped export_{split}_{kind}.csv: {error}")
                # an export of an earlier run would otherwise still be read by Article.Rmd
                for stale in (f"export_{split}_{kind}.csv", f"export_{split}_{kind}_with_shap.csv"):
                    if os.path.exists(os.path.join(args.out, stale)):
                        os.remove(os.path.join(args.out, stale))
                        print(f"Removed {stale} of an earlier run")
    if skipped:
        print(f"{len(skipped)} exports skipped (no prediction file): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
