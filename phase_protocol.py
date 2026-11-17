"""Pre-specified phase protocol: second-phase contrast and direction of effects (PROTOCOL.md).

Tests the hypotheses of PROTOCOL.md, taken from Karim et al. 2017 (Transl Psychiatry 7:e1204),
on anatomical ROI sets and on the 18 network templates, in two windows fixed from the task
timing (not from the curves):

    induction    seconds 6-15 of the figures: statement (0-9 s of the block) + 5 s BOLD delay
    maintenance  seconds 16-24: worry / reappraisal period (10-24 s) + 5 s delay, last second out
    baseline     seconds 1-2: BOLD of the preceding fixation (before the response to the block)

Four measures per block, set and second (the SHAP head is the condition of the block):

    amount     block-normalised |SHAP| (utils/network.py) averaged over the set with
               non-negative weights (uniform for an anatomical set, positive part of the
               template for a network): the set's share of the block's attribution. Shares
               are compositional: a large effect in one set lowers every other set.
    enrichment amount of the set / median |SHAP| over the 481 ROIs at the same second of the
               same block: how much more than a typical ROI the set contributes. Set-specific
               (a shift of one set barely moves the median) and free of the scale of the
               output: the measure of the hypotheses on the use of a set by the model
    direction  same, each ROI signed by sign(input - mean of the training blocks at that
               second), i.e. by the side of the DeepExplainer background the input lies on;
               direction index = sum / sum of the amounts, from -1 (the attribution comes
               only from activity below the average block) to +1 (only from activity above)
    bold       input (z-scored per participant and ROI) minus its baseline window: activity
               relative to the preceding fixation, the counterpart of "task vs rest" in 2017

Statistics on participant means (blocks averaged within participant): paired or one-sample
t-tests in the predicted direction, slopes across the blocks of a condition (habituation),
regressions on PSWQ; Benjamini-Hochberg FDR within each family. The last second of the
window (25) is never used: it is the input of the explained output itself.

Inputs (config.py; no window cache needed): SHAP files of the population, block predictions
of the population (inputs, labels, identifiers) and of the training set (background mean),
clinical files (or --clinical). Outputs in --out:

    block_values.csv.gz    block x set x window x measure (for mixed models)
    participant_means.csv  participant x condition x set x window x measure
    tests.csv              every test, with its family, FDR q value and verdict
    summary.md             the pre-specified hypotheses and their verdicts
    phase_contrast_<measure>.png, timecourse_<measure>.png (enrichment, amount, bold)

Usage::

    python phase_protocol.py --pop test            # RAW external validation set (confirmatory)
    python phase_protocol.py --pop val             # FINA test set (replication)
"""
import glob
import os
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from config import CHECKPOINT, PREDICTIONS_DIR, SHAP_DIR  # noqa: E402
from utils.network import load_templates  # noqa: E402
from utils.shap_io import load_shap  # noqa: E402

CONDITIONS = ["Worry", "Neutral", "Reappraisal"]
POPULATION_LABELS = {"train": "FINA: training set", "val": "FINA: test set", "test": "RAW: external validation set"}
T = 25
LAST_SECOND = 25                                   # 1-based second of the explained output: left out
WINDOWS = {"baseline": (1, 2), "induction": (6, 15), "maintenance": (16, 24), "block": (6, 24)}
EXPLORATORY_WINDOWS = {"induction_obs": (7, 11), "maintenance_obs": (14, 18)}   # read on the curves
TEMPLATES = os.path.join(REPO, "external", "schaeffer_subcortical_smith_templates.csv")
CENTROIDS = os.path.join(REPO, "external", "schaefer400_7networks_centroids.csv")

# Anatomical ROI sets, fixed before the analysis: Schaefer label patterns, AAL3 names, or boxes on
# the parcel centroids (MNI mm; |x| for both hemispheres)
ROI_SETS = {
    "visual_cortex":          {"label": r"_Vis_"},
    "thalamus":               {"aal": r"^Thal_"},
    "caudate":                {"aal": r"^Caudate_"},
    "sma_dacc":               {"label": r"SalVentAttn_Med|Cont_Cing"},
    "hippocampus":            {"aal": r"^Hippocampus_"},
    "parahippocampal":        {"aal": r"^ParaHippocampal_"},
    "amygdala":               {"aal": r"^Amygdala_"},
    "posterior_insula":       {"box": {"ax": (32, 44), "y": (-28, -2), "z": (-6, 16)}},
    "ba7_superior_parietal":  {"box": {"ax": (12, 36), "y": (-75, -45), "z": (45, 90)}},
    "ba19_lateral_occipital": {"box": {"ax": (20, 90), "y": (-110, -70), "z": (0, 40)}},
    "mpfc":                   {"box": {"ax": (0, 14), "y": (36, 90), "z": (-10, 30)}},
    "ofc":                    {"box": {"ax": (0, 45), "y": (15, 90), "z": (-40, -8)}},
    "dlpfc":                  {"label": r"Cont_PFCl"},
    "supramarginal":          {"box": {"ax": (46, 90), "y": (-50, -22), "z": (20, 50)}},
    "inferior_parietal":      {"box": {"ax": (36, 90), "y": (-75, -50), "z": (20, 50)}},
}

INITIATION = ["visual_cortex", "thalamus", "caudate", "sma_dacc"]
MAINTENANCE = ["thalamus", "hippocampus", "parahippocampal", "amygdala", "posterior_insula"]
REAPPRAISAL_DOWN = ["sma_dacc", "mpfc", "ofc"]
SEVERITY = ["ba19_lateral_occipital", "ba7_superior_parietal", "sma_dacc", "mpfc", "ofc", "dlpfc",
            "supramarginal", "inferior_parietal"]
SEVERITY_DEPRESSION = ["parahippocampal", "hippocampus", "amygdala", "ba7_superior_parietal"]

# Pre-specified hypotheses (Karim et al. 2017). contrast: "W>0" (one-sample), "W-R" (paired),
# "slope_W", "slopeW-slopeN" (across blocks), "pswq" / "pswq_dep" (regression on PSWQ).
# alternative: "greater", "less" or "two-sided" (two-sided = reported, no direction predicted).
HYPOTHESES = (
    [("H1", "initiation: worry activates", "bold", "induction", s, "W>0", "greater") for s in INITIATION]
    + [("H1", "initiation: worry-specific use", "enrichment", "induction", s, "W-N", "greater") for s in ["caudate", "sma_dacc"]]
    + [("H1", "initiation: reappraisal lowers SMA/dACC", "bold", "induction", "sma_dacc", "W-R", "greater")]
    + [("H1", "initiation: shared with reappraisal", "enrichment", "induction", s, "W-R", "two-sided") for s in ["visual_cortex", "thalamus"]]
    + [("H2", "maintenance: second-phase contrast", "enrichment", "maintenance", s, c, "greater") for s in MAINTENANCE for c in ["W-R", "W-N"]]
    + [("H2", "maintenance: worry activates", "bold", "maintenance", s, "W>0", "greater") for s in MAINTENANCE]
    + [("H2", "maintenance: worry > reappraisal", "bold", "maintenance", s, "W-R", "greater") for s in MAINTENANCE]
    + [("H2", "no habituation across worry blocks", "bold", "maintenance", s, "slope_W", "greater") for s in MAINTENANCE]
    + [("H2", "no habituation: worry vs neutral slope", "bold", "maintenance", s, "slopeW-slopeN", "greater") for s in MAINTENANCE]
    + [("H3", "reappraisal lowers medial frontal/OFC", "bold", "induction", s, "R>0", "less") for s in REAPPRAISAL_DOWN]
    + [("H3", "reappraisal attribution from decreases", "direction", "induction", s, "R>0", "less") for s in REAPPRAISAL_DOWN]
    + [("H3", "worry attribution from increases", "direction", "induction", s, "W>0", "greater") for s in INITIATION]
    + [("H4", "severity: activation", "bold", "block", s, "pswq", "greater") for s in SEVERITY]
    + [("H4", "severity: use by the model", "enrichment", "block", s, "pswq", "two-sided") for s in SEVERITY]
    + [("H4", "severity after depression", "bold", "block", s, "pswq_dep", "greater") for s in SEVERITY_DEPRESSION]
)
HYPOTHESIS_COLUMNS = ["family", "hypothesis", "measure", "window", "set", "contrast", "alternative"]


# ----------------------------------------------------------------------------------------- data
def norm_id(values):
    s = pd.Series(values).astype(str).str.replace("RAW_", "", regex=False).str.strip()
    return s.str.replace(r"\.0$", "", regex=True).to_numpy()


def block_keys(id1, id2):
    key = pd.Series(norm_id(id1)) + "_" + pd.Series(norm_id(id2))
    return (key + "_" + key.groupby(key).cumcount().astype(str)).to_numpy()


def load_block_file(name, checkpoint, step):
    import torch
    path = os.path.join(PREDICTIONS_DIR, name, f"{Path(checkpoint).stem}_sub-true_type-block_step-{step}_n-all.pt")
    if not os.path.exists(path):
        sys.exit(f"Not found: {path}")
    data, _, labels, id1, id2 = torch.load(path, weights_only=False)[:5]   # local file
    return np.asarray(data, dtype=np.float32), np.asarray(labels, dtype=float), id1, id2


def shap_files(pop, shap_dir):
    files = sorted(glob.glob(os.path.join(shap_dir, f"shap-lstmv4vf_pop-{pop}_batch-*.npz")),
                   key=lambda f: int(f.rsplit("-", 1)[1].split(".")[0]))
    if not files:
        sys.exit(f"No SHAP file shap-lstmv4vf_pop-{pop}_batch-*.npz in {shap_dir}")
    return files


def set_weights(roi_names, network_names, templates):
    """(sets, 481) non-negative weights summing to 1: anatomical sets then the 18 networks."""
    cen = pd.read_csv(CENTROIDS)
    if len(cen) != 400 or not all(roi_names[i] == f"schaefer{i + 1:03d}" for i in range(400)):
        sys.exit("The first 400 ROIs must be the Schaefer-2018 parcels in atlas order")
    labels = list(cen["label"]) + [r[1:] for r in roi_names[400:]]          # AAL3 names without "r"
    ax, y, z = np.abs(cen["x"].to_numpy()), cen["y"].to_numpy(), cen["z"].to_numpy()
    names, rows = [], []
    for name, rule in ROI_SETS.items():
        m = np.zeros(len(roi_names), bool)
        if "label" in rule:
            m[:400] = [bool(re.search(rule["label"], lab)) for lab in labels[:400]]
        if "aal" in rule:
            m[400:] = [bool(re.search(rule["aal"], lab)) for lab in labels[400:]]
        if "box" in rule:
            b = rule["box"]
            m[:400] = ((ax >= b["ax"][0]) & (ax <= b["ax"][1]) & (y >= b["y"][0]) & (y <= b["y"][1])
                       & (z >= b["z"][0]) & (z <= b["z"][1]))
        if not m.any():
            sys.exit(f"ROI set {name} is empty")
        names.append(name)
        rows.append(m / m.sum())
    for k, net in enumerate(network_names):
        w = np.clip(templates[k], 0, None)
        names.append(f"net:{net}")
        rows.append(w / w.sum())
    return names, np.array(rows), {n: int((r > 0).sum()) for n, r in zip(names, rows)}


def load_scores(pop, participants, clinical_csv=None):
    """PSWQ, age, sex and depression of each participant (key = id1_id2), or None."""
    keys = pd.Series(participants)
    if clinical_csv:
        c = pd.read_csv(clinical_csv, dtype={"key": str}).set_index("key")
        out = pd.DataFrame(index=participants)
        for col in ["pswq_total", "age", "sex", "depression"]:
            k = keys.str.split("_").str[0] if pop == "test" else keys.str.split("_").str[1]
            out[col] = c[col].reindex(k).to_numpy() if col in c else np.nan
        return out
    try:
        from export_R_data import load_clinical
        fina, raw = load_clinical()
    except Exception as error:                                              # missing clinical files
        print(f"Clinical data not loaded ({error}): H4 left out")
        return None
    if pop == "test":
        src, key, dep = raw, keys.str.split("_").str[0], "hamd_total"       # cohort 2: joined on id1
    else:
        src, key, dep = fina, keys.str.split("_").str[1], "madrs_total"     # cohort 1: joined on id2
    src = src.reindex(key.to_numpy())
    return pd.DataFrame({"pswq_total": src["pswq_total"].to_numpy(), "age": src["age"].to_numpy(),
                         "sex": src["sex"].to_numpy(), "depression": src[dep].to_numpy()}, index=participants)


# --------------------------------------------------------------------------------- block values
def compute_block_values(args):
    network_names, roi_names, templates = load_templates(TEMPLATES)
    set_names, Wset, n_rois = set_weights(roi_names, network_names, templates)

    data, labels, pid1, pid2 = load_block_file(args.name, args.checkpoint, args.pop)
    x_bg = load_block_file(args.name, args.checkpoint, "train")[0].mean(axis=0)   # DeepExplainer background
    position = {k: i for i, k in enumerate(block_keys(pid1, pid2))}
    participant = pd.Series(norm_id(pid1)) + "_" + pd.Series(norm_id(pid2))

    # inputs z-scored per participant and ROI (all its blocks and seconds), for the BOLD measure
    z = np.empty_like(data)
    for p in participant.unique():
        idx = np.flatnonzero(participant.to_numpy() == p)
        v = data[idx]
        z[idx] = (v - v.mean(axis=(0, 1))) / (v.std(axis=(0, 1)) + 1e-8)

    windows = {**WINDOWS, **EXPLORATORY_WINDOWS}
    rows, curves = [], []
    for f in shap_files(args.pop, args.shap_dir):
        shap, sid1, sid2 = load_shap(f)                                      # axis order checked
        keys = block_keys(sid1, sid2)
        missing = [k for k in keys if k not in position]
        if missing:
            sys.exit(f"{len(missing)} SHAP blocks have no input (e.g. {missing[:3]})")
        idx = np.array([position[k] for k in keys])
        cond = labels[idx].sum(axis=1).argmax(axis=-1)                       # condition of each block
        phi = shap[np.arange(len(idx)), :, :, cond].astype(np.float64)         # head = own condition
        # block normalisation over the seconds analysed (1-24): the last second is left out of
        # the norm as of every measure (enrichment and direction are ratios: unaffected)
        phi = phi / np.sqrt((phi[:, :LAST_SECOND - 1] ** 2).sum(axis=(1, 2), keepdims=True))
        a = np.abs(phi)
        sign = np.sign(data[idx] - x_bg)
        amount = a @ Wset.T                                                    # (blocks, 25, sets)
        typical = np.median(a, axis=2)                                         # (blocks, 25)
        signed = (a * sign) @ Wset.T
        bz = z[idx] @ Wset.T
        bold = bz - bz[:, WINDOWS["baseline"][0] - 1:WINDOWS["baseline"][1]].mean(axis=1, keepdims=True)
        for j, b in enumerate(idx):
            meta = {"participant": participant[b], "block": int(keys[j].rsplit("_", 1)[1]),
                    "condition": CONDITIONS[cond[j]]}
            for w, (lo, hi) in windows.items():
                if w == "baseline":
                    continue
                sl = slice(lo - 1, hi)                                         # 1-based seconds, inclusive
                rows.append({**meta, "window": w, "amount": amount[j, sl].mean(axis=0),
                             "enrichment": amount[j, sl].sum(axis=0) / typical[j, sl].sum(),
                             "direction_num": signed[j, sl].sum(axis=0), "direction_den": amount[j, sl].sum(axis=0),
                             "bold": bold[j, sl].mean(axis=0)})
            curves.append({**meta, "amount": amount[j, :LAST_SECOND - 1],
                           "enrichment": amount[j, :LAST_SECOND - 1] / typical[j, :LAST_SECOND - 1, None],
                           "bold": bold[j, :LAST_SECOND - 1],
                           "direction": signed[j, :LAST_SECOND - 1]})
    # long tables
    long = []
    for r in rows:
        for s, name in enumerate(set_names):
            long.append((r["participant"], r["block"], r["condition"], r["window"], name, r["amount"][s],
                         r["enrichment"][s], r["direction_num"][s], r["direction_den"][s], r["bold"][s]))
    blocks = pd.DataFrame(long, columns=["participant", "block", "condition", "window", "set", "amount",
                                         "enrichment", "direction_num", "direction_den", "bold"])
    # order of the block among the blocks of its condition (blocks are in scan order in val/test)
    order = (blocks[["participant", "block", "condition"]].drop_duplicates()
             .sort_values(["participant", "block"]))
    order["rank_in_condition"] = order.groupby(["participant", "condition"]).cumcount() + 1
    blocks = blocks.merge(order, on=["participant", "block", "condition"])
    return blocks, curves, set_names, n_rois


def participant_means(blocks):
    g = blocks.groupby(["participant", "condition", "window", "set"])
    out = g[["amount", "enrichment", "bold"]].mean()
    out["direction"] = g["direction_num"].sum() / g["direction_den"].sum()
    out["n_blocks"] = g.size()
    out = out.reset_index().melt(id_vars=["participant", "condition", "window", "set", "n_blocks"],
                                 value_vars=["amount", "enrichment", "direction", "bold"], var_name="measure")
    return out


# ---------------------------------------------------------------------------------------- tests
def _summary(d, alternative, one_sample):
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return dict(n=n, estimate=np.nan, ci_low=np.nan, ci_high=np.nan, t=np.nan, p=np.nan, dz=np.nan)
    res = stats.ttest_1samp(d, 0.0, alternative=alternative)
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return dict(n=n, estimate=d.mean(), ci_low=d.mean() - half, ci_high=d.mean() + half, t=res.statistic,
                p=res.pvalue, dz=d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan)


def slopes(blocks, measure, window, set_name, condition):
    b = blocks[(blocks.window == window) & (blocks.set == set_name) & (blocks.condition == condition)]
    out = {}
    for p, g in b.groupby("participant"):
        if g["rank_in_condition"].nunique() >= 3:
            out[p] = np.polyfit(g["rank_in_condition"], g[measure], 1)[0]
    return pd.Series(out)


def ols_pswq(y, scores, with_depression):
    df = scores.join(y.rename("y"), how="inner")
    cols = ["pswq_total", "age", "sex"] + (["depression"] if with_depression else [])
    df = df[cols + ["y"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < len(cols) + 8 or df["pswq_total"].std() == 0:
        return dict(n=len(df), estimate=np.nan, ci_low=np.nan, ci_high=np.nan, t=np.nan, p=np.nan, dz=np.nan)
    X = df[cols].to_numpy(float)
    X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1)  # standardised predictors
    X = np.column_stack([np.ones(len(df)), X])
    yv = df["y"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    dof = len(yv) - X.shape[1]
    sigma2 = ((yv - X @ beta) ** 2).sum() / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    t = beta[1] / se[1]
    half = stats.t.ppf(0.975, dof) * se[1]
    return dict(n=len(yv), estimate=beta[1], ci_low=beta[1] - half, ci_high=beta[1] + half, t=t, p=None,
                dz=np.nan, _t=t, _dof=dof)


def run_test(means, blocks, scores, measure, window, set_name, contrast, alternative):
    wide = means[(means.measure == measure) & (means.window == window) & (means.set == set_name)] \
        .pivot(index="participant", columns="condition", values="value")
    abbr = {"W": "Worry", "N": "Neutral", "R": "Reappraisal"}
    if contrast.endswith(">0"):
        return _summary(wide.get(abbr[contrast[0]], pd.Series(dtype=float)).to_numpy(float), alternative, True)
    if re.fullmatch(r"[WNR]-[WNR]", contrast):
        a, b = abbr[contrast[0]], abbr[contrast[2]]
        if a not in wide or b not in wide:
            return _summary(np.array([]), alternative, False)
        return _summary((wide[a] - wide[b]).to_numpy(float), alternative, False)
    if contrast == "slope_W":
        return _summary(slopes(blocks, measure, window, set_name, "Worry").to_numpy(float), alternative, True)
    if contrast == "slopeW-slopeN":
        sw, sn = slopes(blocks, measure, window, set_name, "Worry"), slopes(blocks, measure, window, set_name, "Neutral")
        return _summary((sw - sn).dropna().to_numpy(float), alternative, False)
    if contrast in ("pswq", "pswq_dep"):
        if scores is None or "Worry" not in wide:
            return dict(n=0, estimate=np.nan, ci_low=np.nan, ci_high=np.nan, t=np.nan, p=np.nan, dz=np.nan)
        r = ols_pswq(wide["Worry"], scores, contrast == "pswq_dep")
        if "_t" in r:
            r["p"] = {"greater": stats.t.sf(r["_t"], r["_dof"]), "less": stats.t.cdf(r["_t"], r["_dof"]),
                      "two-sided": 2 * stats.t.sf(abs(r["_t"]), r["_dof"])}[alternative]
            r.pop("_t"), r.pop("_dof")
        return r
    raise ValueError(contrast)


def fdr(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        pv = p[ok]
        order = np.argsort(pv)
        ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty_like(pv)
        out[order] = np.minimum(ranked, 1)
        q[ok] = out
    return q


def all_tests(means, blocks, scores, set_names):
    rows = [dict(zip(HYPOTHESIS_COLUMNS, h)) for h in HYPOTHESES]
    # exploratory: every set, second-phase contrasts and directions, both kinds of windows
    for s in set_names:
        for w in ["maintenance", "maintenance_obs"]:
            for c in ["W-R", "W-N", "R-N"]:
                for m in ["enrichment", "amount"]:
                    rows.append(dict(family="E", hypothesis="second-phase contrast", measure=m, window=w,
                                     set=s, contrast=c, alternative="two-sided"))
        for w in ["induction", "maintenance", "induction_obs", "maintenance_obs"]:
            for c in ["W>0", "N>0", "R>0"]:
                for m in ["direction", "bold"]:
                    rows.append(dict(family="E", hypothesis="direction of effect", measure=m, window=w, set=s,
                                     contrast=c, alternative="two-sided"))
    out = []
    for r in rows:
        out.append({**r, **run_test(means, blocks, scores, r["measure"], r["window"], r["set"], r["contrast"],
                                    r["alternative"])})
    tests = pd.DataFrame(out)
    # composition control: estimate minus the median estimate of the 15 anatomical sets for the
    # same measure, window and contrast (a set-specific effect stands out from this median)
    anat = tests[tests.set.isin(list(ROI_SETS))]
    med = anat.groupby(["measure", "window", "contrast"])["estimate"].median().rename("median_set_estimate")
    tests = tests.join(med, on=["measure", "window", "contrast"])
    tests["vs_median_set"] = tests["estimate"] - tests["median_set_estimate"]
    tests = tests.drop(columns="median_set_estimate")
    tests["q_fdr"] = np.nan
    for fam, idx in tests.groupby("family").groups.items():
        tests.loc[idx, "q_fdr"] = fdr(tests.loc[idx, "p"])
    sign_ok = np.where(tests.alternative == "less", tests.estimate < 0,
                       np.where(tests.alternative == "greater", tests.estimate > 0, True))
    tests["verdict"] = np.where(tests.p.isna(), "not tested",
                                np.where((tests.q_fdr < 0.05) & sign_ok,
                                         np.where(tests.alternative == "two-sided", "difference", "supported"),
                                         np.where(tests.alternative == "two-sided", "no difference", "not supported")))
    return tests


# -------------------------------------------------------------------------------------- figures
def figures(means, curves, out_dir, sets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Worry": "#c0392b", "Neutral": "#7f8c8d", "Reappraisal": "#2471a3"}
    for measure in ["enrichment", "amount", "direction", "bold"]:
        fig, axes = plt.subplots(3, 5, figsize=(18, 10), sharex=True)
        for ax, s in zip(axes.ravel(), sets):
            for k, cond in enumerate(CONDITIONS):
                for j, w in enumerate(["induction", "maintenance"]):
                    v = means[(means.measure == measure) & (means.set == s) & (means.window == w)
                              & (means.condition == cond)]["value"].dropna()
                    if len(v) > 1:
                        h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
                        ax.errorbar(j + (k - 1) * 0.2, v.mean(), yerr=h, fmt="o", color=colors[cond],
                                    label=cond if j == 0 else None, capsize=3)
            if measure in ("direction", "bold"):
                ax.axhline(0, color="k", lw=0.5)
            elif measure == "enrichment":
                ax.axhline(1, color="k", lw=0.5)
            ax.set_title(s, fontsize=10)
            ax.set_xticks([0, 1], ["induction\n6-15 s", "maintenance\n16-24 s"])
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{measure}: participant means, 95 % CI across participants")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"phase_contrast_{measure}.png"), dpi=150)
        plt.close(fig)

        # time courses of the anatomical sets (seconds 1-24), participant means
        if measure == "direction":
            continue
        set_index = {s: i for i, s in enumerate(sets)}
        fig, axes = plt.subplots(3, 5, figsize=(18, 10), sharex=True)
        df = pd.DataFrame([{"participant": c["participant"], "condition": c["condition"], "values": c[measure]}
                           for c in curves])
        for cond in CONDITIONS:
            sub = df[df.condition == cond]
            if sub.empty:
                continue
            per_p = np.stack([np.mean(np.stack(g["values"].to_list()), axis=0)
                              for _, g in sub.groupby("participant")])        # (participants, 24, sets)
            m = per_p.mean(axis=0)
            h = stats.t.ppf(0.975, max(len(per_p) - 1, 1)) * per_p.std(axis=0, ddof=1) / np.sqrt(len(per_p))
            for ax, s in zip(axes.ravel(), sets):
                i = set_index[s]
                t = np.arange(1, LAST_SECOND)
                ax.plot(t, m[:, i], color=colors[cond], label=cond)
                ax.fill_between(t, m[:, i] - h[:, i], m[:, i] + h[:, i], color=colors[cond], alpha=0.2)
        for ax, s in zip(axes.ravel(), sets):
            for w, (lo, hi) in [("induction", WINDOWS["induction"]), ("maintenance", WINDOWS["maintenance"])]:
                ax.axvspan(lo - 0.5, hi + 0.5, color="k", alpha=0.04 if w == "induction" else 0.09)
            if measure == "bold":
                ax.axhline(0, color="k", lw=0.5)
            elif measure == "enrichment":
                ax.axhline(1, color="k", lw=0.5)
            ax.set_title(s, fontsize=10)
        axes[0, 0].legend(fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("Time (s) task synchronized")
        fig.suptitle(f"{measure} per second (1-24 s; shaded: induction and maintenance windows)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"timecourse_{measure}.png"), dpi=150)
        plt.close(fig)
    # the 15 anatomical sets only (networks: tests.csv); curves hold every set
    return


def write_summary(tests, n_rois, out_dir, pop, n_participants):
    pre = tests[tests.family != "E"].copy()
    lines = [f"# Phase protocol - {POPULATION_LABELS.get(pop, pop)} (`{pop}`, {n_participants} participants)", "",
             "Windows (seconds of the figures): induction 6-15, maintenance 16-24, baseline 1-2; "
             "second 25 left out. FDR (Benjamini-Hochberg) within each family.", "",
             "Composition check: 'vs median set' is the estimate minus the median estimate of the 15 "
             "anatomical sets for the same measure, window and contrast. For amount and enrichment, a "
             "set-specific effect should be well above this median, not only above zero.", "",
             "| Family | Hypothesis | Measure | Window | Set (ROIs) | Contrast | Predicted | n | Estimate [95 % CI] | vs median set | p | q | Verdict |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in pre.iterrows():
        est = "" if pd.isna(r.estimate) else f"{r.estimate:.4g} [{r.ci_low:.3g}, {r.ci_high:.3g}]"
        p = "" if pd.isna(r.p) else f"{r.p:.3g}"
        q = "" if pd.isna(r.q_fdr) else f"{r.q_fdr:.3g}"
        vm = "" if pd.isna(r.vs_median_set) else f"{r.vs_median_set:.3g}"
        lines.append(f"| {r.family} | {r.hypothesis} | {r.measure} | {r.window} | {r.set} ({n_rois.get(r.set, '')}) "
                     f"| {r.contrast} | {r.alternative} | {r.n} | {est} | {vm} | {p} | {q} | {r.verdict} |")
    lines += ["", "Exploratory tests (family E: every set and network, data-driven windows 7-11 s and "
              "14-18 s included) are in tests.csv."]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pop", default="test", choices=["test", "val"],
                        help="test = RAW external validation set (default), val = FINA test set")
    parser.add_argument("--name", default="lstmv4vlt", help="predictions/<name>/ folder")
    parser.add_argument("--checkpoint", default=CHECKPOINT, help="only its file name is used (prediction files)")
    parser.add_argument("--shap_dir", default=SHAP_DIR)
    parser.add_argument("--clinical", default=None,
                        help="optional CSV (key = id1 for test, id2 for val; pswq_total, age, sex, depression) "
                             "instead of the clinical files of config.py")
    parser.add_argument("--out", default=None, help="output folder (default: results/protocol/<pop>/)")
    parser.add_argument("--no_figures", action="store_true")
    args = parser.parse_args()
    out_dir = args.out or os.path.join(REPO, "results", "protocol", args.pop)
    os.makedirs(out_dir, exist_ok=True)

    blocks, curves, set_names, n_rois = compute_block_values(args)
    means = participant_means(blocks)
    participants = sorted(blocks.participant.unique())
    scores = load_scores(args.pop, participants, args.clinical)
    tests = all_tests(means, blocks, scores, set_names)

    blocks.to_csv(os.path.join(out_dir, "block_values.csv.gz"), index=False)
    means.to_csv(os.path.join(out_dir, "participant_means.csv"), index=False)
    tests.to_csv(os.path.join(out_dir, "tests.csv"), index=False)
    write_summary(tests, n_rois, out_dir, args.pop, len(participants))
    if not args.no_figures:
        figures(means, curves, out_dir, list(ROI_SETS))
    pre = tests[tests.family != "E"]
    print(f"{len(participants)} participants, {blocks[['participant', 'block']].drop_duplicates().shape[0]} blocks; "
          f"{len(pre)} pre-specified tests: " + ", ".join(f"{v} {k}" for k, v in pre.verdict.value_counts().items()))
    print(f"Results in {out_dir}")


if __name__ == "__main__":
    main()
