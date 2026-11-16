# Predicting Worry Mental States using Long Short-Term Memory (LSTM) Recurrent Deep Neural Networks

Code of the paper:

> Campion J.Y., Desmidt T., Gross J.J., Tudorascu D.L., Andreescu C., Karim H.T.
> **Predicting Worry Mental States using Long Short-Term Memory (LSTM) Recurrent Deep Neural Networks.**
> *medRxiv* (2026). doi: [10.64898/2026.08.14.26360462](https://doi.org/10.64898/2026.08.14.26360462)

A bidirectional LSTM network reads regional fMRI time series (481 regions) recorded
during an in-scanner worry induction and reappraisal task, and predicts at each time
point whether the participant is in a **Worry**, **Neutral** or **Reappraisal** state.
SHAP attributions are then projected on 18 canonical brain networks to identify the
regions and networks that drive the predictions.

## Contents

- [Data](#data)
- [Installation](#installation)
- [Reproducing the results](#reproducing-the-results)
- [Re-running the analyses of the paper](#re-running-the-analyses-of-the-paper)
- [Repository structure](#repository-structure)
- [Model](#model)
- [Notes and limitations](#notes-and-limitations)
- [Citation](#citation)
- [License](#license)

## Data

Two cohorts of older adults performed the same worry induction and reappraisal task:

| Cohort | Study | N | Role in the paper | Name in the code |
|---|---|---|---|---|
| 1 | FINA | 116 | training set (80 % of participants, 92) and test set (20 %, 24 participants) | `train`, `val` |
| 2 | RAW | 88 | external validation set (20 participants were also enrolled in FINA, see `common_file.csv`) | `test` |

Names: the code keeps the identifiers `train`, `val` and `test` (file names, `step`, `--pop`);
every figure and table shows the names of the paper: **FINA: training set**, **FINA: test set**
(`val`) and **RAW: external validation set** (`test`) (`POPULATION_LABELS` in
`visualization/base.py`, factor labels in `R/Article.Rmd`). The FINA test set is held out
from training but was used for early stopping and to select the checkpoint (lowest L1 loss
on it, `main.py`): its metrics are therefore slightly optimistic, and the RAW external
validation set is the independent evaluation.

Both studies were approved by the University of Pittsburgh IRB. **The fMRI and clinical
data are not public**; they are available from the corresponding authors upon
reasonable request. The repository contains no participant data.

All paths are set in [`config.py`](config.py) and can be changed with environment
variables (`WORRY_*`, see below). The time series (to train the model or rebuild the window
caches):

```
$WORRY_DATA_ROOT/                                  (default: /mnt/argo/Studies)
├── FINA/Public/Analysis/
│   ├── data/<9xxxxx>/<8xxxxx>/step02_WorryInduction/schaeffer_subcortical_timeseries.csv
│   ├── data/<9xxxxx>/<8xxxxx>/converted/WorryInduction/behavioral_data/responses*.mat
│   └── misc/FINA_2023_01_06.csv                   clinical scores stored in the caches (PSWQ, RSQ, HARS)
└── RAW/Public/Analysis/
    ├── data/<1xxx>/<1xxx>/step05_worry/schaeffer_subcortical_timeseries.csv
    └── data/<1xxx>/<1xxx>/converted/WorryTask/responses*.mat
```

The inputs of the analyses of the paper, in one folder:

```
$WORRY_RECOVERY_ROOT/                              (default: /ix1/hkarim/users/jpc211/postdoc/recovery)
├── checkpoints/lstm_v4_host-smp-n217.crc.pitt.edu_version3/model_best_score=val_loss=0.1113.ckpt
├── shap/shap-lstmv4vf_pop-val_batch-{0..7}.npz, shap-lstmv4vf_pop-test_batch-{0..29}.npz
├── cached_data/cache_data-TASK_{FINA,RAW}_size-{20_sample-100,25_sample-34,1190_sample-1}_roi-wSubcortical_seed-123.pt
├── predictions/lstmv4vlt/model_best_score=val_loss=0.1113_sub-true_type-{training,block,full}_step-{train,val,test}_n-all.pt
├── clinical/FINA_2025_01_17.csv                   clinical variables of cohort 1
├── clinical/RAW_JY_Data_Request_2.10.25.csv       clinical scores of cohort 2
├── clinical/pheno_raw.csv                         education of cohort 2 (sample description)
└── common_file.csv                                raw_id <-> mr_studyid (participants of both studies)
```

Each `schaeffer_subcortical_timeseries.csv` has one row per volume and 484 columns:
the 400 cortical parcels of the Schaefer-2018 atlas (7 networks), 81 subcortical and
cerebellar regions of AAL3, then the 3 task regressors (Worry, Neutral, Reappraisal).
The task has 34 blocks of 35 s starting at 10 s (16 Worry, 10 Neutral, 8 Reappraisal).
Cohort 2 (TR = 0.7 s) is resampled to 1 s. The `responses*.mat` files contain the
in-scanner worry ratings (1-5).

The network templates used by the analyses are described in [`external/README.md`](external/README.md).

## Installation

```bash
git clone https://github.com/tetra-tools/Predicting-Worry-Mental-States.git
cd Predicting-Worry-Mental-States
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then, if your data are not on the original storage:

```bash
export WORRY_RECOVERY_ROOT=/path/to/recovery   # inputs of the analyses (layout above)
export WORRY_DATA_ROOT=/path/to/Studies        # time series
export WANDB_MODE=offline   # optional: train without a Weights & Biases account
```

Single files can also be moved with `WORRY_CHECKPOINT`, `WORRY_SHAP_DIR`, `WORRY_CACHE_DIR`,
`WORRY_PREDICTIONS_DIR`, `WORRY_COMMON_FILE`, `WORRY_FINA_CLINICAL_CSV`,
`WORRY_RAW_CLINICAL_CSV`, `WORRY_RAW_PHENO_CSV` and `WORRY_FINA_SCORES_CSV`;
`WORRY_RECOMPUTE_PREDICTIONS=1` recomputes missing predictions. Relative paths are
taken from the repository root.

## Reproducing the results

All commands are run from the repository root.

### 1. Train the model

```bash
python main.py --model lstm_v4
```

The defaults are the settings of the published model: windows of 20 TR, 100 random
windows per participant, batch size 32, AdamW (learning rate 5e-4, weight decay 0.2),
ReduceLROnPlateau, dropout 0.1, gradient clipping 0.8, input mixing decaying from 0.3 to
0.2, seed 123, up to 250 epochs with early stopping (patience 50) on the validation L1
loss. Training runs on CPU (20 threads, about 64 GB of RAM).
The best checkpoint is written to `checkpoints/<model>_host-<hostname>_version<n>/`;
the published model is `lstm_v4_host-smp-n217.crc.pitt.edu_version3/model_best_score=val_loss=0.1113.ckpt`.

The first run builds a cache of the windows in `$WORRY_CACHE_DIR` (default
`$WORRY_RECOVERY_ROOT/cached_data/`). Later runs (and the analysis code) reuse it, so keep
it to get exactly the same windows.

### 2. Compute SHAP values

Each run explains one batch of 102 task blocks (3 participants × 34 blocks):

```bash
CKPT=checkpoints/<run>/model_best_score=val_loss=<...>.ckpt   # default: config.CHECKPOINT
for b in $(seq 0 7);  do python main_shap.py --checkpoint $CKPT --population val  --batch_idx $b; done
for b in $(seq 0 29); do python main_shap.py --checkpoint $CKPT --population test --batch_idx $b; done
```

The files are written in the current directory, never in `$WORRY_SHAP_DIR`, so the
published SHAP values cannot be overwritten; move them there to analyse them.

Each batch needs more than 15 GB of RAM (the background distribution is the whole
training set). The study ran them as a SLURM job array, one batch per task.

### 3. Analyses and figures

Open the notebooks of `visualization/` with Jupyter (working directory: `visualization/`);
they read the checkpoint and the other inputs from `config.py`:

| Notebook | Content |
|---|---|
| `results_article.ipynb` | performance (MAE, AUC, sensitivity, specificity, PPV, NPV, LR+/LR-), AUC within the block, effect of the in-scanner ratings, SHAP network heatmaps, SHAP–clinical correlations, sample description |
| `create_R_data.ipynb` | exports metrics and SHAP summaries to `results/` for the R figures |

All the analysis code is in [`visualization/base.py`](visualization/base.py)
(`Visualization` class). Predictions are cached in `$WORRY_PREDICTIONS_DIR/<name>/` (a missing
file is recomputed only with `WORRY_RECOMPUTE_PREDICTIONS=1`);
heatmaps are written to `results/<name>/predictions/{validation,test}/heatmap/` (folder names
from the code: `validation` = FINA test set, `test` = RAW external validation set).

### 4. Statistical analyses (R)

[`R/Article.Rmd`](R/Article.Rmd) contains the mixed-effects models (block-wise MAE of
cohort 2; SMA network attributions), the per-participant AUCs, the timing of the SHAP
attributions and the network figures. Knit it from the `R/` folder: it reads the
per-time-point exports `R/data/export_{train,val,test}_{training,block,full}.csv` and
`R/data/export_{val,test}_block_with_shap.csv` (labels, predictions, ratings, clinical
scores and SHAP values per ROI and per network) written by `export_R_data.py`, and writes
the figures to `R/plots/`.

Network attributions ([`utils/network.py`](utils/network.py)):

- **Normalisation.** The SHAP map of each block and condition (25 time points x 481 ROIs)
  is L2-normalised over time *and* ROIs, then projected on each L2-normalised template along
  the ROI axis. One normalisation per block keeps how much each second contributes to the
  prediction explained (last time point of the block); the previous normalisation of each
  time point separately (`--normalization timepoint` of `export_R_data.py`) gave every second
  the same weight.
- **Amount and rank.** `prepare_shap_network_data` returns the absolute projection
  (`shap_value`, the amount: the curves) and its min-max across the networks within each
  block and second (`shap_rank`: the selection and the order of the networks in the figures,
  and the SMA model). Scaling block by block makes the rank independent of the number of
  blocks of a condition (16 Worry, 10 Neutral, 8 Reappraisal per participant); pooling the
  blocks of a participant had compressed the Worry values.
- **Confidence intervals.** The curves average the blocks of each participant first, so the
  95 % confidence intervals are computed across participants, not across blocks.

### 5. Pre-specified phase protocol (replication of Karim et al. 2017)

[`PROTOCOL.md`](PROTOCOL.md) fixes, before the analysis, the windows (induction 6-15 s,
maintenance 16-24 s, from the task timing), 15 anatomical ROI sets and the hypotheses of the
2017 CBF study (initiation, maintenance and habituation, direction of the reappraisal effects,
worry severity). [`phase_protocol.py`](phase_protocol.py) tests them on the attributions
(enrichment, direction of the attributed activity) and on the BOLD inputs stored in the block
prediction files (no window cache needed), with participant-level tests and FDR per family:

```bash
python phase_protocol.py --pop test    # writes results/protocol/test/ (summary.md, tests.csv, figures)
```

### 6. Brain maps of the networks (Figure 4)

The right part of Figure 4 shows the templates of the selected networks (those of the
curves), not SHAP values. `tools/make_network_maps.py` writes each template as a volume
(`<network>.nii`), a BrainNet Viewer node file (`<network>.node`) and a preview, plus a
cerebellum surface (`cerebellum.nv`, BrainNet's cortical meshes have none):

```bash
python tools/make_network_maps.py --aal-dir /path/to/AAL3 --networks Cerebellum DAN
```

`--aal-dir` contains `AAL3v1.nii` (2 mm) and `AAL3v1_1mm.xml`; AAL3 regions are painted with
their index from the XML (the indices are not consecutive). In BrainNet Viewer, map
`<network>.nii` on `BrainMesh_ICBM152_smoothed.nv` and on `cerebellum.nv` with the same
options file, e.g. `BrainNet_MapCfg('cerebellum.nv', 'Cerebellum.nii', 'options.mat', 'out.png')`.

## Re-running the analyses of the paper

`rerun_paper_analyses.sh` regenerates the exports and the figures of the R analyses from
the cached predictions and the SHAP values already computed. Neither the model, the SHAP
values nor the predictions are recomputed, and no window cache is needed: an analysis whose
prediction file is missing is left out (listed at the start of the run).

```bash
bash rerun_paper_analyses.sh                 # checks, exports (R/data/), R analyses (R/plots/)
RUN_NOTEBOOK=1 bash rerun_paper_analyses.sh  # also executes visualization/results_article.ipynb
```

Files to provide (folders relative to `$WORRY_RECOVERY_ROOT`, see [Data](#data)):

| File | Location | Used for |
|---|---|---|
| Published checkpoint `model_best_score=val_loss=0.1113.ckpt` | `checkpoints/lstm_v4_host-smp-n217.crc.pitt.edu_version3/` | loading the model |
| 38 SHAP files `shap-lstmv4vf_pop-val_batch-{0..7}.npz` and `shap-lstmv4vf_pop-test_batch-{0..29}.npz` | `shap/` | SHAP columns of the exports |
| 3 block predictions `model_best_score=val_loss=0.1113_sub-true_type-block_step-{train,val,test}_n-all.pt` | `predictions/lstmv4vlt/` | Figures 1-5, Supplementary Figures 2-4 |
| `common_file.csv` | `./` | participants enrolled in both cohorts |
| `FINA_2025_01_17.csv` | `clinical/` | clinical variables of cohort 1 |
| `RAW_JY_Data_Request_2.10.25.csv` | `clinical/` | clinical variables of cohort 2 |
| *For `RUN_NOTEBOOK=1`:* `pheno_raw.csv` | `clinical/` | education of cohort 2 (sample description) |
| *Optional:* the 6 other predictions `..._type-{training,full}_step-{train,val,test}_n-all.pt` | `predictions/lstmv4vlt/` | 20 s and full-run windows (Supplementary Figure 1, AUC tables); with the `val` files only, these windows are compared on the FINA test set (`figure_auc_val_windows.png`, `figure_mae_overall_val_windows.png`) |
| *Optional:* previous exports `export_*.csv` | `R/data/` of the repository (moved to `R/data_previous/` and checked) | `tools/check_network_projection.py`, `R/check_auc_direction.R` |

The script stops if the checkpoint, `common_file.csv`, a clinical file, one of the 38 SHAP
files or one of the 3 block predictions is missing.

A prediction file can be recomputed with `WORRY_RECOMPUTE_PREDICTIONS=1` (window caches in
`cached_data/`, or the time series to rebuild them), but only the original caches give the
windows of the paper: the 100 random 20 s windows per participant are drawn when the cache
is built, and the train/val split follows the order of the participants in the cache. `export_R_data.py` writes the exports (`R/data/export_*.csv`, network attributions
projected along the ROI axis) and `R/Article.Rmd` writes the figures to `R/plots/`.

## Repository structure

```
├── main.py                  training
├── main_shap.py             SHAP attributions (DeepExplainer)
├── export_R_data.py         per-time-point exports read by R/Article.Rmd
├── rerun_paper_analyses.sh  exports + R analyses from the trained model and SHAP values
├── config.py                data paths (environment variables)
├── models/lstm_v4.py        LSTM classifier (PyTorch Lightning)
├── datasets/
│   ├── TimeSeriesDataset.py     windows, labels, ratings and clinical scores (+ cache)
│   └── TimeSeriesDataModule.py  subject-level train/validation split, test cohort
├── visualization/
│   ├── base.py              metrics and heatmaps
│   └── *.ipynb              analyses of the paper
├── R/Article.Rmd            mixed-effects models, per-participant AUCs, SHAP figures
├── R/check_auc_direction.R  effect of pROC's automatic direction on previous exports
├── tools/check_network_projection.py  which projection produced the network columns of an export
├── phase_protocol.py        pre-specified phase protocol (PROTOCOL.md): second-phase contrast, direction
├── tools/check_shap_additivity.py   do the SHAP values add up to the predictions (no cache needed)
├── tools/compare_explainers.py     DeepExplainer vs GradientExplainer on a few participants
├── tools/make_network_maps.py  network templates as brain maps for BrainNet Viewer (Figure 4)
├── utils/                   NaN check, network projection of the SHAP values (network.py)
└── external/                network templates
```

## Model

| Stage | Layers |
|---|---|
| Input | 481 ROIs → Linear 512, ReLU, Dropout, LayerNorm → Linear 256, ReLU |
| Recurrent | 2 residual blocks: bidirectional LSTM → Linear, ReLU, Dropout, LayerNorm (256 → 256 → 128) |
| Output | Linear 64, ReLU, Dropout, LayerNorm → Linear 3 → (tanh + 1) / 2 |

The targets are the three task regressors rescaled to [0, 1] and the loss is the mean
absolute error, at every time point. During training each window is mixed with another
window of the same batch, `x ← (x + λ·x_perm) / (1 + λ)`, with λ = 0.3 for two epochs,
then decreasing linearly to 0.2 over 200 epochs.

For evaluation, the predicted state at each time point is the condition with the highest
output. AUCs are computed one-vs-rest and one-vs-one by pooling all time points of a
population, for three window configurations: the 34 task blocks (25 s from each block
onset), the whole run, and the random 20 s training windows.

## Notes and limitations

- Training was run on CPU; `--use_gpu` only lowers the number of CPU threads.
- `shap.DeepExplainer` has no specific rule for `LSTM` and `LayerNorm` modules (it warns
  about them and `check_additivity` is disabled); attributions explain the prediction at
  the last time point of each 25 s block window. Two checks need neither the window caches
  nor a new SHAP run: `tools/check_shap_additivity.py` (SHAP files and block predictions;
  run by `rerun_paper_analyses.sh`) tests whether the attributions add up to the
  predictions, and `tools/compare_explainers.py` (also the checkpoint) recomputes the
  attributions of a few participants with `shap.GradientExplainer` from the inputs stored
  in the block prediction files and compares the ROI maps and network orders.
- The last second of the block window (25 s) is left out of the SHAP figures, models and
  protocol: it is the input of the explained output and is large by construction.
- The order of the networks partly depends on the spatial extent of their templates: on a
  first set of curves, compact templates (SMA, auditory, visual) ranked higher than
  extended ones (right executive control, anterior salience), Spearman rho about -0.5.
  Compare a network across conditions or seconds rather than networks with each other.
- The identifiers of the participants are parsed from the file paths, so
  `WORRY_DATA_ROOT` must not contain runs of digits such as `9xxxxx` or `1xxx`.

## Citation

```bibtex
@article{campion2026worry,
  title   = {Predicting Worry Mental States using Long Short-Term Memory (LSTM) Recurrent Deep Neural Networks},
  author  = {Campion, Jacques-Yves and Desmidt, Thomas and Gross, James J. and Tudorascu, Dana L. and Andreescu, Carmen and Karim, Helmet T.},
  journal = {medRxiv},
  year    = {2026},
  doi     = {10.64898/2026.08.14.26360462}
}
```

See also [`CITATION.cff`](CITATION.cff).

## License

The code is released under the [MIT License](LICENSE). The atlases and network templates
of `external/` remain under the terms of their original sources (Schaefer-2018, AAL3,
Smith et al.). The data of the two cohorts and the trained model are not part of this
repository and are not covered by this license.
