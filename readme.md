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

## Data

Two cohorts of older adults performed the same worry induction and reappraisal task:

| Cohort | Study | N | Role in the paper | Name in the code |
|---|---|---|---|---|
| 1 | FINA | 116 | training (80 % of participants) and held-out validation (20 %, 24 participants) | `train`, `val` |
| 2 | RAW | 88 | independent test cohort, never seen during training | `test` |

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
`WORRY_RAW_CLINICAL_CSV`, `WORRY_RAW_PHENO_CSV` and `WORRY_FINA_SCORES_CSV`. Relative paths are
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
(`Visualization` class). Predictions are cached in `$WORRY_PREDICTIONS_DIR/<name>/`;
heatmaps are written to `results/<name>/predictions/{validation,test}/heatmap/`.

### 4. Statistical analyses (R)

[`R/Article.Rmd`](R/Article.Rmd) contains the mixed-effects models (block-wise MAE of
cohort 2; SMA network attributions), the per-participant AUCs, the timing of the SHAP
attributions and the network figures. Knit it from the `R/` folder: it reads the
per-time-point exports `R/data/export_{train,val,test}_{training,block,full}.csv` and
`R/data/export_{val,test}_block_with_shap.csv` (labels, predictions, ratings, clinical
scores and SHAP values per ROI and per network) written by `export_R_data.py`, and writes
the figures to `R/plots/`.

### 5. Brain maps of the networks (Figure 4)

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

`rerun_paper_analyses.sh` regenerates the exports and all the figures of the R analyses
from the trained model and the SHAP values already computed (neither is recomputed):

```bash
bash rerun_paper_analyses.sh                 # checks, exports (R/data/), R analyses (R/plots/)
RUN_NOTEBOOK=1 bash rerun_paper_analyses.sh  # also executes visualization/results_article.ipynb
```

Files to provide (folders relative to `$WORRY_RECOVERY_ROOT`, see [Data](#data)):

| File | Location | Used for |
|---|---|---|
| Published checkpoint `model_best_score=val_loss=0.1113.ckpt` | `checkpoints/lstm_v4_host-smp-n217.crc.pitt.edu_version3/` | predictions |
| 38 SHAP files `shap-lstmv4vf_pop-val_batch-{0..7}.npz` and `shap-lstmv4vf_pop-test_batch-{0..29}.npz` | `shap/` | SHAP columns of the exports |
| 6 window caches `cache_data-TASK_{FINA,RAW}_size-{20_sample-100,25_sample-34,1190_sample-1}_roi-wSubcortical_seed-123.pt` | `cached_data/` | the same windows as in the paper (the 100 random windows per participant can only be reproduced from them) |
| `common_file.csv` | `./` | participants enrolled in both cohorts |
| `FINA_2025_01_17.csv` | `clinical/` | clinical variables of cohort 1 |
| `RAW_JY_Data_Request_2.10.25.csv` | `clinical/` | clinical variables of cohort 2 |
| *For `RUN_NOTEBOOK=1`:* `pheno_raw.csv` | `clinical/` | education of cohort 2 (sample description) |
| *Optional:* 9 cached predictions `model_best_score=val_loss=0.1113_sub-true_type-{training,block,full}_step-{train,val,test}_n-all.pt` | `predictions/lstmv4vlt/` | faster (recomputed from the checkpoint otherwise) |
| *Optional:* previous exports `export_*.csv` | `R/data/` of the repository (moved to `R/data_previous/` and checked) | `tools/check_network_projection.py`, `R/check_auc_direction.R` |
| *Only if a cache is missing:* time series, `responses*.mat`, `FINA_2023_01_06.csv` | `$WORRY_DATA_ROOT` | rebuilding the cache |

The script stops if the checkpoint, `common_file.csv`, a clinical file or one of the 38
SHAP files is missing. `export_R_data.py` writes the exports (`R/data/export_*.csv`, network attributions
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
├── tools/make_network_maps.py  network templates as brain maps for BrainNet Viewer (Figure 4)
├── utils/                   NaN check
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
  the last time point of each 25 s block window.
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
