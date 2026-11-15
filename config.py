"""Central location of the data paths used across the project.

Two groups of paths:

* the inputs of the paper's analyses (checkpoint, SHAP values, window caches, cached
  predictions, clinical files, common_file.csv), gathered in one recovery folder
  (``WORRY_RECOVERY_ROOT``, layout below);
* the time series of the two cohorts (``WORRY_DATA_ROOT``), only read to rebuild a missing
  window cache or ratings.

Every path can be changed with the environment variable named next to it, e.g.::

    export WORRY_RECOVERY_ROOT=/data/worry_recovery
    export WORRY_DATA_ROOT=/data/Studies

Layout of the recovery folder::

    <WORRY_RECOVERY_ROOT>/
    ├── checkpoints/lstm_v4_host-smp-n217.crc.pitt.edu_version3/model_best_score=val_loss=0.1113.ckpt
    ├── shap/shap-lstmv4vf_pop-{val_batch-0..7,test_batch-0..29}.npz
    ├── cached_data/cache_data-TASK_{FINA,RAW}_size-*_sample-*_roi-wSubcortical_seed-123.pt
    ├── predictions/model_best_score=val_loss=0.1113_sub-true_type-*_step-*_n-all.pt
    ├── clinical/FINA_2025_01_17.csv, RAW_JY_Data_Request_2.10.25.csv, pheno_raw.csv
    └── common_file.csv
"""
import os

RECOVERY_ROOT = os.environ.get("WORRY_RECOVERY_ROOT", "/ix1/hkarim/users/jpc211/postdoc/recovery")

# Published model (main.py).
CHECKPOINT = os.environ.get(
    "WORRY_CHECKPOINT",
    os.path.join(RECOVERY_ROOT, "checkpoints", "lstm_v4_host-smp-n217.crc.pitt.edu_version3",
                 "model_best_score=val_loss=0.1113.ckpt"),
)

# SHAP files of main_shap.py (38 .npz), read by visualization/base.py.
SHAP_DIR = os.environ.get("WORRY_SHAP_DIR", os.path.join(RECOVERY_ROOT, "shap"))

# Window caches of TimeSeriesDataset (rebuilt from the time series if missing).
CACHE_DIR = os.environ.get("WORRY_CACHE_DIR", os.path.join(RECOVERY_ROOT, "cached_data"))

# Predictions cached by visualization/base.py (recomputed from the checkpoint if missing).
PREDICTIONS_DIR = os.environ.get("WORRY_PREDICTIONS_DIR", os.path.join(RECOVERY_ROOT, "predictions"))

# Participants enrolled in both cohorts (raw_id <-> mr_studyid).
COMMON_FILE = os.environ.get("WORRY_COMMON_FILE", os.path.join(RECOVERY_ROOT, "common_file.csv"))

# Clinical variables of cohort 1 (export_R_data.py, results_article.ipynb).
FINA_CLINICAL_CSV = os.environ.get(
    "WORRY_FINA_CLINICAL_CSV", os.path.join(RECOVERY_ROOT, "clinical", "FINA_2025_01_17.csv"))

# Clinical scores of cohort 2 (PSWQ, RSQ, HARS, age, sex).
RAW_CLINICAL_CSV = os.environ.get(
    "WORRY_RAW_CLINICAL_CSV", os.path.join(RECOVERY_ROOT, "clinical", "RAW_JY_Data_Request_2.10.25.csv"))

# Education of cohort 2 (sample description of results_article.ipynb).
RAW_PHENO_CSV = os.environ.get(
    "WORRY_RAW_PHENO_CSV", os.path.join(RECOVERY_ROOT, "clinical", "pheno_raw.csv"))

# Root folder containing the time series of the two cohorts (only to rebuild a cache):
#   <WORRY_DATA_ROOT>/FINA/Public/Analysis/data/9xxxxx/8xxxxx/step02_WorryInduction/  (cohort 1)
#   <WORRY_DATA_ROOT>/RAW/Public/Analysis/data/1xxx/1xxx/step05_worry/                (cohort 2)
DATA_ROOT = os.environ.get("WORRY_DATA_ROOT", "/mnt/argo/Studies")

# Clinical scores of cohort 1 stored in the window caches (only to rebuild a cache).
FINA_SCORES_CSV = os.environ.get(
    "WORRY_FINA_SCORES_CSV", os.path.join(DATA_ROOT, "FINA", "Public", "Analysis", "misc", "FINA_2023_01_06.csv"))
