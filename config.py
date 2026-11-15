"""Central location of the data paths used across the project.

The defaults are the paths of the University of Pittsburgh storage on which the
study was run, so the code behaves exactly as in the paper when no environment
variable is set. To run the code elsewhere, export the variables below instead
of editing the source files, e.g.::

    export WORRY_DATA_ROOT=/data/Studies
    export WORRY_RAW_CLINICAL_CSV=/data/RAW_clinical.csv
"""
import os

# Root folder containing the two cohorts:
#   <WORRY_DATA_ROOT>/FINA/Public/Analysis/data/9xxxxx/8xxxxx/step02_WorryInduction/  (cohort 1)
#   <WORRY_DATA_ROOT>/RAW/Public/Analysis/data/1xxx/1xxx/step05_worry/                (cohort 2)
DATA_ROOT = os.environ.get("WORRY_DATA_ROOT", "/mnt/argo/Studies")

# Clinical scores (PSWQ, RSQ, HARS) of cohort 2 (RAW).
RAW_CLINICAL_CSV = os.environ.get(
    "WORRY_RAW_CLINICAL_CSV",
    os.path.join("/mnt/argo/Workspaces", "Staff", "Jacques-Yves_Campion", "Public", "WorryInduction",
                 "RAW_JY_Data_Request_2.10.25.csv"),
)

# Clinical variables of cohort 1 used by export_R_data.py (PSWQ, RSQ, HARS, age, sex, ...).
FINA_CLINICAL_CSV = os.environ.get(
    "WORRY_FINA_CLINICAL_CSV",
    os.path.join(DATA_ROOT, "FINA", "Public", "Analysis", "misc", "FINA_2025_01_17.csv"),
)
