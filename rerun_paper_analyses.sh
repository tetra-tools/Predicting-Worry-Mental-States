#!/usr/bin/env bash
# Re-run the analyses of the paper affected by the pre-submission review
# (per-participant AUC direction, network projection of the SHAP values, SMA model scale).
#
# Needs the files listed in readme.md, "Re-running the analyses of the paper".
# Neither the model nor the SHAP values are recomputed.
#
# Usage (from the repository root):
#   bash rerun_paper_analyses.sh                 # checks + exports + R analyses
#   RUN_NOTEBOOK=1 bash rerun_paper_analyses.sh  # also executes visualization/results_article.ipynb
set -euo pipefail
cd "$(dirname "$0")"

# Input paths: config.py (WORRY_RECOVERY_ROOT and the other WORRY_* variables)
cfg() { python -c "import config; print(config.$1)"; }
count() { local n=0 f; for f in "$@"; do [ -e "$f" ] && n=$((n + 1)); done; echo "$n"; }   # 0 if no match
CKPT=${CKPT:-$(cfg CHECKPOINT)}
SHAP_DIR=$(cfg SHAP_DIR)
COMMON_FILE=$(cfg COMMON_FILE)
NAME=${NAME:-lstmv4vlt}
TEMPLATES=external/schaeffer_subcortical_smith_templates.csv

echo "== 0. Inputs"
for f in "$CKPT" "$COMMON_FILE" "$(cfg FINA_CLINICAL_CSV)" "$(cfg RAW_CLINICAL_CSV)"; do
  test -f "$f" || { echo "Not found: $f"; exit 1; }
done
n_shap=$(count "$SHAP_DIR"/shap-lstmv4vf_pop-*_batch-*.npz)
[ "$n_shap" -eq 38 ] || { echo "Expected 38 SHAP files in $SHAP_DIR, found $n_shap"; exit 1; }
n_cache=$(count "$(cfg CACHE_DIR)"/cache_data-TASK_*_roi-wSubcortical_seed-123.pt)
n_pred=$(count "$(cfg PREDICTIONS_DIR)/$NAME"/*_n-all.pt)
echo "checkpoint, common_file.csv, clinical files and 38 SHAP files found"
echo "$n_cache/6 window caches (missing ones are rebuilt from WORRY_DATA_ROOT), $n_pred/9 cached predictions (missing ones are recomputed)"

echo "== 1. Checks of the previous exports (R/data_previous/), if present"
if [ -d R/data ] && [ ! -d R/data_previous ]; then
  mv R/data R/data_previous && echo "previous exports moved to R/data_previous/"
fi
if [ -f R/data_previous/export_test_block_with_shap.csv ]; then
  python tools/check_network_projection.py R/data_previous/export_test_block_with_shap.csv "$TEMPLATES" \
    | tee R/data_previous/check_network_projection.txt
fi
if [ -f R/data_previous/export_test_block.csv ]; then
  (cd R && Rscript check_auc_direction.R data_previous) | tee R/data_previous/check_auc_direction.txt
fi

echo "== 2. Exports for the R analyses (R/data/)"
python export_R_data.py --checkpoint "$CKPT" --name "$NAME" --out R/data
python tools/check_network_projection.py R/data/export_test_block_with_shap.csv "$TEMPLATES"

echo "== 3. R analyses (figures in R/plots/, report R/Article.html)"
(cd R && Rscript -e 'rmarkdown::render("Article.Rmd")')

if [ "${RUN_NOTEBOOK:-0}" = "1" ]; then
  echo "== 4. visualization/results_article.ipynb (AUC per second of the block, heatmaps)"
  (cd visualization && jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
     results_article.ipynb --output results_article.executed.ipynb)
fi
echo "Done."
