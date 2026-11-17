#!/usr/bin/env bash
# Re-run the analyses of the paper affected by the pre-submission review
# (per-participant AUC direction, network projection of the SHAP values, SMA model scale,
# block-level normalisation of the SHAP maps, rank vs amount of the network attributions,
# participant-level confidence intervals, additivity of the SHAP values).
#
# Needs the files listed in readme.md, "Re-running the analyses of the paper".
# Neither the model, the SHAP values nor the predictions are recomputed: every result comes
# from the cached predictions, and the analyses whose predictions are missing are left out.
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
echo "checkpoint, common_file.csv, clinical files and 38 SHAP files found"
# Axis order of the SHAP files (shap < 0.45 stored the conditions first; utils/shap_io.py reorders them)
python -c "import sys; sys.path.insert(0, '.'); from utils.shap_io import layout; print('SHAP layout:', layout(sys.argv[1]))" \
  "$SHAP_DIR/shap-lstmv4vf_pop-test_batch-0.npz"
# ROI order of the time series vs the templates (needs the time series under WORRY_DATA_ROOT)
python tools/check_roi_order.py || echo "WARNING: check the ROI order above before reading the network results"

# Predictions: never recomputed here (without the original window caches, the windows and the
# train/val split would differ from the paper)
export WORRY_RECOMPUTE_PREDICTIONS=0
PRED_DIR="$(cfg PREDICTIONS_DIR)/$NAME"
pred() { echo "$PRED_DIR/$(basename "$CKPT" .ckpt)_sub-true_type-$1_step-$2_n-all.pt"; }   # as Visualization._get_predictions_filename
for s in train val test; do   # task blocks: Figures 1-5, Supplementary Figures 2-4
  test -f "$(pred block $s)" || { echo "Required: $(pred block $s)"; exit 1; }
done
for t in training full; do for s in train val test; do
  test -f "$(pred $t $s)" && echo "available: $t / $s" || echo "not available (analyses left out): $t / $s"
done; done

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
# Additivity of the SHAP values (sum of the attributions vs prediction - background mean)
for pop in val test; do
  python tools/check_shap_additivity.py --pop "$pop" --name "$NAME" --checkpoint "$CKPT" \
    --out "R/data/check_shap_additivity_$pop.csv" || echo "additivity check failed for $pop (left out)"
done

echo "== 2b. Pre-specified phase protocol (PROTOCOL.md; results/protocol/<pop>/)"
for pop in test val; do
  python phase_protocol.py --pop "$pop" --name "$NAME" --checkpoint "$CKPT" || echo "phase protocol failed for $pop (left out)"
done

echo "== 3. R analyses (figures in R/plots/, report R/Article.html)"
(cd R && Rscript -e 'rmarkdown::render("Article.Rmd")')

if [ "${RUN_NOTEBOOK:-0}" = "1" ]; then
  echo "== 4. visualization/results_article.ipynb (AUC per second of the block, heatmaps)"
  (cd visualization && jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
     results_article.ipynb --output results_article.executed.ipynb)
fi
echo "Done."
