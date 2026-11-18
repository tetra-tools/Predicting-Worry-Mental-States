#!/usr/bin/env Rscript
# Re-make plots/figure_timing.png alone (chunk `shap_temporal` of Article.Rmd) without rendering
# the whole article. The code is read from Article.Rmd itself, so the figure stays identical to
# the one of the article.
#
# Needs the SHAP exports of a previous run of rerun_paper_analyses.sh:
#   data/export_val_block_with_shap.csv, data/export_test_block_with_shap.csv
# Run from R/:
#   Rscript figure_timing.R

rmd <- readLines("Article.Rmd")
fences <- grep("^```\\s*$", rmd)

chunk_code <- function(label) {
  start <- grep(sprintf("^```\\{r %s[,} ]", label), rmd)
  if (length(start) != 1) stop("Chunk '", label, "' not found once in Article.Rmd")
  end <- min(fences[fences > start])
  rmd[(start + 1):(end - 1)]
}
run_chunk <- function(label) eval(parse(text = chunk_code(label)), envir = globalenv())

run_chunk("setup")      # packages, plots/
run_chunk("functions")  # read_export, bind_exports, find_local_maxima, last_timepoint_excluded

# SHAP part of the chunk data_loading (the prediction exports are not needed for this figure)
df_block_withshap <- bind_exports(list(read_export("./data/export_val_block_with_shap.csv"),
                                       read_export("./data/export_test_block_with_shap.csv")))

run_chunk("shap_temporal")  # writes plots/figure_timing.png
message("Written: ", normalizePath("plots/figure_timing.png"))
