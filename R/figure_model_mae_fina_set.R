#!/usr/bin/env Rscript
# Re-make plots/figure_model_mae_fina_set.png alone (chunk `mae_model_fina_set` of Article.Rmd),
# without rendering the whole article. The code is read from Article.Rmd itself.
#
# Needs only the block exports of a previous run of rerun_paper_analyses.sh:
#   data/export_train_block.csv, data/export_val_block.csv, data/export_test_block.csv
# (no SHAP export, no full or training exports: a few GB of memory are enough).
# Run from R/:
#   Rscript figure_model_mae_fina_set.R

rmd <- readLines("Article.Rmd")
fences <- grep("^```\\s*$", rmd)

chunk_code <- function(label) {
  start <- grep(sprintf("^```\\{r %s[,} ]", label), rmd)
  if (length(start) != 1) stop("Chunk '", label, "' not found once in Article.Rmd")
  end <- min(fences[fences > start])
  rmd[(start + 1):(end - 1)]
}
run_code <- function(code) eval(parse(text = code), envir = globalenv())

run_code(chunk_code("setup"))      # packages, plots/
run_code(chunk_code("functions"))  # read_export, bind_exports, load_and_combine, prepare_plot_vars

# Block part of the chunk data_loading (FINA training/test sets and RAW)
df <- prepare_plot_vars(load_and_combine("./data/export", "block"))

# Chunk lmer_analysis up to the fit of the Enrollment model: mae_data and model_formula
# (the Enrollment model and figure_model_mae.png are not redone)
lmer_code <- chunk_code("lmer_analysis")
run_code(lmer_code[seq_len(grep("^model <- lmer\\(model_formula", lmer_code)[1] - 1)])

run_code(chunk_code("mae_model_fina_set"))  # prints the counts, writes the figure
print(summary(model_fina_set))
print(anova(model_fina_set))
message("Written: ", normalizePath("plots/figure_model_mae_fina_set.png"))
