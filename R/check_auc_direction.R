# Effect of pROC's direction = "auto" on the per-participant AUCs of R/Article.Rmd
#
# Run from the R/ folder:
#   Rscript check_auc_direction.R [data folder, default: data]
# Uses the "25s task synchronized" windows (export_*_block.csv), as in figure_auc.png (Figure 1).
suppressPackageStartupMessages({library(readr); library(dplyr); library(tidyr); library(pROC)})
args <- commandArgs(TRUE)
data_dir <- if (length(args) > 0) args[1] else "data"

# Only the columns used here, identifiers as text (their type may differ between files)
df <- bind_rows(lapply(c("val", "test", "train"), function(s)
  read_csv(file.path(data_dir, paste0("export_", s, "_block.csv")), show_col_types = FALSE,
           col_select = c(split, id1, id2, starts_with("label_"), starts_with("pred_")),
           col_types = cols(split = col_character(), id1 = col_character(), id2 = col_character(),
                            .default = col_double())))) %>%
  mutate(
    true_class = case_when(
      pmax(label_worry, label_neutral, label_reappraisal) == label_worry ~ "worry",
      pmax(label_worry, label_neutral, label_reappraisal) == label_neutral ~ "neutral",
      pmax(label_worry, label_neutral, label_reappraisal) == label_reappraisal ~ "reappraisal"),
    Population = factor(split, levels = c("train", "val", "test"),
                        labels = c("FINA: training set", "FINA: test set", "RAW: external validation set")))

# AUC as computed in the article (auto) and with the direction fixed a priori
two_aucs <- function(response, predictor) {
  out <- tryCatch({
    r <- roc(response, predictor, quiet = TRUE)                       # as in the article
    f <- auc(response, predictor, levels = c(FALSE, TRUE), direction = "<", quiet = TRUE)
    c(auto = as.numeric(r$auc), fixed = as.numeric(f), flipped = r$direction == ">")
  }, error = function(e) c(auto = NA, fixed = NA, flipped = NA))
  out
}
comparisons <- list(
  "Worry vs Rest"          = list(pos = "worry",   keep = c("worry", "neutral", "reappraisal"), pred = "pred_worry"),
  "Neutral vs Rest"        = list(pos = "neutral", keep = c("worry", "neutral", "reappraisal"), pred = "pred_neutral"),
  "Reappraisal vs Rest"    = list(pos = "reappraisal", keep = c("worry", "neutral", "reappraisal"), pred = "pred_reappraisal"),
  "Worry vs Neutral"       = list(pos = "worry",   keep = c("worry", "neutral"), pred = "pred_worry"),
  "Worry vs Reappraisal"   = list(pos = "worry",   keep = c("worry", "reappraisal"), pred = "pred_worry"),
  "Neutral vs Reappraisal" = list(pos = "neutral", keep = c("neutral", "reappraisal"), pred = "pred_neutral"))

per_subject <- bind_rows(lapply(names(comparisons), function(name) {
  cmp <- comparisons[[name]]
  df %>% filter(true_class %in% cmp$keep) %>%
    group_by(Population, id1, id2) %>%
    summarise(res = list(two_aucs(true_class == cmp$pos, .data[[cmp$pred]])), .groups = "drop") %>%
    unnest_wider(res) %>% mutate(Condition = name)
}))

summary_table <- per_subject %>%
  group_by(Population, Condition) %>%
  summarise(n = n(),
            n_flipped = sum(flipped == 1, na.rm = TRUE),
            mean_auc_article = mean(auto, na.rm = TRUE),
            mean_auc_fixed = mean(fixed, na.rm = TRUE),
            difference = mean_auc_article - mean_auc_fixed,
            .groups = "drop")

options(width = 160)
print(as.data.frame(summary_table), digits = 3, row.names = FALSE)
cat("\nParticipants whose AUC was inverted (article value = 1 - correct value):\n")
print(as.data.frame(per_subject %>% filter(flipped == 1) %>%
                      select(Population, Condition, id1, id2, article = auto, correct = fixed)), digits = 3, row.names = FALSE)
out <- file.path(data_dir, "auc_direction_check.csv")
write_csv(per_subject, out)
cat("\nPer-participant values written to", out, "\n")
