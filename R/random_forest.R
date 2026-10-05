# ============================================================
# Baseline Random Forest model
# ============================================================

library(DBI)
library(duckdb)
library(dplyr)
library(randomForest)
library(caret)
library(pROC)
library(PRROC)
library(ggplot2)

source("config.R")

# Create output folders if needed
dir.create(
  "output",
  showWarnings = FALSE,
  recursive = TRUE
)

dir.create(
  "output/plots",
  showWarnings = FALSE,
  recursive = TRUE
)


# ============================================================
# 1. LOAD MODEL DATASET
# ============================================================

model_data <- dbReadTable(
  con,
  "model_dataset"
)

cat("\nModel dataset loaded.\n")
cat("Rows:", nrow(model_data), "\n")


# ============================================================
# 2. DEFINE FEATURES
# ============================================================

feature_cols <- c(
  "equity_ratio",
  "tier1_leverage",
  "npl_ratio",
  "reserve_ratio",
  "re_loan_share",
  "construction_share",
  "bank_age",
  "roa",
  "nim",
  "loans_to_assets",
  "loans_to_deposits",
  "brokered_share",
  "uninsured_share",
  "funding_cost",
  "securities_share",
  "log_assets"
)


# ============================================================
# 3. CONVERT TARGET TO FACTOR
# ============================================================

model_data <- model_data %>%
  mutate(
    failed_12m = factor(
      failed_12m,
      levels = c(0, 1)
    )
  )


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

# Feature engineering already created:
#
# sample = "train"
# sample = "test"


train_data <- model_data %>%
  filter(sample == "train") %>%
  select(
    failed_12m,
    all_of(feature_cols)
  )

test_data <- model_data %>%
  filter(sample == "test") %>%
  select(
    failed_12m,
    all_of(feature_cols)
  )


cat("\nTraining observations:", nrow(train_data), "\n")
cat("Test observations:", nrow(test_data), "\n")


# ============================================================
# 5. CHECK CLASS IMBALANCE
# ============================================================

cat("\nTRAINING CLASS COUNTS\n")

print(
  table(train_data$failed_12m)
)

cat("\nTRAINING CLASS PROPORTIONS\n")

print(
  prop.table(
    table(train_data$failed_12m)
  )
)


# ============================================================
# 6. SET SEED
# ============================================================

set.seed(123)


# ============================================================
# 7. DEFAULT RANDOM FOREST
# ============================================================

rf_default <- randomForest(
  failed_12m ~ .,
  data = train_data,
  importance = TRUE
)


# ============================================================
# 8. PRINT MODEL
# ============================================================

print(rf_default)


# ============================================================
# 9. PREDICT CLASSES ON TEST DATA
# ============================================================

rf_class_pred <- predict(
  rf_default,
  newdata = test_data,
  type = "class"
)


# ============================================================
# 10. PREDICT FAILURE PROBABILITIES
# ============================================================

rf_prob_pred <- predict(
  rf_default,
  newdata = test_data,
  type = "prob"
)[, "1"]


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm <- confusionMatrix(
  data = rf_class_pred,
  reference = test_data$failed_12m,
  positive = "1"
)

print(cm)


# ============================================================
# 12. EXTRACT IMPORTANT METRICS
# ============================================================

accuracy <- cm$overall["Accuracy"]

recall <- cm$byClass["Sensitivity"]

specificity <- cm$byClass["Specificity"]

precision <- cm$byClass["Pos Pred Value"]

f1 <- cm$byClass["F1"]


cat("\nMODEL PERFORMANCE\n")

cat("Accuracy:", round(accuracy, 4), "\n")
cat("Recall:", round(recall, 4), "\n")
cat("Precision:", round(precision, 4), "\n")
cat("F1:", round(f1, 4), "\n")
cat("Specificity:", round(specificity, 4), "\n")


# ============================================================
# 13. ROC-AUC
# ============================================================

roc_object <- roc(
  response = test_data$failed_12m,
  predictor = rf_prob_pred,
  levels = c("0", "1")
)

roc_auc <- auc(roc_object)

cat(
  "ROC-AUC:",
  round(as.numeric(roc_auc), 4),
  "\n"
)

# ============================================================
# 14. PR-AUC
# ============================================================

# Precision-Recall AUC is particularly useful because bank
# failures are extremely rare in our sample.

positive_scores <- rf_prob_pred[
  test_data$failed_12m == "1"
]

negative_scores <- rf_prob_pred[
  test_data$failed_12m == "0"
]

pr_object <- pr.curve(
  scores.class0 = positive_scores,
  scores.class1 = negative_scores,
  curve = TRUE
)

pr_auc <- pr_object$auc.integral

test_prevalence <- mean(
  test_data$failed_12m == "1"
)

cat(
  "Test failure prevalence / naive PR baseline:",
  round(test_prevalence, 6),
  "\n"
)

cat(
  "PR-AUC:",
  round(pr_auc, 4),
  "\n"
)

png(
  "output/plots/rf_roc_curve.png",
  width = 900,
  height = 700
)

plot(
  roc_object,
  main = paste0(
    "Random Forest ROC Curve (AUC = ",
    round(as.numeric(roc_auc), 3),
    ")"
  )
)

dev.off()

png(
  "output/plots/rf_precision_recall_curve.png",
  width = 900,
  height = 700
)

plot(
  pr_object,
  main = paste0(
    "Random Forest Precision-Recall Curve (AUC = ",
    round(pr_auc, 3),
    ")"
  )
)

dev.off()


# ============================================================
# 15. NAIVE ALL-SURVIVE BENCHMARK
# ============================================================

# If we predict that every bank survives, accuracy will already
# be extremely high because failures are so rare.

naive_accuracy <- mean(
  test_data$failed_12m == "0"
)

cat(
  "Naive all-survive accuracy:",
  round(naive_accuracy, 4),
  "\n"
)

cat(
  "Random Forest accuracy:",
  round(as.numeric(accuracy), 4),
  "\n"
)


# ============================================================
# 16. FAILURE PROBABILITY DISTRIBUTIONS
# ============================================================

cat("\nPREDICTED PROBABILITIES - FAILED BANKS\n")

print(
  summary(
    rf_prob_pred[
      test_data$failed_12m == "1"
    ]
  )
)

cat("\nPREDICTED PROBABILITIES - SURVIVING BANKS\n")

print(
  summary(
    rf_prob_pred[
      test_data$failed_12m == "0"
    ]
  )
)


# Create dataframe for analysis
probability_results <- data.frame(
  actual = test_data$failed_12m,
  predicted_probability = rf_prob_pred
)


# Summary by actual outcome
probability_summary <- probability_results %>%
  group_by(actual) %>%
  summarise(
    n = n(),
    mean_probability = mean(predicted_probability),
    median_probability = median(predicted_probability),
    min_probability = min(predicted_probability),
    max_probability = max(predicted_probability),
    .groups = "drop"
  )

print(probability_summary)


# ============================================================
# 17. PLOT PREDICTED FAILURE PROBABILITIES
# ============================================================

dir.create(
  "output/plots",
  showWarnings = FALSE,
  recursive = TRUE
)

probability_plot <- ggplot(
  probability_results,
  aes(
    x = actual,
    y = predicted_probability
  )
) +
  geom_boxplot() +
  scale_y_continuous(
    trans = "sqrt"
  ) +
  labs(
    title = "Predicted Failure Risk by Actual Outcome",
    x = "Failure within next 12 months",
    y = "Predicted probability (square-root scale)"
  ) +
  theme_minimal()

ggsave(
  "output/plots/rf_probability_by_outcome.png",
  probability_plot,
  width = 7,
  height = 5
)

# ============================================================
# 18. ADD BANK IDENTIFIERS BACK TO PREDICTIONS
# ============================================================

# These variables were correctly excluded from model training,
# but we want them back when interpreting predictions.

test_identifiers <- model_data %>%
  filter(sample == "test") %>%
  select(
    CERT,
    NAME,
    report_date,
    failed_12m
  )

test_results <- test_identifiers %>%
  mutate(
    predicted_probability = rf_prob_pred
  ) %>%
  arrange(desc(predicted_probability)) %>%
  mutate(
    risk_rank = row_number(),
    risk_percentile = 100 * percent_rank(predicted_probability)
  )


# ============================================================
# 19. TOP-RISK CAPTURE RATE
# ============================================================

# This measures how many actual failures would be captured if
# regulators investigated only the riskiest X% of observations.

calculate_capture_rate <- function(data, top_percent) {
  
  number_to_review <- ceiling(
    nrow(data) * top_percent
  )
  
  top_risk <- data %>%
    arrange(desc(predicted_probability)) %>%
    slice_head(n = number_to_review)
  
  total_positive_observations <- sum(
    data$failed_12m == "1"
  )
  
  positive_observations_captured <- sum(
    top_risk$failed_12m == "1"
  )
  
  capture_rate <- positive_observations_captured /
    total_positive_observations
  
  data.frame(
    top_percent = top_percent,
    observations_reviewed = number_to_review,
    positive_observations_captured = positive_observations_captured,
    total_positive_observations = total_positive_observations,
    capture_rate = capture_rate
  )
}


capture_1 <- calculate_capture_rate(
  test_results,
  0.01
)

capture_5 <- calculate_capture_rate(
  test_results,
  0.05
)

capture_10 <- calculate_capture_rate(
  test_results,
  0.10
)

capture_results <- bind_rows(
  capture_1,
  capture_5,
  capture_10
)

cat("\nTOP-RISK CAPTURE RESULTS\n")

print(capture_results)


# ============================================================
# 20. SHOW HIGHEST-RISK OBSERVATIONS
# ============================================================

cat("\nTOP 20 HIGHEST-RISK BANK-QUARTER OBSERVATIONS\n")

print(
  test_results %>%
    select(
      CERT,
      NAME,
      report_date,
      failed_12m,
      predicted_probability
    ) %>%
    slice_head(n = 20)
)


# ============================================================
# 21. CHECK ACTUAL FAILED BANKS
# ============================================================

failed_bank_results <- test_results %>%
  filter(
    failed_12m == "1"
  ) %>%
  select(
    CERT,
    NAME,
    report_date,
    predicted_probability,
    risk_rank,
    risk_percentile
  ) %>%
  arrange(
    desc(predicted_probability)
  )

cat("\nACTUAL FAILURES RANKED BY PREDICTED RISK\n")

print(failed_bank_results)

# ============================================================
# 22. PERMUTATION VARIABLE IMPORTANCE
# ============================================================

importance_values <- importance(
  rf_default,
  type = 1
)

importance_df <- data.frame(
  variable = rownames(importance_values),
  permutation_importance = importance_values[, 1]
) %>%
  arrange(desc(permutation_importance)) %>%
  mutate(
    rank = row_number()
  ) %>%
  select(
    rank,
    variable,
    permutation_importance
  )

cat("\nPERMUTATION IMPORTANCE RANKING\n")

print(importance_df)

# Plot
varImpPlot(
  rf_default,
  type = 1,
  main = "Random Forest Permutation Importance"
)

# Save
write.csv(
  importance_df,
  "output/rf_default_importance.csv",
  row.names = FALSE
)

# 23. Consolidated Performance Table
if (is.na(f1) || is.nan(f1)) {
  f1 <- 0
}

performance_results <- data.frame(
  model = "Default Random Forest",
  accuracy = as.numeric(accuracy),
  naive_accuracy = as.numeric(naive_accuracy),
  recall = as.numeric(recall),
  precision = as.numeric(precision),
  f1 = as.numeric(f1),
  specificity = as.numeric(specificity),
  balanced_accuracy = as.numeric(
    cm$byClass["Balanced Accuracy"]
  ),
  roc_auc = as.numeric(roc_auc),
  pr_auc = as.numeric(pr_auc)
)

print(performance_results)

write.csv(
  performance_results,
  "output/rf_default_performance.csv",
  row.names = FALSE
)

# ============================================================
# 24. SAVE NEW RESULTS
# ============================================================

write.csv(
  probability_summary,
  "output/rf_probability_summary.csv",
  row.names = FALSE
)

write.csv(
  capture_results,
  "output/rf_capture_rates.csv",
  row.names = FALSE
)

write.csv(
  test_results,
  "output/rf_test_predictions.csv",
  row.names = FALSE
)

write.csv(
  failed_bank_results,
  "output/rf_failed_bank_predictions.csv",
  row.names = FALSE
)

# 25. Disconnect
dbDisconnect(
  con,
  shutdown = TRUE
)

message("Baseline Random Forest analysis complete.")
