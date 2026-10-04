# ============================================================
# Baseline Random Forest model
# ============================================================

library(DBI)
library(duckdb)
library(dplyr)
library(randomForest)
library(caret)
library(pROC)

source("config.R")


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
# 14. VARIABLE IMPORTANCE
# ============================================================

importance_values <- importance(
  rf_default,
  type = 1
)

print(
  importance_values
)

varImpPlot(
  rf_default,
  type = 1,
  main = "Random Forest Permutation Importance"
)


# ============================================================
# 15. SAVE PERFORMANCE RESULTS
# ============================================================

dir.create(
  "output",
  showWarnings = FALSE,
  recursive = TRUE
)

performance_results <- data.frame(
  model = "Default Random Forest",
  accuracy = as.numeric(accuracy),
  recall = as.numeric(recall),
  precision = as.numeric(precision),
  f1 = as.numeric(f1),
  specificity = as.numeric(specificity),
  roc_auc = as.numeric(roc_auc)
)

write.csv(
  performance_results,
  "output/rf_default_performance.csv",
  row.names = FALSE
)


# ============================================================
# 16. SAVE FEATURE IMPORTANCE
# ============================================================

importance_df <- data.frame(
  variable = rownames(importance_values),
  permutation_importance = importance_values[, 1]
)

importance_df <- importance_df %>%
  arrange(
    desc(permutation_importance)
  )

write.csv(
  importance_df,
  "output/rf_default_importance.csv",
  row.names = FALSE
)


# ============================================================
# 17. DISCONNECT
# ============================================================

dbDisconnect(
  con,
  shutdown = TRUE
)

message("Default Random Forest complete.")