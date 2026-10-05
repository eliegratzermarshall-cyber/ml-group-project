################################################################################
# MODELS_LOGISTIC_NN.R
# Logistic Regression (Benchmark) and Neural Networks for Bank Failures
################################################################################

# ============================================================
# 1. LOAD REQUIRED PACKAGES
# ============================================================
# install.packages(c("nnet", "dplyr", "caret", "pROC", "PRROC", "ggplot2"))

library(nnet)      # Fast neural networks for large datasets
library(dplyr)     # Data manipulation
library(caret)     # Confusion matrix and evaluation
library(pROC)      # ROC calculations
library(PRROC)     # Precision-Recall calculations (crucial for rare events)
library(ggplot2)   # Data visualization

# ============================================================
# 2. CREATE OUTPUT DIRECTORIES
# ============================================================
dir.create("output", showWarnings = FALSE, recursive = TRUE)
dir.create("output/plots", showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 3. LOAD AND PREPARE DATASET
# ============================================================
model_data <- read.csv("data/model_dataset.csv")

cat("\nModel dataset loaded.\n")
cat("Total Rows:", nrow(model_data), "\n")

# Define the 16 CAMELS features used for prediction
feature_cols <- c(
  "equity_ratio", "tier1_leverage", "npl_ratio", "reserve_ratio",
  "re_loan_share", "construction_share", "bank_age", "roa", "nim",
  "loans_to_assets", "loans_to_deposits", "brokered_share",
  "uninsured_share", "funding_cost", "securities_share", "log_assets"
)

# Convert target variable: factor for classification, numeric for neural network
model_data <- model_data %>%
  mutate(
    failed_12m = factor(failed_12m, levels = c(0, 1)),
    failed_numeric = as.numeric(as.character(failed_12m)) # Ensure 0 and 1
  )

# Split into training and test data based on pre-defined sample column
train_data <- model_data %>% filter(sample == "train")
test_data  <- model_data %>% filter(sample == "test")

# Extract identifiers for later analysis (needed for top-risk rankings)
test_identifiers <- test_data %>%
  select(CERT, NAME, report_date, failed_12m)

cat("\nTraining observations:", nrow(train_data), "\n")
cat("Test observations:", nrow(test_data), "\n")

# ============================================================
# 4. HELPER FUNCTION: CALCULATE CAPTURE RATE
# ============================================================
# This function calculates how many actual failures are captured 
# if regulators only investigate the riskiest X% of banks.
calculate_capture_rate <- function(data, top_percent) {
  number_to_review <- ceiling(nrow(data) * top_percent)
  
  top_risk <- data %>%
    arrange(desc(predicted_probability)) %>%
    slice_head(n = number_to_review)
  
  total_positive <- sum(data$failed_12m == "1")
  positive_captured <- sum(top_risk$failed_12m == "1")
  capture_rate <- positive_captured / total_positive
  
  data.frame(
    top_percent = top_percent,
    observations_reviewed = number_to_review,
    positive_observations_captured = positive_captured,
    total_positive_observations = total_positive,
    capture_rate = capture_rate
  )
}

################################################################################
# PART A: LOGISTIC REGRESSION (BENCHMARK)
################################################################################
cat("\n============================================================\n")
cat("TRAINING LOGISTIC REGRESSION\n")
cat("============================================================\n")

# Build formula dynamically
log_formula <- as.formula(paste("failed_12m ~", paste(feature_cols, collapse = " + ")))

# Train the model
log_model <- glm(log_formula, data = train_data, family = binomial(link = "logit"))

# Predict probabilities and classes (0.5 threshold)
log_prob_pred <- predict(log_model, newdata = test_data, type = "response")
log_class_pred <- factor(ifelse(log_prob_pred > 0.5, 1, 0), levels = c(0, 1))

# ------------------------------------------------------------
# EVALUATION: LOGISTIC REGRESSION
# ------------------------------------------------------------
cm_log <- confusionMatrix(data = log_class_pred, reference = test_data$failed_12m, positive = "1")
roc_log <- auc(roc(response = test_data$failed_12m, predictor = log_prob_pred, levels = c("0", "1")))

# Precision-Recall AUC (PR-AUC)
log_pr_object <- pr.curve(
  scores.class0 = log_prob_pred[test_data$failed_12m == "1"],
  scores.class1 = log_prob_pred[test_data$failed_12m == "0"],
  curve = TRUE
)

# Extract metrics safely
naive_acc <- mean(test_data$failed_12m == "0")
f1_log <- cm_log$byClass["F1"]
if (is.na(f1_log) || is.nan(f1_log)) f1_log <- 0

log_results <- data.frame(
  model = "Logistic Regression",
  accuracy = as.numeric(cm_log$overall["Accuracy"]),
  naive_accuracy = as.numeric(naive_acc),
  recall = as.numeric(cm_log$byClass["Sensitivity"]),
  precision = as.numeric(cm_log$byClass["Pos Pred Value"]),
  f1 = as.numeric(f1_log),
  specificity = as.numeric(cm_log$byClass["Specificity"]),
  balanced_accuracy = as.numeric(cm_log$byClass["Balanced Accuracy"]),
  roc_auc = as.numeric(roc_log),
  pr_auc = as.numeric(log_pr_object$auc.integral)
)

# Create Identifiable Results DataFrame
log_test_results <- test_identifiers %>%
  mutate(predicted_probability = log_prob_pred) %>%
  arrange(desc(predicted_probability)) %>%
  mutate(
    risk_rank = row_number(),
    risk_percentile = 100 * percent_rank(predicted_probability)
  )

# Calculate Capture Rates
log_capture_results <- bind_rows(
  calculate_capture_rate(log_test_results, 0.01),
  calculate_capture_rate(log_test_results, 0.05),
  calculate_capture_rate(log_test_results, 0.10)
)

# Plot ROC & PR Curves
png("output/plots/logreg_roc_curve.png", width = 900, height = 700)
plot(roc(test_data$failed_12m, log_prob_pred), main = paste0("Logistic Regression ROC (AUC = ", round(roc_log, 3), ")"))
dev.off()

png("output/plots/logreg_pr_curve.png", width = 900, height = 700)
plot(log_pr_object, main = paste0("Logistic Regression PR Curve (AUC = ", round(log_pr_object$auc.integral, 3), ")"))
dev.off()

# Save LogReg Results
write.csv(log_results, "output/logreg_performance.csv", row.names = FALSE)
write.csv(log_capture_results, "output/logreg_capture_rates.csv", row.names = FALSE)
write.csv(log_test_results, "output/logreg_test_predictions.csv", row.names = FALSE)


################################################################################
# PART B: NEURAL NETWORK
################################################################################
cat("\n============================================================\n")
cat("TRAINING NEURAL NETWORK\n")
cat("============================================================\n")

# ------------------------------------------------------------
# NN Preprocessing (Min-Max Scaling)
# ------------------------------------------------------------
maxs <- apply(train_data[, feature_cols], MARGIN = 2, max)
mins <- apply(train_data[, feature_cols], MARGIN = 2, min)

train_scaled <- train_data
test_scaled <- test_data

train_scaled[, feature_cols] <- scale(train_data[, feature_cols], center = mins, scale = maxs - mins)
test_scaled[, feature_cols] <- scale(test_data[, feature_cols], center = mins, scale = maxs - mins)

# Handle class imbalance with weights
count_0 <- sum(train_scaled$failed_numeric == 0)
count_1 <- sum(train_scaled$failed_numeric == 1)
train_weights <- ifelse(train_scaled$failed_numeric == 1, count_0 / count_1, 1)

# Train the model
set.seed(123)
nn_model <- nnet(
  log_formula, 
  data = train_scaled, 
  weights = train_weights,    
  size = 5,                   
  maxit = 500, 
  trace = FALSE, # Set to TRUE if you want to see the iterations           
  MaxNWts = 5000
)

# Predict probabilities and classes
nn_prob_pred <- as.numeric(predict(nn_model, test_scaled[, feature_cols], type = "raw"))
nn_class_pred <- factor(ifelse(nn_prob_pred > 0.5, 1, 0), levels = c(0, 1))

# ------------------------------------------------------------
# EVALUATION: NEURAL NETWORK
# ------------------------------------------------------------
cm_nn <- confusionMatrix(data = nn_class_pred, reference = test_data$failed_12m, positive = "1")
roc_nn <- auc(roc(response = test_data$failed_12m, predictor = nn_prob_pred, levels = c("0", "1")))

# Precision-Recall AUC (PR-AUC)
nn_pr_object <- pr.curve(
  scores.class0 = nn_prob_pred[test_data$failed_12m == "1"],
  scores.class1 = nn_prob_pred[test_data$failed_12m == "0"],
  curve = TRUE
)

# Extract metrics safely
f1_nn <- cm_nn$byClass["F1"]
if (is.na(f1_nn) || is.nan(f1_nn)) f1_nn <- 0

nn_results <- data.frame(
  model = "Neural Network (nnet, size 5)",
  accuracy = as.numeric(cm_nn$overall["Accuracy"]),
  naive_accuracy = as.numeric(naive_acc),
  recall = as.numeric(cm_nn$byClass["Sensitivity"]),
  precision = as.numeric(cm_nn$byClass["Pos Pred Value"]),
  f1 = as.numeric(f1_nn),
  specificity = as.numeric(cm_nn$byClass["Specificity"]),
  balanced_accuracy = as.numeric(cm_nn$byClass["Balanced Accuracy"]),
  roc_auc = as.numeric(roc_nn),
  pr_auc = as.numeric(nn_pr_object$auc.integral)
)

# Create Identifiable Results DataFrame
nn_test_results <- test_identifiers %>%
  mutate(predicted_probability = nn_prob_pred) %>%
  arrange(desc(predicted_probability)) %>%
  mutate(
    risk_rank = row_number(),
    risk_percentile = 100 * percent_rank(predicted_probability)
  )

# Calculate Capture Rates
nn_capture_results <- bind_rows(
  calculate_capture_rate(nn_test_results, 0.01),
  calculate_capture_rate(nn_test_results, 0.05),
  calculate_capture_rate(nn_test_results, 0.10)
)

# Plot ROC & PR Curves
png("output/plots/nn_roc_curve.png", width = 900, height = 700)
plot(roc(test_data$failed_12m, nn_prob_pred), main = paste0("Neural Network ROC (AUC = ", round(roc_nn, 3), ")"))
dev.off()

png("output/plots/nn_pr_curve.png", width = 900, height = 700)
plot(nn_pr_object, main = paste0("Neural Network PR Curve (AUC = ", round(nn_pr_object$auc.integral, 3), ")"))
dev.off()

# Save NN Results
write.csv(nn_results, "output/nn_performance.csv", row.names = FALSE)
write.csv(nn_capture_results, "output/nn_capture_rates.csv", row.names = FALSE)
write.csv(nn_test_results, "output/nn_test_predictions.csv", row.names = FALSE)

# ============================================================
# FINAL SUMMARY PRINT
# ============================================================
cat("\n################################################################################\n")
cat("FINAL MODEL PERFORMANCE (ROC-AUC):\n")
cat("Logistic Regression: ", round(roc_log, 4), "\n")
cat("Neural Network:      ", round(roc_nn, 4), "\n")
cat("################################################################################\n")
message("Analysis complete. All standardized metrics, CSVs, and plots saved in 'output/' folder.")
