################################################################################
# Logistic Regression and Neural Network
################################################################################

# ============================================================
# 1. Packages & Environment Setup
# ============================================================
library(nnet)      # Feed-forward neural networks
library(dplyr)     # Data manipulation and piping
library(caret)     # Confusion matrix and performance metrics
library(pROC)      # ROC curve and AUC calculation
library(PRROC)     # Precision-Recall curve and AUC calculation
library(ggplot2)   # Data visualization and plotting

# ============================================================
# 2. Directory Structure Initialization
# ============================================================
dir.create("output", showWarnings = FALSE, recursive = TRUE)
dir.create("output/plots", showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 3. Data Loading & Preprocessing
# ============================================================
model_data <- read.csv("data/model_dataset.csv")

cat("\nModel dataset loaded.\n")
cat("Total Rows:", nrow(model_data), "\n")

# Define the 16 CAMELS predictor variables
feature_cols <- c(
  "equity_ratio", "tier1_leverage", "npl_ratio", "reserve_ratio",
  "re_loan_share", "construction_share", "bank_age", "roa", "nim",
  "loans_to_assets", "loans_to_deposits", "brokered_share",
  "uninsured_share", "funding_cost", "securities_share", "log_assets"
)

# Convert target variable to factor and create numeric counterpart for neural network
model_data <- model_data %>%
  mutate(
    failed_12m = factor(failed_12m, levels = c(0, 1)),
    failed_numeric = as.numeric(as.character(failed_12m))
  )

# Partition dataset into training and testing splits
train_data <- model_data %>% filter(sample == "train")
test_data  <- model_data %>% filter(sample == "test")

# Extract panel identifiers for risk ranking and validation
test_identifiers <- test_data %>%
  select(CERT, NAME, report_date, failed_12m)

cat("\nTraining observations:", nrow(train_data), "\n")
cat("Test observations:", nrow(test_data), "\n")

# ============================================================
# 4. Evaluation Helper Function: Capture Rate
# ============================================================
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
# PART A: Logistic Regression (Benchmark Model)
################################################################################
cat("\n============================================================\n")
cat("TRAINING LOGISTIC REGRESSION\n")
cat("============================================================\n")

log_formula <- as.formula(paste("failed_12m ~", paste(feature_cols, collapse = " + ")))
log_model <- glm(log_formula, data = train_data, family = binomial(link = "logit"))

log_prob_pred <- predict(log_model, newdata = test_data, type = "response")
log_class_pred <- factor(ifelse(log_prob_pred > 0.5, 1, 0), levels = c(0, 1))

# Performance Evaluation
cm_log <- confusionMatrix(data = log_class_pred, reference = test_data$failed_12m, positive = "1")
roc_log <- auc(roc(response = test_data$failed_12m, predictor = log_prob_pred, levels = c("0", "1")))

log_pr_object <- pr.curve(
  scores.class0 = log_prob_pred[test_data$failed_12m == "1"],
  scores.class1 = log_prob_pred[test_data$failed_12m == "0"],
  curve = TRUE
)

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

log_test_results <- test_identifiers %>%
  mutate(predicted_probability = log_prob_pred) %>%
  arrange(desc(predicted_probability)) %>%
  mutate(
    risk_rank = row_number(),
    risk_percentile = 100 * percent_rank(predicted_probability)
  )

log_capture_results <- bind_rows(
  calculate_capture_rate(log_test_results, 0.01),
  calculate_capture_rate(log_test_results, 0.05),
  calculate_capture_rate(log_test_results, 0.10)
)

# Export Outputs
png("output/plots/logreg_roc_curve.png", width = 900, height = 700)
plot(roc(test_data$failed_12m, log_prob_pred), main = paste0("Logistic Regression ROC (AUC = ", round(roc_log, 3), ")"))
dev.off()

png("output/plots/logreg_pr_curve.png", width = 900, height = 700)
plot(log_pr_object, main = paste0("Logistic Regression PR Curve (AUC = ", round(log_pr_object$auc.integral, 3), ")"))
dev.off()

write.csv(log_results, "output/logreg_performance.csv", row.names = FALSE)
write.csv(log_capture_results, "output/logreg_capture_rates.csv", row.names = FALSE)
write.csv(log_test_results, "output/logreg_test_predictions.csv", row.names = FALSE)
write.csv(log_test_results %>% filter(failed_12m == "1"), "output/logreg_failed_bank_predictions.csv", row.names = FALSE)


################################################################################
# PART B: Neural Network Model
################################################################################
cat("\n============================================================\n")
cat("TRAINING NEURAL NETWORK\n")
cat("============================================================\n")

# Min-Max Feature Scaling
maxs <- apply(train_data[, feature_cols], MARGIN = 2, max)
mins <- apply(train_data[, feature_cols], MARGIN = 2, min)

train_scaled <- train_data
test_scaled <- test_data

train_scaled[, feature_cols] <- scale(train_data[, feature_cols], center = mins, scale = maxs - mins)
test_scaled[, feature_cols] <- scale(test_data[, feature_cols], center = mins, scale = maxs - mins)

# Class Imbalance Adjustment via Observation Weights
count_0 <- sum(train_scaled$failed_numeric == 0)
count_1 <- sum(train_scaled$failed_numeric == 1)
train_weights <- ifelse(train_scaled$failed_numeric == 1, count_0 / count_1, 1)

set.seed(123)
nn_model <- nnet(
  log_formula, 
  data = train_scaled, 
  weights = train_weights,    
  size = 5,                   
  maxit = 500, 
  trace = FALSE,           
  MaxNWts = 5000
)

nn_prob_pred <- as.numeric(predict(nn_model, test_scaled[, feature_cols], type = "raw"))
nn_class_pred <- factor(ifelse(nn_prob_pred > 0.5, 1, 0), levels = c(0, 1))

# Performance Evaluation
cm_nn <- confusionMatrix(data = nn_class_pred, reference = test_data$failed_12m, positive = "1")
roc_nn <- auc(roc(response = test_data$failed_12m, predictor = nn_prob_pred, levels = c("0", "1")))

nn_pr_object <- pr.curve(
  scores.class0 = nn_prob_pred[test_data$failed_12m == "1"],
  scores.class1 = nn_prob_pred[test_data$failed_12m == "0"],
  curve = TRUE
)

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

nn_test_results <- test_identifiers %>%
  mutate(predicted_probability = nn_prob_pred) %>%
  arrange(desc(predicted_probability)) %>%
  mutate(
    risk_rank = row_number(),
    risk_percentile = 100 * percent_rank(predicted_probability)
  )

nn_capture_results <- bind_rows(
  calculate_capture_rate(nn_test_results, 0.01),
  calculate_capture_rate(nn_test_results, 0.05),
  calculate_capture_rate(nn_test_results, 0.10)
)

# Export Outputs
png("output/plots/nn_roc_curve.png", width = 900, height = 700)
plot(roc(test_data$failed_12m, nn_prob_pred), main = paste0("Neural Network ROC (AUC = ", round(roc_nn, 3), ")"))
dev.off()

png("output/plots/nn_pr_curve.png", width = 900, height = 700)
plot(nn_pr_object, main = paste0("Neural Network PR Curve (AUC = ", round(nn_pr_object$auc.integral, 3), ")"))
dev.off()

write.csv(nn_results, "output/nn_performance.csv", row.names = FALSE)
write.csv(nn_capture_results, "output/nn_capture_rates.csv", row.names = FALSE)
write.csv(nn_test_results, "output/nn_test_predictions.csv", row.names = FALSE)
write.csv(nn_test_results %>% filter(failed_12m == "1"), "output/nn_failed_bank_predictions.csv", row.names = FALSE)


################################################################################
# PART C: Comparative Visualizations & Feature Importance (CAMELS)
################################################################################

combined_performance <- bind_rows(log_results, nn_results)
write.csv(combined_performance, "output/logreg_nn_performance.csv", row.names = FALSE)

# 1. Probability Boxplots by Actual Outcome
prob_results <- bind_rows(
  data.frame(actual = test_data$failed_12m, prob = log_prob_pred, model = "Logistic Regression"),
  data.frame(actual = test_data$failed_12m, prob = nn_prob_pred, model = "Neural Network")
)
prob_plot <- ggplot(prob_results, aes(x = actual, y = prob)) +
  geom_boxplot() +
  scale_y_continuous(trans = "sqrt") +
  facet_wrap(~ model) +
  labs(title = "Predicted Failure Risk by Actual Outcome",
       x = "Failure (0 = Survived, 1 = Failed)", y = "Predicted probability (sqrt scale)") +
  theme_minimal()
ggsave("output/plots/logreg_nn_probability_by_outcome.png", prob_plot, width = 8, height = 5)

# 2. Visual Confusion Matrices
png("output/plots/logreg_nn_confusion_matrices.png", width = 900, height = 500)
par(mfrow = c(1, 2))
cm_list <- list("Logistic Regression" = cm_log$table, "Neural Network" = cm_nn$table)
for (mod_name in names(cm_list)) {
  tab <- cm_list[[mod_name]]
  plot(c(0, 2), c(0, 2), type = 'n', axes = FALSE, xlab = 'Actual', ylab = 'Predicted', main = mod_name)
  rect(c(0, 1, 0, 1), c(1, 1, 0, 0), c(1, 2, 1, 2), c(2, 2, 1, 1),
       col = c('palegreen', 'mistyrose', 'mistyrose', 'palegreen'), border = 'white')
  text(c(0.5, 1.5, 0.5, 1.5), c(1.5, 1.5, 0.5, 0.5),
       c(tab[1, 1], tab[1, 2], tab[2, 1], tab[2, 2]), cex = 1.5)
  axis(1, at = c(0.5, 1.5), labels = c('survived', 'failed'), tick = FALSE)
  axis(2, at = c(1.5, 0.5), labels = c('survived', 'failed'), tick = FALSE)
}
dev.off()
par(mfrow = c(1, 1))

# 3. CAMELS Feature Mapping & Importance Plots
camels_mapping <- c(equity_ratio="Capital", tier1_leverage="Capital", npl_ratio="Asset quality", 
                    reserve_ratio="Asset quality", re_loan_share="Asset quality", construction_share="Asset quality",
                    roa="Earnings", nim="Earnings", loans_to_assets="Liquidity", loans_to_deposits="Liquidity",
                    brokered_share="Liquidity", uninsured_share="Liquidity", securities_share="Liquidity", 
                    funding_cost="Sensitivity", bank_age="Control", log_assets="Control")

camels_cols <- c("Capital"='steelblue', "Asset quality"='orange', "Earnings"='darkgreen', 
                 "Liquidity"='purple', "Sensitivity"='firebrick', "Control"='grey60')

# Logistic Regression Importance
raw_imp_log <- varImp(log_model)
if(is.data.frame(raw_imp_log)) {
  imp_df_log <- data.frame(variable = rownames(raw_imp_log), importance = raw_imp_log[, 1])
} else {
  imp_df_log <- data.frame(variable = names(raw_imp_log$Overall), importance = raw_imp_log$Overall)
}
imp_df_log <- imp_df_log %>% arrange(desc(importance)) %>% mutate(scaled_imp = importance / max(importance) * 100)

png("output/plots/logreg_importance_camels.png", width = 1100, height = 800)
par(mar = c(5, 14, 4, 2))
barplot(rev(imp_df_log$scaled_imp), horiz = TRUE, las = 1, names.arg = rev(imp_df_log$variable),
        col = camels_cols[camels_mapping[rev(imp_df_log$variable)]],
        xlab = 'Relative Importance (Scaled 0-100)', main = 'Logistic Regression Importance by CAMELS')
legend('bottomright', legend = names(camels_cols), fill = camels_cols, cex = 0.9)
dev.off()
par(mar = c(5.1, 4.1, 4.1, 2.1))

# Neural Network Importance
raw_imp_nn <- varImp(nn_model)
if(is.data.frame(raw_imp_nn)) {
  imp_df_nn <- data.frame(variable = rownames(raw_imp_nn), importance = raw_imp_nn[, 1])
} else {
  imp_df_nn <- data.frame(variable = names(raw_imp_nn$Overall), importance = raw_imp_nn$Overall)
}
imp_df_nn <- imp_df_nn %>% arrange(desc(importance)) %>% mutate(scaled_imp = importance / max(importance) * 100)

png("output/plots/nn_importance_camels.png", width = 1100, height = 800)
par(mar = c(5, 14, 4, 2))
barplot(rev(imp_df_nn$scaled_imp), horiz = TRUE, las = 1, names.arg = rev(imp_df_nn$variable),
        col = camels_cols[camels_mapping[rev(imp_df_nn$variable)]],
        xlab = 'Relative Importance (Scaled 0-100)', main = 'Neural Network Importance by CAMELS')
legend('bottomright', legend = names(camels_cols), fill = camels_cols, cex = 0.9)
dev.off()
par(mar = c(5.1, 4.1, 4.1, 2.1))


# ============================================================
# 5. Final Execution Summary
# ============================================================
cat("\n################################################################################\n")
cat("FINAL MODEL PERFORMANCE (ROC-AUC):\n")
cat("Logistic Regression: ", round(roc_log, 4), "\n")
cat("Neural Network:      ", round(roc_nn, 4), "\n")
cat("################################################################################\n")
message("Analysis complete. Original metrics restored + symmetrical outputs saved in 'output/' folder.")
