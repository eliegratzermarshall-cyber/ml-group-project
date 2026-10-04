################################################################################
# 07_MODELS_LOGISTIC_NN.R
# Logistic Regression (Benchmark) and Neural Networks for Bank Failures
################################################################################

# Install packages if needed (only run once)
# install.packages(c("nnet", "dplyr", "caret", "pROC"))

# Load packages
library(nnet)      # Fast neural networks for large datasets
library(dplyr)     # Data manipulation
library(caret)     # Confusion matrix and evaluation
library(pROC)      # ROC and AUC calculations

################################################################################
# 1. Create Output Directory
################################################################################
dir.create("output", showWarnings = FALSE, recursive = TRUE)

################################################################################
# 2. Load and Check the Data
################################################################################
model_data <- read.csv("data/model_dataset.csv")

# Define the 16 features used for prediction
feature_cols <- c(
  "equity_ratio", "tier1_leverage", "npl_ratio", "reserve_ratio",
  "re_loan_share", "construction_share", "bank_age", "roa", "nim",
  "loans_to_assets", "loans_to_deposits", "brokered_share",
  "uninsured_share", "funding_cost", "securities_share", "log_assets"
)

# Convert the target variable to a factor for classification (0 = survived, 1 = failed)
# Also keep a numeric version for the Neural Network
model_data <- model_data %>%
  mutate(
    failed_factor = factor(failed_12m, levels = c(0, 1)),
    failed_numeric = as.numeric(failed_12m)
  )

# Split into training and test data 
# (The sample column was already created during feature engineering)
train_data <- model_data %>% filter(sample == "train")
test_data  <- model_data %>% filter(sample == "test")

print(paste("Training observations:", nrow(train_data)))
print(paste("Test observations:", nrow(test_data)))

################################################################################
# 3. Logistic Regression (Benchmark Model)
################################################################################
# The formula has the form y ~ col1 + col2 + ...
# Writing all 16 column names by hand is tedious, so we build the formula:
log_formula <- as.formula(paste("failed_factor ~", paste(feature_cols, collapse = " + ")))

# Train the logistic regression model
log_model <- glm(log_formula, data = train_data, family = binomial(link = "logit"))

# Predict probabilities on the test data
log_prob_pred <- predict(log_model, newdata = test_data, type = "response")

# Convert probabilities to class labels (using 0.5 threshold)
log_class_pred <- factor(ifelse(log_prob_pred > 0.5, 1, 0), levels = c(0, 1))

# Evaluate the model
cm_log <- confusionMatrix(data = log_class_pred, reference = test_data$failed_factor, positive = "1")
roc_log <- auc(roc(response = test_data$failed_factor, predictor = log_prob_pred, levels = c("0", "1")))

# Save performance results to output folder
log_results <- data.frame(
  model = "Logistic Regression",
  accuracy = as.numeric(cm_log$overall["Accuracy"]),
  recall = as.numeric(cm_log$byClass["Sensitivity"]),
  precision = as.numeric(cm_log$byClass["Pos Pred Value"]),
  f1 = as.numeric(cm_log$byClass["F1"]),
  specificity = as.numeric(cm_log$byClass["Specificity"]),
  roc_auc = as.numeric(roc_log)
)
write.csv(log_results, "output/log_reg_performance.csv", row.names = FALSE)

################################################################################
# 4. Neural Network Preprocessing (Scaling)
################################################################################
# Neural networks are sensitive to the scale of the inputs.
# We therefore bring every column to the range 0 to 1 (min-max scaling).

# Grab the min and max value per column using the apply function.
# MARGIN = 2 means "apply the function to each column".
maxs <- apply(train_data[, feature_cols], MARGIN = 2, max)
mins <- apply(train_data[, feature_cols], MARGIN = 2, min)

train_scaled <- train_data
test_scaled <- test_data

# scale() subtracts the min, divide by (max - min). The result is between 0 and 1.
train_scaled[, feature_cols] <- scale(train_data[, feature_cols], center = mins, scale = maxs - mins)
test_scaled[, feature_cols] <- scale(test_data[, feature_cols], center = mins, scale = maxs - mins)

################################################################################
# 5. Train the Neural Network
################################################################################
# Note: While 'neuralnet' is a great educational package, it fails to converge 
# on highly imbalanced datasets with ~200,000 rows. We therefore implement 'nnet', 
# which utilizes the more efficient BFGS optimization algorithm.

nn_formula <- as.formula(paste("failed_factor ~", paste(feature_cols, collapse = " + ")))

# Handle class imbalance (very few bank failures compared to healthy banks).
# We calculate class weights so the network pays attention to the rare failures.
count_0 <- sum(train_scaled$failed_numeric == 0)
count_1 <- sum(train_scaled$failed_numeric == 1)
weight_1 <- count_0 / count_1  

train_weights <- ifelse(train_scaled$failed_numeric == 1, weight_1, 1)

# Training is random (the starting weights are random), so we set a seed.
set.seed(123)

# Train the model (size = 5 hidden neurons)
nn_model <- nnet(
  nn_formula, 
  data = train_scaled, 
  weights = train_weights,    
  size = 5,                   
  maxit = 500, 
  trace = TRUE,               
  MaxNWts = 5000
)

################################################################################
# 6. Predict on the test data and Evaluate
################################################################################
# Give the model the test data and predict probabilities
nn_prob_pred <- as.numeric(predict(nn_model, test_scaled[, feature_cols], type = "raw"))

# Convert to classes based on 0.5 threshold
nn_class_pred <- factor(ifelse(nn_prob_pred > 0.5, 1, 0), levels = c(0, 1))

# Evaluate the Neural Network
cm_nn <- confusionMatrix(data = nn_class_pred, reference = test_scaled$failed_factor, positive = "1")
roc_nn <- auc(roc(response = test_scaled$failed_factor, predictor = nn_prob_pred, levels = c("0", "1")))

# Save performance results
nn_results <- data.frame(
  model = "Neural Network (nnet, size 5)",
  accuracy = as.numeric(cm_nn$overall["Accuracy"]),
  recall = as.numeric(cm_nn$byClass["Sensitivity"]),
  precision = as.numeric(cm_nn$byClass["Pos Pred Value"]),
  f1 = as.numeric(cm_nn$byClass["F1"]),
  specificity = as.numeric(cm_nn$byClass["Specificity"]),
  roc_auc = as.numeric(roc_nn)
)
write.csv(nn_results, "output/nn_performance.csv", row.names = FALSE)

# Print Final Summary
cat("\n################################################################################\n")
cat("FINAL MODEL PERFORMANCE:\n")
cat("Logistic Regression ROC-AUC:", round(as.numeric(roc_log), 4), "\n")
cat("Neural Network ROC-AUC:     ", round(as.numeric(roc_nn), 4), "\n")
cat("################################################################################\n")
