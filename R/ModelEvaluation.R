################################################################################
#Tuned Random Forest model
################################################################################

library(DBI)
library(duckdb)
library(randomForest)
library(caret)   # tuning (train) and confusionMatrix()
library(pROC)
library(PRROC)

source("config.R")

#Create the output folders if they do not exist yet
dir.create("output/plots", showWarnings = FALSE, recursive = TRUE)


################################################################################
#1. Load model dataset
################################################################################
#Same data source as the baseline. We read the table and then close
#the database connection, since it is not needed afterwards.

model_data <- dbReadTable(con, "model_dataset")
dbDisconnect(con, shutdown = TRUE)

dim(model_data)


################################################################################
#2. Features
################################################################################
#Identical to the baseline, so that only the tuning differs.

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


################################################################################
#3. Target
################################################################################
#Defines parameters for bank failing or surviving (within span of 12 months)


model_data$failed_lbl <- factor(
  ifelse(model_data$failed_12m == 1, "failed", "survived"),
  levels = c("failed", "survived")
)

table(model_data$failed_lbl)


################################################################################
#4. Train / test split
################################################################################
#Same train/test split as the baseline (column "sample"), for a fair comparison.
#IDs are kept for matching later; features (x) & target (y) stored separately.

train_data <- model_data[model_data$sample == "train", ]
test_data  <- model_data[model_data$sample == "test", ]

train_x <- train_data[, feature_cols]
train_y <- train_data$failed_lbl
test_x  <- test_data[, feature_cols]
test_y  <- test_data$failed_lbl

#Double check the size of the two datasets and the class counts
dim(train_x)
dim(test_x)
table(train_y)


################################################################################
#5. Tuning settings
################################################################################
#Two hyperparameters are tuned: mtry (variables tried per split) and
#ntree (number of trees). Baseline defaults: mtry = floor(sqrt(16)) = 4, ntree = 500.
#mtry = 16 means no random variable selection (bagging).

mtry_grid  <- c(2, 3, 4, 6, 8, 12, 16)
ntree_grid <- c(100, 300, 500)

default_mtry  <- floor(sqrt(length(feature_cols)))   # baseline values, for labelling
default_ntree <- 500

#LOOCV settings: subsample size and share of failures in it
loocv_n          <- 1000
loocv_fail_share <- 0.10


################################################################################
#6. Tuning with 10-fold cross validation (caret)
################################################################################
#caret's "rf" method tunes mtry only, so we loop over ntree and
#let train() tune mtry inside the loop. Criterion: ROC-AUC.
#Slow: once sections 6-7 have run, results are saved (7e). In later sessions
#run sections 1-5, then only the load() line in 7e, and continue with section 8.

#6a. Parallel processing -------------------------------------------------------

#caret trains several forests at once on different cores (same results, faster).
#One core is left free; use fewer if the Mac runs low on memory.
library(doParallel)
cl <- makePSOCKcluster(parallel::detectCores() - 1)
registerDoParallel(cl)

#Fixed seeds for every resample, so parallel runs give identical results
make_seeds <- function(n_resamples) {
  set.seed(123)
  seeds <- vector("list", n_resamples + 1)
  for (b in 1:n_resamples) seeds[[b]] <- sample.int(100000, length(mtry_grid))
  seeds[[n_resamples + 1]] <- sample.int(100000, 1)   # final model
  seeds
}

#6b. Folds ---------------------------------------------------------------------

#Fixed folds, so every ntree is compared on exactly the same splits.
#createFolds() is stratified: every fold contains failures.
set.seed(123)
cv_folds <- createFolds(train_y, k = 10, returnTrain = TRUE)

ctrl_cv10 <- trainControl(method          = "cv",
                          index           = cv_folds,
                          seeds           = make_seeds(10),
                          classProbs      = TRUE,          # needed for ROC-AUC
                          summaryFunction = twoClassSummary)

#6c. Tuning loop ---------------------------------------------------------------

cv10_results <- NULL

for (t in 1:length(ntree_grid)) {
  set.seed(123)
  fit <- train(x = train_x, y = train_y,
               method    = "rf",
               metric    = "ROC",
               trControl = ctrl_cv10,
               tuneGrid  = data.frame(mtry = mtry_grid),
               ntree     = ntree_grid[t])
  
  cv10_results <- rbind(cv10_results,
                        data.frame(ntree = ntree_grid[t],
                                   fit$results[, c("mtry", "ROC", "ROCSD", "Sens", "Spec")]))
  print(paste('10-fold CV, ntree', ntree_grid[t], 'done'))
}

#6d. Best combination ----------------------------------------------------------

best_cv10       <- cv10_results[which.max(cv10_results$ROC), ]
best_mtry_cv10  <- best_cv10$mtry
best_ntree_cv10 <- best_cv10$ntree
best_cv10

#6e. Final model ---------------------------------------------------------------

#Refit on the full training data with the best combination
set.seed(123)
rf_cv10 <- randomForest(x = train_x, y = train_y,
                        mtry = best_mtry_cv10, ntree = best_ntree_cv10,
                        importance = TRUE)


################################################################################
#7. Tuning with leave-one-out cv (LOOCV, caret)
################################################################################
#One forest per observation is too slow on the full data, so LOOCV runs on a
#subsample. It is only used to choose the hyperparameters; the final model
#is refit on all training data.

#7a. Subsample -----------------------------------------------------------------

#Failures oversampled, so the subsample contains enough of them
fail_idx <- which(train_y == "failed")
surv_idx <- which(train_y == "survived")

set.seed(123)
n_fail_sub <- min(length(fail_idx), round(loocv_n * loocv_fail_share))
n_surv_sub <- min(length(surv_idx), loocv_n - n_fail_sub)

loocv_idx <- c(sample(fail_idx, n_fail_sub), sample(surv_idx, n_surv_sub))
loocv_x   <- train_x[loocv_idx, ]
loocv_y   <- train_y[loocv_idx]

table(loocv_y)   # check: class counts

ctrl_loocv <- trainControl(method          = "LOOCV",
                           seeds           = make_seeds(nrow(loocv_x)),
                           classProbs      = TRUE,
                           summaryFunction = twoClassSummary)

#7b. Tuning loop ---------------------------------------------------------------

#LOOCV gives one prediction per bank; caret computes the ROC-AUC from all of them
loocv_results <- NULL

for (t in 1:length(ntree_grid)) {
  set.seed(123)
  fit <- train(x = loocv_x, y = loocv_y,
               method    = "rf",
               metric    = "ROC",
               trControl = ctrl_loocv,
               tuneGrid  = data.frame(mtry = mtry_grid),
               ntree     = ntree_grid[t])
  
  loocv_results <- rbind(loocv_results,
                         data.frame(ntree = ntree_grid[t],
                                    fit$results[, c("mtry", "ROC", "Sens", "Spec")]))
  print(paste('LOOCV, ntree', ntree_grid[t], 'done'))
}

#7c. Best combination ----------------------------------------------------------

best_loocv       <- loocv_results[which.max(loocv_results$ROC), ]
best_mtry_loocv  <- best_loocv$mtry
best_ntree_loocv <- best_loocv$ntree
best_loocv

#7d. Final model ---------------------------------------------------------------

#Refit on the full training data with the best combination
set.seed(123)
rf_loocv <- randomForest(x = train_x, y = train_y,
                         mtry = best_mtry_loocv, ntree = best_ntree_loocv,
                         importance = TRUE)

#7e. Save tuning results -------------------------------------------------------

stopCluster(cl)   # tuning done: release the cores

#Rerun sections 6-7 only if the data or the tuning settings change
save(cv10_results, best_cv10, best_mtry_cv10, best_ntree_cv10, rf_cv10,
     loocv_results, best_loocv, best_mtry_loocv, best_ntree_loocv, rf_loocv,
     file = "output/tuning_results.RData")

#Later sessions: run this line instead of sections 6-7
# load("output/tuning_results.RData")


################################################################################
#8. Tuning tables
################################################################################
#ROC-AUC for every ntree/mtry combination. Sens = recall of failed banks.

cv10_results$best              <- 1:nrow(cv10_results) == which.max(cv10_results$ROC)
loocv_results$best             <- 1:nrow(loocv_results) == which.max(loocv_results$ROC)
cv10_results$baseline_default  <- cv10_results$mtry == default_mtry & cv10_results$ntree == default_ntree
loocv_results$baseline_default <- loocv_results$mtry == default_mtry & loocv_results$ntree == default_ntree

print(cv10_results, digits = 3)
print(loocv_results, digits = 3)

write.csv(cv10_results,  "output/rf_tuning_cv10.csv",  row.names = FALSE)
write.csv(loocv_results, "output/rf_tuning_loocv.csv", row.names = FALSE)

#Plot: ROC-AUC vs mtry, one line per ntree (dashed line = baseline mtry)
ntree_cols <- c('grey60', 'blue', 'black')

png("output/plots/rf_tuning_curves.png", width = 1200, height = 550)
par(mfrow = c(1, 2))

tuning_list  <- list(cv10_results, loocv_results)
tuning_names <- c('10-fold CV', 'LOOCV (subsample)')

for (r in 1:2) {
  res <- tuning_list[[r]]
  plot(range(mtry_grid), range(res$ROC), type = 'n',
       xlab = 'mtry', ylab = 'ROC-AUC', main = tuning_names[r])
  for (t in 1:length(ntree_grid)) {
    sub <- res[res$ntree == ntree_grid[t], ]
    points(sub$mtry, sub$ROC, type = 'b', col = ntree_cols[t])
  }
  abline(v = default_mtry, lty = 2)
  legend('bottomright', legend = paste('ntree =', ntree_grid),
         col = ntree_cols, lty = 1, pch = 1)
}

dev.off()


################################################################################
#9. Final model and predictions on the test data
################################################################################
#Final model: the 10-fold CV tuned forest. It is tuned on the full training
#data (LOOCV only on a subsample). The LOOCV model is a robustness check.

final_model <- rf_cv10

prob_cv10  <- predict(rf_cv10,  newdata = test_x, type = "prob")[, "failed"]
prob_loocv <- predict(rf_loocv, newdata = test_x, type = "prob")[, "failed"]

#The baseline predictions were saved by random_forest.R, sorted by risk.
#We match them back to the test set by bank (CERT) and quarter (report_date).

baseline_pred <- read.csv("output/rf_test_predictions.csv")

key_test     <- paste(test_data$CERT, as.character(test_data$report_date))
key_baseline <- paste(baseline_pred$CERT, as.character(baseline_pred$report_date))

prob_base <- baseline_pred$predicted_probability[match(key_test, key_baseline)]

#Double check that every test observation found its baseline prediction
if (any(is.na(prob_base))) {
  stop("Baseline predictions could not be matched to the test set. ",
       "Check that CERT/report_date formats agree.")
}


################################################################################
#10. Classification thresholds (training data only)
################################################################################
#The baseline classifies at 0.5. With rare failures that flags very
#few banks, so tuned models also get a threshold chosen on the
#out-of-bag predictions of the training data (Youden's J).
#The test set is never used to choose a threshold.

pick_threshold <- function(rf_model) {
  r <- roc(train_y, rf_model$votes[, "failed"],
           levels = c("survived", "failed"),
           direction = "<", quiet = TRUE)
  coords(r, "best", best.method = "youden", ret = "threshold")$threshold[1]
}

thr_cv10  <- pick_threshold(rf_cv10)
thr_loocv <- pick_threshold(rf_loocv)

thr_cv10
thr_loocv


################################################################################
#11. Confusion matrices, recall, precision, F1
################################################################################
#ROC-AUC from true classes and predicted failure probabilities
auc_fn <- function(truth, prob) {
  roc_obj <- roc(truth, prob, levels = c("survived", "failed"),
                 direction = "<", quiet = TRUE)
  as.numeric(auc(roc_obj))
}

#Tuned models are evaluated at 0.5 (same as baseline) and at their tuned threshold.

model_names <- c("Baseline (0.5)",
                 "10-fold CV (0.5)", "10-fold CV (tuned threshold)",
                 "LOOCV (0.5)",      "LOOCV (tuned threshold)")
model_probs <- list(prob_base, prob_cv10, prob_cv10, prob_loocv, prob_loocv)
model_thr   <- c(0.5, 0.5, thr_cv10, 0.5, thr_loocv)
n_models    <- length(model_names)

results <- data.frame(model = model_names, threshold = model_thr,
                      accuracy = 0, balanced_accuracy = 0, recall = 0,
                      precision = 0, f1 = 0, specificity = 0,
                      roc_auc = 0, pr_auc = 0)
cm_tables <- vector("list", n_models)

for (j in 1:n_models) {
  p    <- model_probs[[j]]
  pred <- factor(ifelse(p >= model_thr[j], "failed", "survived"),
                 levels = c("failed", "survived"))
  cm   <- confusionMatrix(pred, test_y, positive = "failed")
  cm_tables[[j]] <- cm$table
  
  results$accuracy[j]          <- cm$overall["Accuracy"]
  results$balanced_accuracy[j] <- cm$byClass["Balanced Accuracy"]
  results$recall[j]            <- cm$byClass["Recall"]
  results$precision[j]         <- cm$byClass["Precision"]   # NA if no bank is flagged
  results$f1[j]                <- cm$byClass["F1"]
  results$specificity[j]       <- cm$byClass["Specificity"]
  results$roc_auc[j]           <- auc_fn(test_y, p)
  results$pr_auc[j]            <- pr.curve(scores.class0 = p[test_y == "failed"],
                                           scores.class1 = p[test_y == "survived"])$auc.integral
}

print(results, digits = 3)
mean(test_y == "survived")   # accuracy when predicting "survived" for every bank

write.csv(results, "output/rf_tuning_comparison.csv", row.names = FALSE)

#Confusion matrices (rows = predicted, columns = actual)
for (j in 1:n_models) {
  print(model_names[j])
  print(cm_tables[[j]])
}

#Plot: green = correct, red = wrong
png("output/plots/rf_confusion_matrices.png", width = 900, height = 1100)
par(mfrow = c(3, 2))

for (j in 1:n_models) {
  tab <- cm_tables[[j]]
  plot(c(0, 2), c(0, 2), type = 'n', axes = FALSE,
       xlab = 'Actual', ylab = 'Predicted', main = model_names[j])
  rect(c(0, 1, 0, 1), c(1, 1, 0, 0), c(1, 2, 1, 2), c(2, 2, 1, 1),
       col = c('palegreen', 'mistyrose', 'mistyrose', 'palegreen'), border = 'white')
  text(c(0.5, 1.5, 0.5, 1.5), c(1.5, 1.5, 0.5, 0.5),
       c(tab[1, 1], tab[1, 2], tab[2, 1], tab[2, 2]), cex = 1.5)
  axis(1, at = c(0.5, 1.5), labels = c('failed', 'survived'), tick = FALSE)
  axis(2, at = c(1.5, 0.5), labels = c('failed', 'survived'), tick = FALSE)
}

dev.off()


################################################################################
#12. ROC and precision-recall curves
################################################################################
#Curves do not depend on the threshold, so one curve per model.

curve_names <- c("Baseline", "10-fold CV", "LOOCV")
curve_probs <- list(prob_base, prob_cv10, prob_loocv)
cols        <- c('black', 'blue', 'orange')

#ROC curves
png("output/plots/rf_tuned_roc_curves.png", width = 900, height = 700)
for (j in 1:3) {
  roc_obj <- roc(test_y, curve_probs[[j]], levels = c("survived", "failed"),
                 direction = "<", quiet = TRUE)
  plot(roc_obj, col = cols[j], add = (j > 1), main = 'ROC curves (test set)')
}
legend('bottomright', legend = curve_names, col = cols, lty = 1)
dev.off()

#Precision-recall curves (dashed line = failure rate, i.e. no skill)
png("output/plots/rf_tuned_pr_curves.png", width = 900, height = 700)
plot(c(0, 1), c(0, 1), type = 'n', xlab = 'Recall', ylab = 'Precision',
     main = 'Precision-recall curves (test set)')
for (j in 1:3) {
  p  <- curve_probs[[j]]
  pr <- pr.curve(scores.class0 = p[test_y == "failed"],
                 scores.class1 = p[test_y == "survived"], curve = TRUE)
  lines(pr$curve[, 1], pr$curve[, 2], col = cols[j])
}
abline(h = mean(test_y == "failed"), lty = 2)
legend('topright', legend = curve_names, col = cols, lty = 1)
dev.off()


################################################################################
#13. Permutation importance and CAMELS
################################################################################
#Mean decrease in accuracy when a variable is shuffled (out-of-bag data).
#Management has no variable in our data.

camels <- c(equity_ratio     = "Capital",       tier1_leverage     = "Capital",
            npl_ratio        = "Asset quality", reserve_ratio      = "Asset quality",
            re_loan_share    = "Asset quality", construction_share = "Asset quality",
            roa              = "Earnings",      nim                = "Earnings",
            loans_to_assets  = "Liquidity",     loans_to_deposits  = "Liquidity",
            brokered_share   = "Liquidity",     uninsured_share    = "Liquidity",
            securities_share = "Liquidity",     funding_cost       = "Sensitivity",
            bank_age         = "Control",       log_assets         = "Control")

#Importance sorted from most to least important
imp_sorted <- function(rf_model) {
  sort(importance(rf_model, type = 1)[, 1], decreasing = TRUE)
}
imp_cv10  <- imp_sorted(rf_cv10)
imp_loocv <- imp_sorted(rf_loocv)

#Importance table (ranked by the 10-fold CV model)
importance_df <- data.frame(rank             = 1:length(imp_cv10),
                            variable         = names(imp_cv10),
                            camels           = camels[names(imp_cv10)],
                            importance_cv10  = imp_cv10,
                            importance_loocv = imp_loocv[names(imp_cv10)],
                            row.names = NULL)

top5 <- importance_df[1:5, ]
print(top5)
table(top5$camels)   # CAMELS components in the top 5

write.csv(importance_df, "output/rf_tuned_importance.csv", row.names = FALSE)

#Rank stability: same top variables across models = robust result
baseline_imp <- read.csv("output/rf_default_importance.csv")

rank_comparison <- data.frame(
  variable      = names(imp_cv10),
  camels        = camels[names(imp_cv10)],
  rank_baseline = baseline_imp$rank[match(names(imp_cv10), baseline_imp$variable)],
  rank_cv10     = 1:length(imp_cv10),
  rank_loocv    = match(names(imp_cv10), names(imp_loocv)),
  row.names = NULL
)
print(rank_comparison)

write.csv(rank_comparison, "output/rf_importance_rank_comparison.csv", row.names = FALSE)

#Bar plot coloured by CAMELS component (final model = 10-fold CV)
camels_cols <- c("Capital"     = 'steelblue', "Asset quality" = 'orange',
                 "Earnings"    = 'darkgreen', "Liquidity"     = 'purple',
                 "Sensitivity" = 'firebrick', "Control"       = 'grey60')

png("output/plots/rf_importance_camels.png", width = 1000, height = 750)
par(mar = c(5, 11, 4, 2))
barplot(rev(imp_cv10), horiz = TRUE, las = 1,
        col = camels_cols[camels[names(rev(imp_cv10))]],
        xlab = 'Mean decrease in accuracy',
        main = 'Permutation importance by CAMELS component')
legend('bottomright', legend = names(camels_cols), fill = camels_cols)
dev.off()


################################################################################
#14. Closer look at the top 5 variables
################################################################################

top5_vars <- top5$variable

#Summary statistics: failed vs. surviving banks (training data)
stats <- list(mean   = function(x) mean(x, na.rm = TRUE),
              median = function(x) median(x, na.rm = TRUE),
              p25    = function(x) quantile(x, 0.25, na.rm = TRUE),
              p75    = function(x) quantile(x, 0.75, na.rm = TRUE))

top5_summary <- NULL
for (s in names(stats)) {
  by_outcome   <- aggregate(train_x[top5_vars], list(outcome = train_y), stats[[s]])
  top5_summary <- rbind(top5_summary, cbind(stat = s, by_outcome))
}
print(top5_summary)

write.csv(top5_summary, "output/rf_top5_by_outcome.csv", row.names = FALSE)

#Plot per variable: distribution (left) and partial dependence (right).
#Partial dependence = predicted failure risk as the variable changes, all else equal.
#do.call() passes the variable name stored in v.

set.seed(123)
pd_data <- train_x[sample(nrow(train_x), min(2000, nrow(train_x))), ]   # subsample for speed

png("output/plots/rf_top5_closer_look.png", width = 900, height = 1400)
par(mfrow = c(5, 2))

for (v in top5_vars) {
  boxplot(train_x[[v]] ~ train_y, outline = FALSE,
          col = c('firebrick', 'steelblue'), main = v, xlab = '', ylab = '')
  
  pd <- do.call("partialPlot", list(x = final_model, pred.data = pd_data, x.var = v,
                                    which.class = "failed", plot = FALSE))
  plot(pd$x, pd$y, type = 'l', main = paste(v, '- partial dependence'),
       xlab = v, ylab = 'Failure risk (logit scale)')
}

dev.off()


################################################################################
#15. Save models
################################################################################

saveRDS(final_model, "output/rf_final_model.rds")   # 10-fold CV tuned
saveRDS(rf_cv10,  "output/rf_tuned_cv10.rds")
saveRDS(rf_loocv, "output/rf_tuned_loocv.rds")

print("Random Forest tuning and evaluation complete.")