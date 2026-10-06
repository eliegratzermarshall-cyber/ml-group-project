################################################################################
# TUNED RANDOM FOREST: TUNING, PREDICTION, IMPORTANCE AND REMARKS
################################################################################
library(DBI)
library(duckdb)
library(randomForest)
library(caret)       # tuning (train) and confusionMatrix()
library(pROC)        # ROC curves and ROC-AUC
library(PRROC)       # precision-recall curves and PR-AUC
library(doParallel)  # parallel processing for the tuning

source("config.R")

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
#Bank fails within the next 12 months ("failed") or not ("survived").
#"failed" is the first level: caret and pROC treat it as the positive class.

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
#We tune mtry (variables tried per split) and ntree (number of trees).
#Baseline: mtry = 4, ntree = 500. Every combination below is tried.

mtry_grid  <- c(2, 3, 4, 6, 8, 12)
ntree_grid <- c(100, 300, 500)
tune_grid  <- expand.grid(mtry = mtry_grid, ntree = ntree_grid)

default_mtry  <- floor(sqrt(length(feature_cols)))   # baseline values, for labelling
default_ntree <- 500

#LOOCV settings: subsample size and share of failures in it
loocv_n          <- 500
loocv_fail_share <- 0.10

#TRUE  = run the tuning (sections 6-7, the slow part)
#FALSE = load the results saved by an earlier run (takes seconds)
#After one successful run, set this to FALSE. Set it to TRUE again only if
#the data or the tuning settings change.
run_tuning <- TRUE

#Memory available for the tuning: set this to your computer's RAM in GB.
#It decides how many forests are grown at the same time (section 6c).
ram_gb <- 32

#Tuning subsample: all failed banks plus this many surviving banks (section 6b).
#One forest on the full training data needs about 11 GB and many minutes, so
#tuning on the full data would take many hours on a laptop.
tune_n_surv <- 20000

################################################################################
#6. Tuning with 10-fold cross validation (caret)
################################################################################
#Criterion: ROC-AUC. Accuracy is not useful here, because predicting
#"survived" for every bank already gives a very high accuracy (see section 11).

#6a. A random forest method for caret that tunes mtry AND ntree ---------------

#caret's "rf" method only tunes mtry, so we define our own method that tunes
#both. For each mtry, only the 500-tree forest is grown; 100 and 300 trees are
#evaluated with its first trees. Keep these functions self-contained: in
#parallel they run in separate R sessions.

rf_ntree_mtry <- list(
  label      = "Random forest (mtry and ntree)",
  library    = "randomForest",
  type       = "Classification",
  parameters = data.frame(parameter = c("mtry", "ntree"),
                          class     = c("numeric", "numeric"),
                          label     = c("variables per split", "number of trees")),
  
  #Default grid (not used here, since we always give tuneGrid)
  grid = function(x, y, len = NULL, search = "grid") {
    expand.grid(mtry = floor(sqrt(ncol(x))), ntree = 500)
  },
  
  #For each mtry: grow only the largest ntree, the smaller ones are submodels
  loop = function(grid) {
    loop <- aggregate(ntree ~ mtry, data = grid, FUN = max)
    submodels <- vector("list", nrow(loop))
    for (i in 1:nrow(loop)) {
      ntrees <- grid$ntree[grid$mtry == loop$mtry[i]]
      submodels[[i]] <- data.frame(ntree = ntrees[ntrees < loop$ntree[i]])
    }
    list(loop = loop, submodels = submodels)
  },
  
  #Grow the forest. last = TRUE only for the final model on all training data:
  #only that model needs the (slow) permutation importance.
  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
    randomForest::randomForest(x = x, y = y, mtry = param$mtry,
                               ntree = param$ntree, importance = last, ...)
  },
  
  #Failure probability = share of the first k trees that vote "failed"
  prob = function(modelFit, newdata, submodels = NULL) {
    votes  <- predict(modelFit, newdata, predict.all = TRUE)$individual
    ntrees <- c(modelFit$ntree, submodels$ntree)
    out    <- vector("list", length(ntrees))
    for (k in 1:length(ntrees)) {
      p <- rowMeans(votes[, 1:ntrees[k], drop = FALSE] == modelFit$classes[1])
      out[[k]] <- data.frame(p, 1 - p)
      names(out[[k]]) <- modelFit$classes
    }
    if (is.null(submodels)) out[[1]] else out
  },
  
  #Predicted class = the class with the majority of the votes
  predict = function(modelFit, newdata, submodels = NULL) {
    votes  <- predict(modelFit, newdata, predict.all = TRUE)$individual
    ntrees <- c(modelFit$ntree, submodels$ntree)
    out    <- vector("list", length(ntrees))
    for (k in 1:length(ntrees)) {
      p <- rowMeans(votes[, 1:ntrees[k], drop = FALSE] == modelFit$classes[1])
      out[[k]] <- factor(ifelse(p > 0.5, modelFit$classes[1], modelFit$classes[2]),
                         levels = modelFit$classes)
    }
    if (is.null(submodels)) out[[1]] else out
  },
  
  #If two combinations are equally good: prefer fewer trees, then smaller mtry
  sort   = function(x) x[order(x$ntree, x$mtry), ],
  levels = function(x) x$classes
)

#6b. Tuning subsample ----------------------------------------------------------

#The hyperparameters are chosen on a stratified subsample of the training data:
#all failed banks plus tune_n_surv randomly drawn surviving banks. This makes
#each forest small enough to grow several at once. ROC-AUC only depends on the
#ranking of banks, so it is comparable across mtry/ntree even though failures
#are more frequent in the subsample. The chosen combination is then refit on
#the FULL training data (section 8b), with the natural class mix, like the
#baseline.

fail_idx <- which(train_y == "failed")
surv_idx <- which(train_y == "survived")

set.seed(123)
tune_idx <- c(fail_idx, sample(surv_idx, min(length(surv_idx), tune_n_surv)))
tune_x   <- train_x[tune_idx, ]
tune_y   <- train_y[tune_idx]

table(tune_y)   # check: class counts

#6c. Parallel processing and seeds ---------------------------------------------

#caret grows several forests at once, each in its own R session (worker).
#While growing, randomForest reserves memory for the largest possible tree
#(about 2 x n nodes per tree, for all trees), roughly 56 bytes x n x ntree,
#plus temporary copies. Only afterwards is the forest trimmed to its real size.
#With one worker per core this exceeds the RAM of a laptop, so the number of
#workers is limited by memory: at most half of ram_gb is used for forests.
#Because every resample has fixed seeds (make_seeds below), the results are
#identical for any number of workers; fewer workers only take longer.

gb_per_forest <- 2 * 56 * nrow(tune_x) * max(ntree_grid) / 1e9   # rough upper estimate
n_workers     <- max(1, min(parallel::detectCores(logical = FALSE) - 1,
                            floor(0.5 * ram_gb / gb_per_forest)))
cat("Approx. GB per forest:", round(gb_per_forest, 2),
    "-> parallel workers:", n_workers, "\n")

#To measure the real memory use of one forest instead of the estimate:
#  gc(reset = TRUE); tmp <- randomForest(train_x, train_y, ntree = 500); gc()
#and look at the "max used" column.

#Runs train() on n_workers parallel workers. on.exit() always shuts the
#workers down, also when train() stops with an error or is interrupted,
#so no R sessions are left behind holding memory.
#Progress: the workers write one line per forest to log_file. Watch it in
#RStudio's Terminal tab with:   tail -f output/tuning_log.txt
#Lines starting with "-" are finished forests (60 for 10-fold CV, 3000 for
#LOOCV). Count them with:       grep -c "^-" output/tuning_log.txt
train_parallel <- function(..., log_file = "output/tuning_log.txt") {
  if (file.exists(log_file)) file.remove(log_file)
  cl <- makePSOCKcluster(n_workers, outfile = log_file)
  registerDoParallel(cl)
  on.exit({
    stopCluster(cl)
    registerDoSEQ()
  })
  train(...)
}

#Fixed seeds for every resample, so parallel runs give identical results
make_seeds <- function(n_resamples) {
  set.seed(123)
  seeds <- vector("list", n_resamples + 1)
  for (b in 1:n_resamples) seeds[[b]] <- sample.int(100000, nrow(tune_grid))
  seeds[[n_resamples + 1]] <- sample.int(100000, 1)   # final model
  seeds
}

#6d. Folds ---------------------------------------------------------------------

#Fixed folds, so every combination is compared on exactly the same splits.
#createFolds() is stratified: every fold contains failures.
set.seed(123)
cv_folds <- createFolds(tune_y, k = 10, returnTrain = TRUE)

ctrl_cv10 <- trainControl(method          = "cv",
                          index           = cv_folds,
                          seeds           = make_seeds(10),
                          classProbs      = TRUE,          # needed for ROC-AUC
                          summaryFunction = twoClassSummary,
                          returnData      = FALSE,         # saves memory
                          verboseIter     = TRUE)          # progress in the log

#6e. Tuning --------------------------------------------------------------------

#One call to train() evaluates all mtry/ntree combinations on the tuning
#subsample. caret's own final model is also fit on the subsample only; the
#model we use is refit on the full training data in section 8b.
if (run_tuning) {
  tuning_start <- Sys.time()
  set.seed(123)
  fit_cv10 <- train_parallel(x = tune_x, y = tune_y,
                             method    = rf_ntree_mtry,
                             metric    = "ROC",
                             trControl = ctrl_cv10,
                             tuneGrid  = tune_grid)
  print(paste('10-fold CV done after', round(difftime(Sys.time(), tuning_start,
                                                      units = "mins"), 1), 'minutes'))
  gc()   # free the memory used during tuning
}


################################################################################
#7. Tuning with leave-one-out cv (LOOCV, caret)
################################################################################
#One forest per observation is far too slow on the full data, so LOOCV runs on
#a subsample. It is only used to choose the hyperparameters; the final LOOCV
#model is refit on all training data.

#7a. Subsample -----------------------------------------------------------------

#Failures oversampled, so the subsample contains enough of them
#(fail_idx and surv_idx were defined in section 6b)
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
                           summaryFunction = twoClassSummary,
                           returnData      = FALSE,
                           verboseIter     = TRUE)

#7b. Tuning --------------------------------------------------------------------

#LOOCV gives one prediction per bank; caret computes the ROC-AUC from all of them
if (run_tuning) {
  tuning_start <- Sys.time()
  set.seed(123)
  fit_loocv <- train_parallel(x = loocv_x, y = loocv_y,
                              method    = rf_ntree_mtry,
                              metric    = "ROC",
                              trControl = ctrl_loocv,
                              tuneGrid  = tune_grid)
  print(paste('LOOCV done after', round(difftime(Sys.time(), tuning_start,
                                                 units = "mins"), 1), 'minutes'))
}

#7c. Save or load the tuning results -------------------------------------------

if (run_tuning) {
  save(fit_cv10, fit_loocv, file = "output/tuning_results.RData")
} else {
  load("output/tuning_results.RData")
}


################################################################################
#8. Best combinations and tuning tables
################################################################################
#8a. Best combination per resampling method ------------------------------------

best_mtry_cv10   <- fit_cv10$bestTune$mtry
best_ntree_cv10  <- fit_cv10$bestTune$ntree
best_mtry_loocv  <- fit_loocv$bestTune$mtry
best_ntree_loocv <- fit_loocv$bestTune$ntree

fit_cv10$bestTune
fit_loocv$bestTune

#8b. Final models on the full training data ------------------------------------

#Both tunings used subsamples, so the chosen combinations are refit on the
#full training data (natural class mix, as the baseline), with permutation
#importance. Each refit needs about as much memory as a baseline forest and
#takes a while, so the refits are saved and simply loaded when
#run_tuning = FALSE. do.trace = 50 prints progress every 50 trees.
#If both methods chose the same combination, the model is the same and we
#simply reuse it.
if (run_tuning) {
  refit_start <- Sys.time()
  set.seed(123)
  rf_cv10 <- randomForest(x = train_x, y = train_y,
                          mtry = best_mtry_cv10, ntree = best_ntree_cv10,
                          importance = TRUE, do.trace = 50)
  
  if (best_mtry_loocv == best_mtry_cv10 && best_ntree_loocv == best_ntree_cv10) {
    rf_loocv <- rf_cv10
  } else {
    gc()
    set.seed(123)
    rf_loocv <- randomForest(x = train_x, y = train_y,
                             mtry = best_mtry_loocv, ntree = best_ntree_loocv,
                             importance = TRUE, do.trace = 50)
  }
  print(paste('Final refits done after', round(difftime(Sys.time(), refit_start,
                                                        units = "mins"), 1), 'minutes'))
  save(rf_cv10, rf_loocv, file = "output/rf_final_fits.RData")
} else {
  load("output/rf_final_fits.RData")
}

#8c. Tuning tables -------------------------------------------------------------

#ROC-AUC for every ntree/mtry combination. Sens = recall of failed banks.
cv10_results  <- fit_cv10$results[, c("ntree", "mtry", "ROC", "ROCSD", "Sens", "Spec")]
loocv_results <- fit_loocv$results[, c("ntree", "mtry", "ROC", "Sens", "Spec")]
cv10_results  <- cv10_results[order(cv10_results$ntree, cv10_results$mtry), ]
loocv_results <- loocv_results[order(loocv_results$ntree, loocv_results$mtry), ]

cv10_results$best  <- cv10_results$mtry == best_mtry_cv10 & cv10_results$ntree == best_ntree_cv10
loocv_results$best <- loocv_results$mtry == best_mtry_loocv & loocv_results$ntree == best_ntree_loocv
cv10_results$baseline_default  <- cv10_results$mtry == default_mtry & cv10_results$ntree == default_ntree
loocv_results$baseline_default <- loocv_results$mtry == default_mtry & loocv_results$ntree == default_ntree

print(cv10_results, digits = 3, row.names = FALSE)
print(loocv_results, digits = 3, row.names = FALSE)

write.csv(cv10_results,  "output/rf_tuning_cv10.csv",  row.names = FALSE)
write.csv(loocv_results, "output/rf_tuning_loocv.csv", row.names = FALSE)

#8d. Plot: ROC-AUC vs mtry, one line per ntree (dashed line = baseline mtry) ---

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
par(mfrow = c(1, 1))


################################################################################
#9. (d) Final model and predictions on the test data
################################################################################
#Final model: the 10-fold CV tuned forest, refit on the full training data.
#Its tuning subsample is much larger than the LOOCV one, so its choice is
#more reliable. The LOOCV model is a robustness check.
#The test data was not used for any choice so far, so the test results below
#are an honest estimate of how the model predicts on new data.

final_model <- rf_cv10
final_model

#Predicted failure probabilities on the test data
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
#11. (e) Prediction accuracy: confusion matrices, recall, precision, F1
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

#Benchmarks: accuracy when predicting "survived" for every bank, and the
#PR-AUC of random guessing (= share of failed banks in the test data)
naive_accuracy <- mean(test_y == "survived")
naive_pr_auc   <- mean(test_y == "failed")
naive_accuracy
naive_pr_auc

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
par(mfrow = c(1, 1))


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
abline(h = naive_pr_auc, lty = 2)
legend('topright', legend = curve_names, col = cols, lty = 1)
dev.off()

#12a. Short discussion of the prediction accuracy -----------------------------

#The key numbers for the discussion, taken from the results above
final_row <- results[results$model == "10-fold CV (tuned threshold)", ]
base_row  <- results[results$model == "Baseline (0.5)", ]

cat("\nFINAL MODEL: mtry =", best_mtry_cv10, ", ntree =", best_ntree_cv10, "\n")
cat("Test ROC-AUC:", round(final_row$roc_auc, 3),
    "(baseline:", round(base_row$roc_auc, 3), ")\n")
cat("Test PR-AUC:", round(final_row$pr_auc, 3),
    "(baseline:", round(base_row$pr_auc, 3), "; random guessing:", round(naive_pr_auc, 4), ")\n")
cat("Recall at tuned threshold:", round(final_row$recall, 3),
    "; precision:", round(final_row$precision, 3), "\n")
cat("Accuracy:", round(final_row$accuracy, 4),
    "vs all-survive accuracy:", round(naive_accuracy, 4), "\n")


################################################################################
#13. (f) Permutation importance and CAMELS
################################################################################
#Permutation importance (randomForest package): for each variable, shuffle its
#values in the out-of-bag data, predict again, and measure how much the
#accuracy drops (mean decrease in accuracy). type = 1 selects this measure;
#scale = TRUE divides by its standard error (as in R_code-forests.R).
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
  sort(importance(rf_model, type = 1, scale = TRUE)[, 1], decreasing = TRUE)
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
        xlab = 'Mean decrease in accuracy (scaled)',
        main = 'Permutation importance by CAMELS component')
legend('bottomright', legend = names(camels_cols), fill = camels_cols)
dev.off()
par(mar = c(5.1, 4.1, 4.1, 2.1))


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
#A subsample of 1000 banks is enough for smooth curves and keeps this fast.

set.seed(123)
pd_data <- train_x[sample(nrow(train_x), min(1000, nrow(train_x))), ]

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
par(mfrow = c(1, 1))


################################################################################
#15. Save models
################################################################################

saveRDS(final_model, "output/rf_final_model.rds")   # 10-fold CV tuned
saveRDS(rf_cv10,  "output/rf_tuned_cv10.rds")
saveRDS(rf_loocv, "output/rf_tuned_loocv.rds")



print("Random Forest tuning and evaluation complete.")