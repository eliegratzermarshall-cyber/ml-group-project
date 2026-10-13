# Predicting U.S. Commercial Bank Failures with Machine Learning

**Machine Learning in Finance (6,155,1.00) · University of St. Gallen · Group 2**
Prof. Dr. Despoina Makariou · Autumn Semester 2026

## Project overview

This project asks whether bank-level financial characteristics observed in quarter 
*t* can predict whether a U.S. commercial bank fails within the following 12 months. 
It also asks whether a random forest trained on the Global Financial Crisis generalizes 
to the 2023 failures better than a logistic regression.

- **Data:** Quarterly FDIC call report data (BankFind Suite API), institution 
characteristics, and the FDIC failed bank list
- **Unit of observation:** bank-quarter
- **Target:** `failed_12m`, which is 1 if the bank fails within 12 months after 
quarter *t*
- **Training sample:** 2004Q1–2014Q4 (195,990 bank-quarters)
- **Out-of-time test sample:** 2021Q1–2023Q4 (35,052 bank-quarters)
- **Models:** random forest (primary), logistic regression and neural network 
(benchmarks)
- **Evaluation:** ROC-AUC, PR-AUC, balanced accuracy, recall, precision; 
classification threshold chosen with Youden's J on out-of-bag predictions

## Repository structure

```
.
├── config.R                          # Paths, seeds and global settings (source first)
├── data/
│   ├── download_raw_data.R           # Downloads raw FDIC data via the BankFind Suite API
│   ├── bank_failure.duckdb           # Local DuckDB database with the raw FDIC tables
│   ├── Feature_engineering.R         # Builds the target and the 16 CAMELS-based predictors
│   ├── model_dataset.csv             # Final modelling dataset (output of feature engineering)
│   └── summar_statistics.R           # Descriptive statistics and class imbalance
├── R/
│   ├── random_forest.R               # Baseline random forest (ntree = 500, mtry = 4)
│   ├── ModelTuning:Evaluation.R      # 10-fold CV / LOOCV tuning, OOB threshold, test evaluation
│   ├── logistic_regression_neural_network.R  # Benchmark models        
├── README.md
└── .gitignore
```

#List of aids for constructing our algorithm
Chat GPT: Used in the data download, summary statistics and default random forest
script to check over the code, debug and help with next steps. Most of the code was
rewritten by chat to be more efficient.
Claude: Used in the ModelEvaluation Script. Assistance in making the code more 
efficient and especially for section 6 - the tuning the 10-fold CV. Used to 
collect inspiration for future steps and interesting add-ons.
Gemini: Used to help write the code for the other machine learning methods
