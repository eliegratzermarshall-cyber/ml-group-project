#Summary Statistics

library(DBI)
library(duckdb)
library(dplyr)
library(ggplot2)
library(readr)
library(psych)
source("config.R")

#Create output folder
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

# Connect to database
con <- dbConnect(
  duckdb(),
  "data/bank_failure.duckdb",
)


# LOAD FINAL MODEL DATASET
model_data <- dbReadTable(
  con,
  "model_dataset"
)

# BASIC INFORMATION
cat("\n")
cat("BASIC DATA INFORMATION\n")

cat("Observations:", nrow(model_data), "\n")

cat("Variables:", ncol(model_data), "\n")

cat("Unique banks:",
    n_distinct(model_data$CERT),
    "\n")

cat("Unique quarters:",
    n_distinct(model_data$REPDTE),
    "\n")

# CLASS BALANCE
cat("\n")
cat("FAILURE DISTRIBUTION\n")

table(model_data$failed_12m)

prop.table(
  table(model_data$failed_12m)
)

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


# NUMERIC VARIABLES
numeric_data <- model_data %>%
  select(all_of(feature_cols))


# SUMMARY STATISTICS
summary_stats <-
  
  psych::describe(
    numeric_data
  )

summary_stats <-
  
  summary_stats |>
  
  select(
    
    mean,
    sd,
    median,
    min,
    max,
    n
    
  )

print(summary_stats)

write_csv(
  summary_stats,
  "output/summary_statistics.csv"
)


# CORRELATION MATRIX

correlation_matrix <-
  
  cor(
    numeric_data,
    use = "pairwise.complete.obs"
  )

write.csv(
  correlation_matrix,
  "output/correlation_matrix.csv"
)

# HISTOGRAMS

dir.create(
  "output/plots",
  showWarnings = FALSE,
  recursive = TRUE
)

for(variable in names(numeric_data)){
  
  p <-
    
    ggplot(
      
      model_data,
      
      aes_string(x = variable)
      
    )+
    
    geom_histogram(
      bins = 30
    )+
    
    theme_minimal()
  
  ggsave(
    
    paste0(
      "output/plots/",
      variable,
      "_histogram.png"
    ),
    
    p,
    
    width = 6,
    height = 4
    
  )
  
}


# BOXPLOTS

for(variable in names(numeric_data)){
  
  p <-
    
    ggplot(
      
      model_data,
      
      aes_string(
        
        y = variable
        
      )
      
    )+
    
    geom_boxplot()
  
  ggsave(
    
    paste0(
      "output/plots/",
      variable,
      "_boxplot.png"
    ),
    
    p,
    
    width = 4,
    height = 6
    
  )
  
}


# SUMMARY BY FAILURE STATUS

summary_by_failure <-
  
  model_data |>
  
  group_by(
    failed_12m
  ) |>
  
  summarise(
    
    across(
      
      where(is.numeric),
      
      list(
        
        mean = mean,
        
        sd = sd
        
      ),
      
      na.rm = TRUE
      
    )
    
  )

write_csv(
  
  summary_by_failure,
  
  "output/summary_by_failure.csv"
  
)


# CLOSE CONNECTION

dbDisconnect(
  con,
  shutdown = TRUE
)

cat("\nSummary statistics complete.\n")