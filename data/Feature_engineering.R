
# 02_feature_engineering.R
# Builds the final model dataset



library(DBI)
library(duckdb)
library(dplyr)
library(lubridate)

# Settings
COUNT_ASSISTANCE_AS_FAILURE <- TRUE        # Assisted banks also count as failure
TRAIN_END <- as.Date("2014-12-31")         # Data is trained until this date, later testing


#  1. load raw data from DuckDB
con <- dbConnect(duckdb(), "data/bank_failure.duckdb")

financials   <- dbReadTable(con, "raw_fdic_financials")
institutions <- dbReadTable(con, "raw_fdic_institutions")
failures     <- dbReadTable(con, "raw_fdic_failures")


# 2. Help Function to read dates 

to_date <- function(x) {
  if (inherits(x, "Date")) return(x)
  as.Date(parse_date_time(as.character(x), orders = c("mdy", "ymd"), quiet = TRUE))
}


# 3. Prepare failures
failures_clean <- failures %>%
  mutate(
    CERT      = as.integer(CERT),
    fail_date = to_date(FAILDATE)
  ) %>%
  filter(COUNT_ASSISTANCE_AS_FAILURE | RESTYPE == "FAILURE") %>%
  group_by(CERT) %>%
  summarise(fail_date = min(fail_date), .groups = "drop")   
#one row for each bank

# 4. Prepare institutions for the bank age feature
institutions_clean <- institutions %>%
  mutate(
    CERT        = as.integer(CERT),
    established = to_date(ESTYMD)
  ) %>%
  select(CERT, established) %>%
  distinct(CERT, .keep_all = TRUE)


# Prepare Financials --> all columns are converted into numbers
number_cols <- c("ASSET", "DEP", "DEPINS", "EQ", "LNLSNET", "LNRE", "LNRECONS",
                 "LNATRES", "BRO", "SC", "RBC1AAJ", "NCLNLSR", "ROA", "NIMY",
                 "LNLSDEPR", "INTEXPY")

panel <- financials %>%
  mutate(
    CERT        = as.integer(CERT),
    report_date = as.Date(requested_report_date),
    across(all_of(number_cols), as.numeric)
  ) %>%
  
  filter(BKCLASS %in% c("N", "NM", "SM")) %>% #keep only commercial banks
  
  filter(ASSET > 0, LNLSNET > 0, DEP > 0) #No institutions without loans or deposits


# 6. Merging the tables
panel <- panel %>%
  left_join(failures_clean,     by = "CERT") %>%
  left_join(institutions_clean, by = "CERT")


# 7. build Y (failed_12m)
panel <- panel %>%
  mutate(
    failed_12m = if_else(
      !is.na(fail_date) &
        fail_date >  report_date &
        fail_date <= report_date %m+% months(12),
      1L, 0L
    )
  ) %>%
  
  filter(is.na(fail_date) | report_date < fail_date) #Drop observations after a banks failure


# 8. build features
panel <- panel %>%
  mutate(
    # Capital features
    equity_ratio       = EQ / ASSET,
    tier1_leverage     = RBC1AAJ,
    # Asset Quality features
    npl_ratio          = NCLNLSR,
    reserve_ratio      = LNATRES / LNLSNET,
    re_loan_share      = LNRE / LNLSNET,
    construction_share = LNRECONS / LNLSNET,
    # Management features
    bank_age           = as.numeric(report_date - established) / 365.25,
    # Earnings features
    roa                = ROA,
    nim                = NIMY,
    # Liquidity and funding features
    loans_to_assets    = LNLSNET / ASSET,
    loans_to_deposits  = LNLSDEPR,
    brokered_share     = BRO / DEP,
    uninsured_share    = pmin(pmax(1 - DEPINS / DEP, 0), 1),
    funding_cost       = INTEXPY,
    # Sensitivity to market risk features
    securities_share   = SC / ASSET,
    # bank size feature
    log_assets         = log(ASSET)
  )

feature_cols <- c("equity_ratio", "tier1_leverage", "npl_ratio", "reserve_ratio",
                  "re_loan_share", "construction_share", "bank_age", "roa", "nim",
                  "loans_to_assets", "loans_to_deposits", "brokered_share",
                  "uninsured_share", "funding_cost", "securities_share", "log_assets")


# 9. Mark train and test sample
panel <- panel %>%
  mutate(sample = if_else(report_date <= TRAIN_END, "train", "test"))


# 10. Dropping missing Values
message("Missing values per feature:")
print(colSums(is.na(panel[, feature_cols])))

rows_before <- nrow(panel)
panel <- panel %>% filter(if_all(all_of(feature_cols), ~ !is.na(.x)))
message("Rows removed due to missing values: ", rows_before - nrow(panel))


# 11. Capping the extreme outliers (1st and 99th percentile)
ratio_cols <- setdiff(feature_cols, c("bank_age", "log_assets", "uninsured_share"))

for (col in ratio_cols) {
  limits <- quantile(panel[[col]][panel$sample == "train"], c(0.01, 0.99))
  panel[[col]] <- pmin(pmax(panel[[col]], limits[1]), limits[2])
}


# 12. keep only relevant/required columns
model_data <- panel %>%
  select(CERT, NAME, report_date, sample, failed_12m, all_of(feature_cols)) %>%
  arrange(report_date, CERT)


# Controls/checks
message("Rows in final dataset: ", format(nrow(model_data), big.mark = "'"))

message("Failures per year:")
print(
  model_data %>%
    group_by(year = year(report_date), sample) %>%
    summarise(banks_quarters = n(), failures = sum(failed_12m), .groups = "drop")
)

print(summary(model_data[, feature_cols]))


# Save
dbWriteTable(con, "model_dataset", model_data, overwrite = TRUE)
write.csv(model_data, "data/model_dataset.csv", row.names = FALSE)

dbDisconnect(con, shutdown = TRUE)
message("Feature engineering complete.")