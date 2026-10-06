# Download raw data and store in DuckDB

library(DBI)
library(duckdb)
library(httr2)
library(dplyr)
library(purrr)
library(tibble)
library(fdic)

source("config.R")

# 1. Connect to DuckDB

con <- dbConnect(
  duckdb(),
  "data/bank_failure.duckdb"
)


# 2. Check API Keys

FDIC_API_KEY <- Sys.getenv("FDIC_API_KEY")


if (FDIC_API_KEY == "") {
  stop("FDIC_API_KEY not found in .Renviron")
}


message("API keys found.")


# 3. FDIC Financial Data

# We want quarterly observations.
#
# Main development period:
# 2004 Q1 - 2014 Q4
#
# 2004 gives us a lag for calculating YoY growth beginning
# in 2005.
#
# External validation:
# 2021 Q1 - 2023 Q4
#
# Core variables:
#
# CERT       Bank identifier
# REPDTE     Reporting date
#
# RBC1AAJ    Tier 1 leverage ratio
# NCLNLSR    Noncurrent loans / loans
# NTLNLSR    Net charge-offs / loans
# ELNATRY    Loan-loss provisions / assets
#
# ROA        Return on assets
# NIMY       Net interest margin
# LNLSDEPR   Loans / deposits
#
# ASSET      Total assets
# EQ         Equity
# DEPUNINS   Uninsured deposits
# DEP        Total deposits
# LNRE       Real-estate loans
# LNLSNET    Net loans
# SC         Securities

# Generate actual FDIC quarter-end dates
make_quarter_ends <- function(start_year, end_year) {
  
  as.Date(
    unlist(
      lapply(start_year:end_year, function(year) {
        c(
          paste0(year, "-03-31"),
          paste0(year, "-06-30"),
          paste0(year, "-09-30"),
          paste0(year, "-12-31")
        )
      })
    )
  )
}

# Historical development sample
training_dates <- make_quarter_ends(
  2004,
  2014
)

# Modern / external validation sample
test_dates <- make_quarter_ends(
  2021,
  2023
)

# Combine periods
report_dates <- c(
  training_dates,
  test_dates
)


message(
  "Number of quarters to download: ",
  length(report_dates)
)

financial_fields <- c(
  
  # IDENTIFIERS
  "CERT",       # FDIC certificate number
  "NAME",       # Bank name
  "REPDTE",     # Report date
  
  # BANK CHARACTERISTICS
  "BKCLASS",    # Bank class
  "STALP",      # State
  
  # CAPITAL
  "RBC1AAJ",    # Tier 1 leverage ratio
  "EQ",         # Total equity
  
  # ASSET QUALITY
  "NCLNLSR",    # Noncurrent loans / total loans
  "NTLNLSR",    # Net charge-offs / total loans
  "ELNATRY",    # Loan-loss provisions / assets
  
  # EARNINGS
  "ROA",        # Return on assets
  "NIMY",       # Net interest margin
  
  # LIQUIDITY / FUNDING
  "LNLSDEPR",   # Loans / deposits
  "DEP",        # Total deposits
  "DEPUNINS",   # Uninsured deposits
  "DEPINS",     # Insured deposits            
  "BRO",        # Brokered deposits           
  "INTEXPY",    # Funding cost                
  
  # SIZE
  "ASSET",      # Total assets
  
  # LOAN EXPOSURE
  "LNRE",       # Real-estate loans
  "LNRECONS",   # Construction loans        
  "LNLSNET",    # Net loans
  "LNATRES",    # Loan-loss reserves          
  
  # SECURITIES EXPOSURE
  "SC"          # Securities
)

# Check for invalid fields

invalid_fields <- setdiff(
  financial_fields,
  fdic::fdic_financials$field
)

if (length(invalid_fields) > 0) {
  stop(
    paste(
      "Invalid FDIC financial fields:",
      paste(invalid_fields, collapse = ", ")
    )
  )
}

message("All requested FDIC financial fields are valid.")

# Download the data
download_fdic_quarter <- function(report_date) {
  
  date_string <- format(
    as.Date(report_date),
    "%Y%m%d"
  )
  
  message(
    "Downloading FDIC financials: ",
    date_string
  )
  
  result <- get_financials(
    api_key = FDIC_API_KEY,
    
    filters = paste0(
      "REPDTE:",
      date_string
    ),
    
    fields = financial_fields,
    
    sort_by = "CERT",
    
    limit = 10000
  )
  
  # Add explicit download-quarter variable
  result$requested_report_date <- as.Date(report_date)
  
  result
}


# Download every quarter
financial_list <- lapply(
  report_dates,
  download_fdic_quarter
)


# Combine into one dataframe
fdic_financials <- bind_rows(
  financial_list
)


message(
  "Financial data downloaded: ",
  format(nrow(fdic_financials), big.mark = ","),
  " rows"
)

# Basic data checks
if (nrow(fdic_financials) == 0) {
  stop("FDIC financial download returned zero rows.")
}


if (!"CERT" %in% names(fdic_financials)) {
  stop("CERT is missing from FDIC financial data.")
}


if (!"REPDTE" %in% names(fdic_financials)) {
  stop("REPDTE is missing from FDIC financial data.")
}


message(
  "Unique banks in financial data: ",
  n_distinct(fdic_financials$CERT)
)

# Write raw financials to DuckDB
dbWriteTable(
  con,
  "raw_fdic_financials",
  fdic_financials,
  overwrite = TRUE
)

message(
  "raw_fdic_financials written to DuckDB"
)


# Download FDIC Institution Data

message("Downloading FDIC institution data...")

institution_fields <- c(
  "CERT",
  "NAME",
  "BKCLASS",
  "STALP",
  "STNAME",
  "ACTIVE",
  "ESTYMD",
  "ENDEFYMD"
)

# Check requested fields
invalid_institution_fields <- setdiff(
  institution_fields,
  fdic::fdic_institutions$field
)

if (length(invalid_institution_fields) > 0) {
  stop(
    paste(
      "Invalid FDIC institution fields:",
      paste(invalid_institution_fields, collapse = ", ")
    )
  )
}

message("All requested institution fields are valid.")


# Determine CERT range required by our financial dataset
certs_needed <- sort(
  unique(
    as.integer(fdic_financials$CERT)
  )
)

certs_needed <- certs_needed[
  !is.na(certs_needed)
]

message(
  "Unique CERTs needed: ",
  length(certs_needed)
)

min_cert <- min(certs_needed)
max_cert <- max(certs_needed)

# Split approximately in half based on the observed CERTs
split_cert <- certs_needed[
  ceiling(length(certs_needed) / 2)
]

message(
  "CERT range: ",
  min_cert,
  " to ",
  max_cert
)

message(
  "Splitting institution query at CERT ",
  split_cert
)



# First half
institutions_part1 <- get_institutions(
  api_key = FDIC_API_KEY,
  
  filters = paste0(
    "CERT:[",
    min_cert,
    " TO ",
    split_cert,
    "]"
  ),
  
  fields = institution_fields,
  
  sort_by = "CERT",
  
  limit = 10000
)



# Second half
institutions_part2 <- get_institutions(
  api_key = FDIC_API_KEY,
  
  filters = paste0(
    "CERT:{",
    split_cert,
    " TO ",
    max_cert,
    "]"
  ),
  
  fields = institution_fields,
  
  sort_by = "CERT",
  
  limit = 10000
)



# Combine both API results
fdic_institutions_raw <- bind_rows(
  institutions_part1,
  institutions_part2
) %>%
  mutate(
    CERT = as.integer(CERT)
  ) %>%
  distinct(
    CERT,
    .keep_all = TRUE
  )


# Keep only banks that actually appear in our financial panel
fdic_institutions_raw <- fdic_institutions_raw %>%
  filter(
    CERT %in% certs_needed
  )


message(
  "Institution data downloaded: ",
  format(
    nrow(fdic_institutions_raw),
    big.mark = ","
  ),
  " rows"
)


# Check coverage
missing_certs <- setdiff(
  certs_needed,
  fdic_institutions_raw$CERT
)

message(
  "CERTs missing institution information: ",
  length(missing_certs)
)

if (length(missing_certs) > 0) {
  
  warning(
    paste(
      length(missing_certs),
      "banks in the financial panel have no matching institution record."
    )
  )
}


# Store in DuckDB
dbWriteTable(
  con,
  "raw_fdic_institutions",
  fdic_institutions_raw,
  overwrite = TRUE
)

message(
  "raw_fdic_institutions written to DuckDB"
)

# Download Failure Data
message(
  "Downloading FDIC failure data..."
)


fdic_failures_raw <- get_failures(
  api_key = FDIC_API_KEY,
  
  fields = c(
    "CERT",
    "NAME",
    "CITYST",
    "FAILDATE",
    "FAILYR",
    "RESTYPE"
  ),
  
  sort_by = "FAILDATE",
  
  descending = FALSE,
  
  limit = 10000
)

failure_fields <- c(
  "CERT",
  "NAME",
  "CITYST",
  "FAILDATE",
  "FAILYR",
  "RESTYPE"
)

#Check invalid failure fields

invalid_failure_fields <- setdiff(
  failure_fields,
  fdic::fdic_failures$field
)

if (length(invalid_failure_fields) > 0) {
  stop(
    paste(
      "Invalid FDIC failure fields:",
      paste(invalid_failure_fields, collapse = ", ")
    )
  )
}


message(
  "Failure data downloaded: ",
  format(
    nrow(fdic_failures_raw),
    big.mark = ","
  ),
  " rows"
)


dbWriteTable(
  con,
  "raw_fdic_failures",
  fdic_failures_raw,
  overwrite = TRUE
)


message(
  "raw_fdic_failures written to DuckDB"
)


# 11. Check Database

print(
  dbListTables(con)
)

# 12. Check Row Counts

tables <- c(
  "raw_fdic_financials",
  "raw_fdic_institutions",
  "raw_fdic_failures"
  
)

for (table in tables) {
  
  result <- dbGetQuery(
    con,
    paste0(
      "SELECT COUNT(*) AS n FROM ",
      table
    )
  )
  
  message(
    table,
    ": ",
    result$n,
    " rows"
  )
}


# 13. Disconnect

dbDisconnect(
  con,
  shutdown = TRUE
)

message("Data download complete.")
