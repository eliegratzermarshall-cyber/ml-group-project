library(DBI)
library(duckdb)

# API keys
FRED_API_KEY <- Sys.getenv("FRED_API_KEY")
FDIC_API_KEY <- Sys.getenv("FDIC_API_KEY")

# Database
DB_PATH <- "data/bank_failure.duckdb"

con <- dbConnect(
  duckdb(),
  DB_PATH
)