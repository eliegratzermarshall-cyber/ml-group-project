# ml-group-project
Group Project for the course "Machine Learning in Finance"

bank-failure-project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── bank_failure.duckdb
│
├── R/
│   ├── 01_download_fdic.R
│   ├── 02_download_fred.R
│   ├── 03_clean_fdic.R
│   ├── 04_construct_target.R
│   ├── 05_merge_dataset.R
│   └── 06_summary_statistics.R
│
├── output/
│   ├── tables/
│   └── figures/
│
└── README.md


Run the following:

install.packages("usethis")   # Only if not already installed
library(usethis)
edit_r_environ()

Add the following:
FRED_API_KEY=your_fred_api_key
FDIC_API_KEY=your_fdic_api_key

Save the file: Cmd + S

Restart R

Run the following to make sure it's saved:
FRED_API_KEY=your_fred_api_key
FDIC_API_KEY=your_fdic_api_key
