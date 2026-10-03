# ============================================================
# 02_feature_engineering.R
# Baut aus den Rohdaten das fertige Modell-Dataset:
# eine Zeile pro Bank und Quartal, mit Y und 16 Features.
# ============================================================

library(DBI)
library(duckdb)
library(dplyr)
library(lubridate)

# ---- Einstellungen -----------------------------------------
COUNT_ASSISTANCE_AS_FAILURE <- TRUE        # gerettete Banken als Ausfall zählen?
TRAIN_END <- as.Date("2014-12-31")         # bis hier Training, danach Test


# ---- 1. Rohdaten aus DuckDB laden --------------------------
con <- dbConnect(duckdb(), "data/bank_failure.duckdb")

financials   <- dbReadTable(con, "raw_fdic_financials")
institutions <- dbReadTable(con, "raw_fdic_institutions")
failures     <- dbReadTable(con, "raw_fdic_failures")


# ---- 2. Hilfsfunktion: Datum einlesen ----------------------
# Die FDIC liefert Daten in verschiedenen Schreibweisen
# (z.B. "1/25/2008" oder "20071231"). Diese Funktion versteht beide.
to_date <- function(x) {
  if (inherits(x, "Date")) return(x)
  as.Date(parse_date_time(as.character(x), orders = c("mdy", "ymd"), quiet = TRUE))
}


# ---- 3. Failures vorbereiten (Grundlage für Y) -------------
failures_clean <- failures %>%
  mutate(
    CERT      = as.integer(CERT),
    fail_date = to_date(FAILDATE)
  ) %>%
  filter(COUNT_ASSISTANCE_AS_FAILURE | RESTYPE == "FAILURE") %>%
  group_by(CERT) %>%
  summarise(fail_date = min(fail_date), .groups = "drop")   # eine Zeile pro Bank


# ---- 4. Institutions vorbereiten (für das Alter) -----------
institutions_clean <- institutions %>%
  mutate(
    CERT        = as.integer(CERT),
    established = to_date(ESTYMD)
  ) %>%
  select(CERT, established) %>%
  distinct(CERT, .keep_all = TRUE)


# ---- 5. Financials vorbereiten -----------------------------
# Alle Zahlenspalten sicher in Zahlen umwandeln
number_cols <- c("ASSET", "DEP", "DEPINS", "EQ", "LNLSNET", "LNRE", "LNRECONS",
                 "LNATRES", "BRO", "SC", "RBC1AAJ", "NCLNLSR", "ROA", "NIMY",
                 "LNLSDEPR", "INTEXPY")

panel <- financials %>%
  mutate(
    CERT        = as.integer(CERT),
    report_date = as.Date(requested_report_date),
    across(all_of(number_cols), as.numeric)
  ) %>%
  # nur Geschäftsbanken
  filter(BKCLASS %in% c("N", "NM", "SM")) %>%
  # keine Spezialinstitute ohne Kredite oder Einlagen
  filter(ASSET > 0, LNLSNET > 0, DEP > 0)


# ---- 6. Tabellen zusammenführen ----------------------------
panel <- panel %>%
  left_join(failures_clean,     by = "CERT") %>%
  left_join(institutions_clean, by = "CERT")


# ---- 7. Y bauen: Ausfall innerhalb von 12 Monaten ----------
panel <- panel %>%
  mutate(
    failed_12m = if_else(
      !is.na(fail_date) &
        fail_date >  report_date &
        fail_date <= report_date %m+% months(12),
      1L, 0L
    )
  ) %>%
  # Beobachtungen nach dem Ausfall entfernen
  filter(is.na(fail_date) | report_date < fail_date)


# ---- 8. Die 16 Features bauen ------------------------------
panel <- panel %>%
  mutate(
    # Kapital
    equity_ratio       = EQ / ASSET,
    tier1_leverage     = RBC1AAJ,
    # Kreditqualität
    npl_ratio          = NCLNLSR,
    reserve_ratio      = LNATRES / LNLSNET,
    re_loan_share      = LNRE / LNLSNET,
    construction_share = LNRECONS / LNLSNET,
    # Management
    bank_age           = as.numeric(report_date - established) / 365.25,
    # Ertrag
    roa                = ROA,
    nim                = NIMY,
    # Liquidität und Finanzierung
    loans_to_assets    = LNLSNET / ASSET,
    loans_to_deposits  = LNLSDEPR,
    brokered_share     = BRO / DEP,
    uninsured_share    = pmin(pmax(1 - DEPINS / DEP, 0), 1),
    funding_cost       = INTEXPY,
    # Zinsrisiko
    securities_share   = SC / ASSET,
    # Grösse
    log_assets         = log(ASSET)
  )

feature_cols <- c("equity_ratio", "tier1_leverage", "npl_ratio", "reserve_ratio",
                  "re_loan_share", "construction_share", "bank_age", "roa", "nim",
                  "loans_to_assets", "loans_to_deposits", "brokered_share",
                  "uninsured_share", "funding_cost", "securities_share", "log_assets")


# ---- 9. Training und Test markieren ------------------------
panel <- panel %>%
  mutate(sample = if_else(report_date <= TRAIN_END, "train", "test"))


# ---- 10. Fehlende Werte ------------------------------------
message("Fehlende Werte pro Feature:")
print(colSums(is.na(panel[, feature_cols])))

rows_before <- nrow(panel)
panel <- panel %>% filter(if_all(all_of(feature_cols), ~ !is.na(.x)))
message("Entfernte Zeilen wegen fehlender Werte: ", rows_before - nrow(panel))


# ---- 11. Extreme Ausreisser kappen (Winsorizing) -----------
# Grenzen = 1. und 99. Perzentil, berechnet NUR aus den Trainingsdaten
ratio_cols <- setdiff(feature_cols, c("bank_age", "log_assets", "uninsured_share"))

for (col in ratio_cols) {
  limits <- quantile(panel[[col]][panel$sample == "train"], c(0.01, 0.99))
  panel[[col]] <- pmin(pmax(panel[[col]], limits[1]), limits[2])
}


# ---- 12. Nur die benötigten Spalten behalten ---------------
model_data <- panel %>%
  select(CERT, NAME, report_date, sample, failed_12m, all_of(feature_cols)) %>%
  arrange(report_date, CERT)


# ---- 13. Kontrollen ----------------------------------------
message("Zeilen im fertigen Dataset: ", format(nrow(model_data), big.mark = "'"))

message("Ausfälle pro Jahr:")
print(
  model_data %>%
    group_by(year = year(report_date), sample) %>%
    summarise(banks_quarters = n(), failures = sum(failed_12m), .groups = "drop")
)

print(summary(model_data[, feature_cols]))


# ---- 14. Speichern -----------------------------------------
dbWriteTable(con, "model_dataset", model_data, overwrite = TRUE)
write.csv(model_data, "data/model_dataset.csv", row.names = FALSE)

dbDisconnect(con, shutdown = TRUE)
message("Feature engineering abgeschlossen.")