"""00_clean_NGFS_IAM.py — reproduce data/raw/NGFS_cleaned_IAM_all.csv from the raw NGFS download.

NGFS scenario data is licensed and must be downloaded by the user
(https://data.ece.iiasa.ac.at/ngfs). Place the IAM-output workbook at
    data/NGFS_raw/IAM_data.xlsx      (sheet "data", IAMC wide format)
and run this script. It extracts the AR6-climate-diagnostics *Infilled* emissions — the
complete, FaIR-ready 52-species set (the IAM-reported species plus the infilled
Montreal-Protocol/minor gases) — for the World region and the three NGFS IAMs, renames the
IAMC variable names to the short forms used downstream, and writes
    data/raw/NGFS_cleaned_IAM_all.csv
which is the input to 01_clean_and_extend_NGFS.py.

Requires openpyxl (see environment.yml). Light — runs on the login node.
"""
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)

RAW_XLSX = "data/NGFS_raw/IAM_data.xlsx"
OUT_CSV  = "data/raw/NGFS_cleaned_IAM_all.csv"
SHEET    = "data"
# the three NGFS IAMs (drop the "…IntegratedPhysicalDamages (median)" variant)
MODELS = [
    "GCAM 6.0 NGFS",
    "MESSAGEix-GLOBIOM 2.0-M-R12-NGFS",
    "REMIND-MAgPIE 3.3-4.8",
]
# AR6 "Infilled" is the complete emulator-ready species set (harmonized + infilled)
PREFIX = "AR6 climate diagnostics|Infilled|Emissions|"


def _short(variable):
    """IAMC variable -> the short species name used in the pipeline.
    Last '|' segment, except CO2 fossil/industry which becomes 'FFI'
    (CO2|AFOLU already reduces to 'AFOLU')."""
    if variable.endswith("CO2|Energy and Industrial Processes"):
        return "FFI"
    return variable.split("|")[-1].strip()


def main():
    if not os.path.exists(RAW_XLSX):
        sys.exit(f"Raw NGFS workbook not found at {RAW_XLSX}. Download the NGFS IAM outputs "
                 f"(https://data.ece.iiasa.ac.at/ngfs) and place it there (sheet '{SHEET}').")
    raw = pd.read_excel(RAW_XLSX, sheet_name=SHEET)

    df = raw[
        (raw["Region"] == "World")
        & (raw["Model"].isin(MODELS))
        & (raw["Variable"].astype(str).str.startswith(PREFIX))
    ].copy()
    df["Variable"] = df["Variable"].map(_short)
    # strip the degree sign that appears in some scenario names ("Below 2°C" -> "Below 2C")
    df["Scenario"] = df["Scenario"].astype(str).str.replace("°", "", regex=False)

    year_cols = [c for c in df.columns if str(c).isdigit()]
    df = df[["Model", "Scenario", "Region", "Variable", "Unit"] + year_cols]
    df = df.sort_values(["Model", "Scenario", "Variable"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}: {df.shape[0]} rows, "
          f"{df['Model'].nunique()} models × {df['Scenario'].nunique()} scenarios × "
          f"{df['Variable'].nunique()} species", flush=True)


if __name__ == "__main__":
    main()
