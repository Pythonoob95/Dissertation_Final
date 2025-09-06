import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR    = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LSEG_AUDIT_DIR = DATA_DIR / "LSEG_data"
LSEG_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

LSEG_EQ_BOND_CSV = LSEG_AUDIT_DIR / "equity_bond_daily_prices.csv"

LSEG_OFFLINE_ONLY  = bool(int(os.getenv("LSEG_OFFLINE_ONLY", "0")))
LSEG_FORCE_REFRESH = bool(int(os.getenv("LSEG_FORCE_REFRESH", "0")))

BTOP50_INPUT_PATH    = DATA_DIR / "BTOP50_Index_historical_data(5).xls"
BTOP50_INPUT_PATH_V6 = DATA_DIR / "BTOP50_Index_historical_data(6).xlsx"

USA_BOND_EXCEL_PATH = DATA_DIR / "US_Bonds.xlsx"

SG_CTA_INPUT_PATH = DATA_DIR / "SG_CTA_Indexes_Correct.xlsx"
SG_CTA_ALT_PATH   = DATA_DIR / "SG CTA Indexes Correct.xlsx"

TBILL_INPUT_PATH   = DATA_DIR / "US3MT=RR.xlsx"
SG_TREND_INPUT_DIR = DATA_DIR / "Top-Down Update"
FUTURES_INPUT_DIR  = DATA_DIR / "futures"

PANEL_OUTPUT_DIR = DATA_DIR / "topdown_panel"
PANEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EQUITY_RIC      = ".MIWO00000PUS"
BOND_RIC        = ".FTWGBIUSDTH"
USA_EQUITY_RIC  = ".MIUS00000PUS"
COMMODITY_RIC   = ".BCOMTR"
DOLLAR_RIC      = ".DXY"

LSEG_DATA_FIELD       = "TRDPRC_1"
LSEG_FETCH_START_DATE = "1987-01-01"

TRADITIONAL_PORTFOLIO_ALLOCATION = {"Equities": 0.50, "Bonds": 0.50}
TRADITIONAL_PORTFOLIO_NAME       = "50/50 Portfolio"

ENHANCED_PORTFOLIO_ALLOCATION = {"Equities": 0.50, "Bonds": 0.25, "BTOP50": 0.25}
ENHANCED_PORTFOLIO_NAME       = "50/25/25 Portfolio (Eq/Bd/CTA)"

CRISIS_PERIODS = {
    "Black Monday":             ["1987-10-01", "1987-12-31"],
    "Japanese Bubble Collapse": ["1990-01-01", "1992-12-31"],
    "Asian Financial Crisis":   ["1997-07-01", "1998-12-31"],
    "Russian Crisis & LTCM":    ["1998-08-01", "1998-12-31"],
    "Dot-com Bubble Burst":     ["2000-03-01", "2002-10-31"],
    "Global Financial Crisis":  ["2007-08-01", "2009-03-31"],
    "COVID-19 Crash":           ["2020-02-01", "2020-04-30"],
    "2022 Inflation & Rate Hikes": ["2022-01-01", "2022-12-31"],
}

BTOP50_COLOR                = "#000080"
TRADITIONAL_PORTFOLIO_COLOR = "#00796B"
ENHANCED_PORTFOLIO_COLOR    = "#FFC107"

FACTOR_COLORS = {
    "Trend_Equities":     "#1f77b4",
    "Trend_Commodities":  "#ff7f0e",
    "Trend_Bonds":        "#2ca02c",
    "Trend_FX":           "#d62728",
    "Carry_FX":           "#9467bd",
    "Carry_Bond":         "#8c564b",
    "CS_Mom_Equities":    "#e377c2",
    "CS_Mom_Commodities": "#7f7f7f",
    "Reversal_ShortTerm": "#bcbd22",
    "Curve_Yield":        "#17becf",
    "Curve_Commodity":    "#aec7e8",
    "Volatility_Short":   "#ffbb78",
    "USD_Factor":         "#98df8a",
}

SG_TREND_COLORS = {
    "small":     "#9b59b6",
    "large":     "#e67e22",
    "blended":   "#2c3e50",
    "benchmark": "#3498db",
    "crisis":    "#e74c3c",
    "normal":    "#27ae60",
    "neutral":   "#95a5a6",
}

FACTOR_ANALYSIS_OUTPUT_DIR = DATA_DIR / "factor_analysis_results"
FACTOR_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SG_TREND_OUTPUT_DIR = DATA_DIR / "topdown_replication_combo"
SG_TREND_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
