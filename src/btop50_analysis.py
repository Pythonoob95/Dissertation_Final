from __future__ import annotations

import os
import gzip
import pickle
import logging
import warnings
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
from pandas.tseries.offsets import BDay
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("btop50_returns_based")

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 100, 'savefig.dpi': 300,
    'font.size': 10, 'font.family': 'serif',
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.grid': True, 'grid.alpha': 0.30, 'lines.linewidth': 1.5,
    'figure.facecolor': 'white', 'axes.facecolor': 'white'
})

BTOP50_EXCEL_PATH = Path(
    os.getenv("BTOP50_XLSX", r"data/BTOP50_Index_historical_data(6).xlsx")
)
US3MT_EXCEL_PATH = Path(
    os.getenv("US3MT_XLSX", r"data/US3MT=RR.xlsx")
)

def _resolve_futures_input_dir() -> Path:
    env_dir = os.getenv("FUTURES_INPUT_DIR")
    cfg_dir = None
    try:
        from . import config as _cfg
        cfg_dir = getattr(_cfg, "FUTURES_INPUT_DIR", None)
    except Exception:
        try:
            import config as _cfg
            cfg_dir = getattr(_cfg, "FUTURES_INPUT_DIR", None)
        except Exception:
            pass
    base = env_dir or cfg_dir or "futures"
    return Path(base).expanduser()

FUTURES_INPUT_DIR = _resolve_futures_input_dir()

FIGURES_DIR = Path("figures")
VIZ_DIR = FIGURES_DIR / "btop50_visualizations"
RB_DIR = VIZ_DIR / "returns_based_results"
RB_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR            = RB_DIR / "cache"
PANEL_OUTPUT_DIR     = RB_DIR / "panel"
WEIGHT_HISTORY_DIR   = RB_DIR / "weight_history"
AUDIT_DIR            = RB_DIR / "audit_trail"
ARCHIVE_DIR          = RB_DIR / "audit_archive"
OUTPUT_DIR           = RB_DIR / "outputs"
ADJUSTED_OUT_DIR     = RB_DIR / "adjusted_prices_csv"

for d in [CACHE_DIR, PANEL_OUTPUT_DIR, WEIGHT_HISTORY_DIR, AUDIT_DIR, ARCHIVE_DIR, OUTPUT_DIR, ADJUSTED_OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SHOW_PLOTS = os.getenv("SHOW_PLOTS", "1").strip() not in {"0", "false", "False", ""}

N_JOBS = int(os.getenv("N_JOBS", min(4, os.cpu_count() or 1)))
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False
    logger.info("joblib not available — alpha tuning will run sequentially")

ANN_BDAYS = 252
FILTER_START = pd.Timestamp("2000-01-03", tz="UTC")
FILTER_END   = pd.Timestamp("2024-03-29", tz="UTC")
OOS_START_DATE = "2021-11-01"

SUFFIX = "returns_based"

_DEFAULT_COLORS = {
    'large': '#2ca02c',
    'model': '#d62728',
    'benchmark': '#9467bd',
    'normal': '#2c3e50',
    'crisis': '#e74c3c',
}
COLOR_PALETTE = dict(_DEFAULT_COLORS)

AGS_DROP = {
    "CORN","SOYBEANS","WHEAT","SUGAR","COFFEE","COCOA","COTTON",
    "ZC","ZS","ZW","C","S","W","SB","KC","CC","CT","KW","CORN_MINI","WHEAT_MINI"
}

REGIME_COLORS = {
    'strong_bear': '#8b0000',
    'bear': '#dc143c',
    'weak_bear': '#ff6347',
    'neutral': '#ffd700',
    'weak_bull': '#90ee90',
    'bull': '#32cd32',
    'strong_bull': '#006400'
}

CRISIS_PERIODS = [
    ('2007-07-01', '2009-03-31', 'GFC'),
    ('2011-05-01', '2011-10-31', 'EU Debt'),
    ('2015-08-01', '2016-02-29', 'China/Oil'),
    ('2020-02-15', '2020-04-30', 'COVID-19'),
    ('2022-02-15', '2022-10-31', 'Ukraine/Inflation'),
]

def savefig_and_maybe_show(fig, outpath: Path, *, show: bool = SHOW_PLOTS, dpi: int = 300):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    if show:
        try:
            fig.canvas.manager.set_window_title(outpath.name)
        except Exception:
            pass
        try:
            plt.show(block=False)
        except Exception:
            plt.show()
    else:
        plt.close(fig)
    return outpath

def to_utc(obj):
    return pd.to_datetime(obj, errors="coerce", utc=True)

def _index_to_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if getattr(idx, "tz", None) is None:
        return idx
    return idx.tz_convert("UTC").tz_localize(None)

def to_naive(ts) -> pd.Timestamp | None:
    if ts is None:
        return None
    t = pd.to_datetime(ts, errors="coerce")
    if t is pd.NaT:
        return None
    if getattr(t, "tzinfo", None) is None:
        return t
    return t.tz_convert("UTC").tz_localize(None)

def _as_naive_dt(x):
    ts = pd.to_datetime(x, errors="coerce")
    if ts is pd.NaT:
        return ts
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert("UTC").tz_localize(None)

def annotate_crisis_periods(ax, y_position: str = 'top'):
    xlim = ax.get_xlim()
    for start, end, label in CRISIS_PERIODS:
        sdt = _as_naive_dt(start)
        edt = _as_naive_dt(end)
        if sdt is pd.NaT or edt is pd.NaT:
            continue
        sn, en = mdates.date2num(sdt), mdates.date2num(edt)
        if en < xlim[0] or sn > xlim[1]:
            continue
        ax.axvspan(sdt, edt, alpha=0.15, color=COLOR_PALETTE['crisis'], zorder=0)
        ylim = ax.get_ylim()
        y_pos = ylim[1] * 0.95 if y_position == 'top' else ylim[0] * 1.05
        mid = sdt + (edt - sdt) / 2
        ax.text(mid, y_pos, label, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

def ema_update(prev: pd.Series, new: pd.Series, alpha: float) -> pd.Series:
    return alpha * new + (1 - alpha) * prev

def trailing_vol(series: pd.Series, lookback: int = 252) -> float:
    if len(series) < max(10, lookback // 4):
        return np.nan
    return float(series.rolling(lookback).std().iloc[-1] * np.sqrt(252))

def classify_regime(rolling_sharpe: float) -> str:
    if rolling_sharpe < -2:   return 'strong_bear'
    if rolling_sharpe < -1:   return 'bear'
    if rolling_sharpe < -0.5: return 'weak_bear'
    if rolling_sharpe < 0.5:  return 'neutral'
    if rolling_sharpe < 1:    return 'weak_bull'
    if rolling_sharpe < 2:    return 'bull'
    return 'strong_bull'

@dataclass
class RiskConfig:
    instrument_vol_target: float = 0.12
    portfolio_vol_target: float = 0.085
    max_leverage_cap: float = 1.0
    vol_halflife: int = 40

DEFAULT_RISK_CONFIG = RiskConfig()

def rescale_to_target(
    ret: pd.Series,
    *,
    half_life: int | None = None,
    ann_bdays: int = ANN_BDAYS,
    target: float | None = None,
    scale_cap: float | None = None,
    risk_config: RiskConfig | None = None
) -> pd.Series:
    if risk_config is None:
        risk_config = DEFAULT_RISK_CONFIG
    if half_life is None:
        half_life = risk_config.vol_halflife
    if target is None:
        target = risk_config.instrument_vol_target
    if scale_cap is None:
        scale_cap = risk_config.max_leverage_cap
    ret = pd.to_numeric(ret, errors="coerce").astype("float64")
    if ret.isna().all():
        return ret
    sigma = ret.ewm(halflife=half_life, adjust=False).std().shift(1) * np.sqrt(ann_bdays)
    eps = max(np.finfo(float).eps, 1e-12)
    sigma_clipped = sigma.clip(lower=eps)
    scale_factor = (target / sigma_clipped).clip(upper=scale_cap)
    scaled = (scale_factor * ret).astype("float64")
    return scaled

def load_rf(cache: Path | None = None, src: Path | None = None, first_date: str = "1990-01-02") -> pd.Series:
    if cache is None:
        cache = CACHE_DIR / "tbill_daily.parquet"
    if src is None:
        src = US3MT_EXCEL_PATH
    if cache.exists():
        out = pd.read_parquet(cache)["rf_daily"]
        if getattr(out.index, "tz", None) is None:
            out.index = out.index.tz_localize("UTC")
        return out
    book = pd.ExcelFile(src)
    sheet = None
    for s in book.sheet_names:
        if "table" in s.lower():
            sheet = s; break
    sheet = sheet or book.sheet_names[0]
    df = pd.read_excel(book, sheet_name=sheet)
    date_col = next((c for c in df.columns if str(c).strip().lower() in {"date","dt","observation date","obs date"}), None)
    if date_col is None:
        for c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().mean() > 0.6:
                date_col = c; break
    yield_candidates = [c for c in df.columns if any(x in str(c).lower() for x in ["yield","bid","last","px_last","close"])]
    if not yield_candidates:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError("Cannot locate yield column in US3MT file.")
        ycol = num_cols[0]
    else:
        ycol = yield_candidates[0]
    tb = (df[[date_col, ycol]].rename(columns={date_col: "Date", ycol: "y"}).dropna())
    tb["y"] = pd.to_numeric(tb["y"], errors="coerce")
    if tb["y"].median() > 1:
        tb["y"] = tb["y"] / 100.0
    tb["y"] = tb["y"] / 360.0
    tb = tb.dropna(subset=["Date"]).copy()
    tb["Date"] = to_utc(tb["Date"])
    tb = tb.set_index("Date").sort_index()
    first_date_tz = pd.Timestamp(first_date, tz="UTC")
    end_date = pd.Timestamp.now(tz="UTC").floor("D")
    full_range = pd.date_range(first_date_tz, end_date, freq="B", tz="UTC")
    tb = tb.reindex(full_range).ffill().bfill().rename(columns={"y": "rf_daily"}).astype("float32")
    tb.to_parquet(cache, compression="snappy")
    return tb["rf_daily"]

def _pick_date_col(df: pd.DataFrame) -> str | None:
    candidates = ["Date", "DATE", "Dstamp", "Observation Date", "Obs Date"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce")
        if s.notna().mean() > 0.80:
            return c
    return None

def _pick_btop50_value(df: pd.DataFrame) -> tuple[str, str] | None:
    return_names = ["Daily Return", "Return", "Returns", "ROR", "BTOP50 Return", "BTOP50 Returns", "TR", "RET"]
    level_names  = ["Index", "Index Level", "BTOP50 Index", "Level", "Last Price", "PX_LAST", "Close", "Value", "Price", "100"]
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for nm in return_names:
        if nm.lower() in lower_map:
            return ("return", lower_map[nm.lower()])
    for nm in level_names:
        if nm.lower() in lower_map:
            return ("level", lower_map[nm.lower()])
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return None
    medabs = df[num_cols].abs().median().sort_values()
    col = medabs.index[0]
    kind = "return" if float(medabs.iloc[0]) < 0.20 else "level"
    return (kind, col)

def process_btop50_data() -> pd.Series:
    rf_daily = load_rf()
    book = pd.ExcelFile(BTOP50_EXCEL_PATH)
    excess_map: dict[str, pd.Series] = {}
    for sheet in book.sheet_names:
        df = pd.read_excel(book, sheet_name=sheet)
        if df.empty:
            continue
        date_col = _pick_date_col(df)
        if not date_col:
            continue
        kind_col = _pick_btop50_value(df)
        if not kind_col:
            continue
        kind, val_col = kind_col
        tmp = df[[date_col, val_col]].dropna()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col]).copy()
        tmp["DateUTC"] = to_utc(tmp[date_col])
        tmp = (tmp.set_index("DateUTC")
               .sort_index()
               .asfreq("B")
               .ffill(limit=1)
               .bfill(limit=1))
        if kind == "return":
            s = pd.to_numeric(tmp[val_col], errors="coerce")
            if s.abs().median() > 0.20:
                s = s / 100.0
            ret = s
        else:
            lvl = pd.to_numeric(tmp[val_col], errors="coerce")
            ret = lvl.pct_change()
        mu, sigma = ret.mean(), ret.std()
        ret = ret.clip(lower=mu - 5 * sigma, upper=mu + 5 * sigma)
        idx = ret.index
        rf_al = rf_daily.reindex(idx).ffill().bfill()
        excess_daily = (ret - rf_al).astype("float32")
        excess_daily = excess_daily.loc[FILTER_START:FILTER_END]
        slug = sheet.replace(" ", "_")
        excess_map[slug] = excess_daily
        print(f"{sheet:<24}| rows {excess_daily.count():6,d}")
    if not excess_map:
        raise ValueError(
            f"Could not parse any usable sheet from '{BTOP50_EXCEL_PATH.name}'. "
            "Ensure there is a date column and either a Return or Index/Level column."
        )
    key = next((k for k in excess_map if "BTOP" in k.upper()), list(excess_map.keys())[0])
    y_excess = excess_map[key].dropna()
    btop_trading_dates = y_excess.index
    pd.to_pickle(btop_trading_dates, CACHE_DIR / "btop50_calendar.pkl")
    pd.to_pickle(y_excess, CACHE_DIR / "y_excess_BTOP50.pkl")
    print(f"y_excess_BTOP50 written — {y_excess.count():,d} days  →  {CACHE_DIR/'y_excess_BTOP50.pkl'}")
    return y_excess

def audit_btop50_excess(
    y_excess: pd.Series,
    *,
    save_dir: Path = RB_DIR,
    show_plots: bool = SHOW_PLOTS,
    label: str = "BTOP50 (excess)"
) -> pd.DataFrame:
    save_dir.mkdir(parents=True, exist_ok=True)
    s = y_excess.dropna()
    ann = np.sqrt(ANN_BDAYS)
    stats_row = {
        "count": int(s.shape[0]),
        "start": s.index.min(),
        "end": s.index.max(),
        "mean_daily": float(s.mean()),
        "ann_return": float(s.mean() * ANN_BDAYS),
        "daily_vol": float(s.std()),
        "ann_vol": float(s.std() * ann),
        "skew": float(s.skew()),
        "excess_kurtosis": float(s.kurtosis()),
        "min": float(s.min()),
        "p01": float(s.quantile(0.01)),
        "p05": float(s.quantile(0.05)),
        "median": float(s.median()),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "hit_rate": float((s > 0).mean()),
        "autocorr_1d": float(s.autocorr(lag=1)),
        "autocorr_5d": float(s.autocorr(lag=5)),
    }
    s.to_csv(save_dir / "btop50_excess_daily.csv", float_format="%.8f")
    pd.DataFrame([stats_row]).to_csv(save_dir / "btop50_excess_summary.csv", index=False)
    try:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(_index_to_naive(s.index), s.values, lw=0.8)
        ax.set_title(f"{label}: Daily Excess Returns"); ax.set_ylabel("Daily return")
        ax.grid(True, alpha=0.3)
        savefig_and_maybe_show(fig, save_dir / "btop50_excess_daily_series.png", show=show_plots)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(s.values, bins=80, edgecolor="black", alpha=0.8)
        ax.set_title(f"{label}: Histogram of Daily Excess Returns")
        ax.set_xlabel("Daily return"); ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        savefig_and_maybe_show(fig, save_dir / "btop50_excess_hist.png", show=show_plots)
        if len(s) > 70:
            vol63 = s.rolling(63).std() * np.sqrt(ANN_BDAYS)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(_index_to_naive(vol63.index), vol63.values, lw=1.2)
            ax.set_title(f"{label}: 63-Day Rolling Volatility (annualized)")
            ax.set_ylabel("Volatility"); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.grid(True, alpha=0.3)
            savefig_and_maybe_show(fig, save_dir / "btop50_excess_rolling_vol63.png", show=show_plots)
    except Exception as e:
        print(f"Plotting failed: {e}")
    return pd.DataFrame([stats_row])

def process_futures_data(futures_input_dir: str | Path | None = None):
    base = Path(futures_input_dir) if futures_input_dir is not None else FUTURES_INPUT_DIR
    base = base.expanduser()
    PRICE_DIR = base / "multiple_prices_csv"
    ROLL_DIR  = base / "roll_calendars_csv"
    CFG_DIR   = base / "csvconfig"
    OUT_DIR   = ADJUSTED_OUT_DIR
    logger.info(f"Using FUTURES_INPUT_DIR = {base.resolve()}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing_dirs = [p for p in [PRICE_DIR, ROLL_DIR] if not p.exists()]
    if missing_dirs:
        raise FileNotFoundError(
            "Missing required futures data folders: "
            + ", ".join(str(p) for p in missing_dirs)
            + "\nExpected tree:\n  multiple_prices_csv/  roll_calendars_csv/  (and csvconfig/ for configs)"
        )
    ROLL_OFFSET_BD = 5
    SKIP = ({"HANGTECH", "IBEX"}
            | {p.stem for p in PRICE_DIR.glob("*_micro.csv")}
            | {p.stem for p in PRICE_DIR.glob("*_mini.csv")}
            | AGS_DROP)
    inst_path = CFG_DIR / "instrumentconfig.csv"
    cost_path = CFG_DIR / "spreadcosts.csv"
    if inst_path.exists():
        inst = pd.read_csv(inst_path).set_index("Instrument")
    else:
        logger.warning(f"{inst_path} not found — default TickSize=0.01 will be used where needed.")
        inst = pd.DataFrame(columns=["TickSize"]).set_index(pd.Index([], name="Instrument"))
    if cost_path.exists():
        cost = pd.read_csv(cost_path).set_index("Instrument")
    else:
        logger.warning(f"{cost_path} not found — default SpreadCost=1.0 tick will be used where needed.")
        cost = pd.DataFrame(columns=["SpreadCost"]).set_index(pd.Index([], name="Instrument"))
    rf = load_rf()
    def reshape_to_long(path: Path, root: str) -> pd.DataFrame:
        wide = pd.read_csv(path)
        def pick(*alts):
            for a in alts:
                if a in wide.columns:
                    return a
            raise KeyError(f"{alts} missing in {path.name}")
        wide = wide.rename(columns={
            pick("DATETIME", "DATE_TIME", "DATE"): "Date",
            pick("PRICE", "PX_LAST", "CLOSE"): "Price",
            pick("PRICE_CONTRACT", "CONTRACT"): "PriceContract",
            pick("CARRY", "CARRY_LAST"): "Carry",
            pick("CARRY_CONTRACT"): "CarryContract",
            pick("FORWARD", "FORWARD_LAST"): "Forward",
            pick("FORWARD_CONTRACT"): "ForwardContract"
        })
        UNIT_FIX = {"Silver": .01, "GAS_": .01, "Corn_mini": .01, "Wheat_mini": .01, "Sugar": .01, "HG1": .01, "RB": .01}
        fac = next((UNIT_FIX[p] for p in UNIT_FIX if root.startswith(p.rstrip("_"))), 1)
        wide[["Price", "Carry", "Forward"]] = (wide[["Price", "Carry", "Forward"]].apply(pd.to_numeric, errors="coerce") * fac)
        price_rows = (wide[["Date", "PriceContract", "Price"]]
                      .rename(columns={"PriceContract": "Contract", "Price": "Close"})
                      .assign(Kind="Price"))
        carry_rows = (wide[["Date", "CarryContract", "Carry"]]
                      .rename(columns={"CarryContract": "Contract", "Carry": "Close"})
                      .assign(Kind="Carry"))
        fwd_rows   = (wide[["Date", "ForwardContract", "Forward"]]
                      .rename(columns={"ForwardContract": "Contract", "Forward": "Close"})
                      .assign(Kind="Forward"))
        long = (pd.concat([price_rows, carry_rows, fwd_rows], ignore_index=True)
                .dropna(subset=["Close"])
                .assign(Date=lambda d: to_utc(pd.to_datetime(d["Date"]).dt.normalize()))
                .sort_values(["Date", "Contract"]))
        long = long[(long["Date"] >= FILTER_START) & (long["Date"] <= FILTER_END)]
        return long
    for csv_path in sorted(PRICE_DIR.glob("*.csv")):
        root = csv_path.stem
        if root in SKIP:
            continue
        long = reshape_to_long(csv_path, root)
        roll_file = ROLL_DIR / f"{root}.csv"
        if not roll_file.exists():
            logger.warning(f"{root}: missing roll calendar — skipping instrument")
            continue
        roll = (pd.read_csv(roll_file, parse_dates=["DATE_TIME"])
                .rename(columns={"DATE_TIME": "RollDate", "current_contract": "Current", "next_contract": "Next"})
                .sort_values("RollDate")
                .assign(ExecDate=lambda d: d["RollDate"] - BDay(ROLL_OFFSET_BD)))
        roll["ExecDate"] = to_utc(pd.to_datetime(roll["ExecDate"]))
        roll = roll[(roll["ExecDate"] >= FILTER_START) & (roll["ExecDate"] <= FILTER_END)]
        adj_factor, factor = {}, 1.0
        for cur, nxt, ex in zip(roll["Current"][::-1], roll["Next"][::-1], roll["ExecDate"][::-1]):
            adj_factor[cur] = factor
            p_cur = long.query("Contract == @cur & Date == @ex & Kind == 'Price'")["Close"]
            p_nxt = long.query("Contract == @nxt & Date == @ex & Kind == 'Price'")["Close"]
            if not p_cur.empty and not p_nxt.empty:
                ratio = p_nxt.iat[-1] / p_cur.iat[0]
                factor *= ratio if ratio < 5 else 1/ratio
        long["AdjPrice"] = long["Close"] * long["Contract"].map(adj_factor).fillna(1.0)
        price = (long.query("Kind == 'Price'").set_index("Date")["AdjPrice"].groupby("Date").last().ffill(limit=3))
        gap_mask = price.index.to_series().diff().dt.days.gt(7)
        price[gap_mask] = np.nan
        price = price.ffill(limit=3)
        price = price.loc[FILTER_START:FILTER_END]
        price_ret = price.pct_change()
        has_carry = not long.query("Kind == 'Carry'").empty
        if has_carry:
            carry = (long.query("Kind == 'Carry'").pivot_table(index="Date", columns="Contract", values="Close", aggfunc="last"))
            carry_ret = carry.diff().stack().groupby(level=0).sum() / price.shift(1)
            roll_ret = carry_ret.reindex(price_ret.index).fillna(0.0)
        else:
            jumps = {}
            for cur, nxt, ex in zip(roll["Current"], roll["Next"], roll["ExecDate"]):
                p_cur = long.query("Contract == @cur & Date == @ex & Kind == 'Price'")["Close"]
                p_nxt = long.query("Contract == @nxt & Date == @ex & Kind == 'Price'")["Close"]
                if not p_cur.empty and not p_nxt.empty:
                    jumps[ex] = p_nxt.iat[0] / p_cur.iat[0] - 1
            roll_ret = pd.Series(jumps).reindex(price_ret.index).fillna(0.0)
        total_ret = (price_ret + roll_ret).dropna()
        roll_flag = (roll_ret.abs() > 1e-12).astype(int)
        try:
            row_ic = inst.loc[root]
            tick_sz = row_ic.get("TickSize", 0.01)
            spread = float(cost.loc[root, "SpreadCost"])
        except Exception:
            tick_sz = 0.01
            spread = 1.0
        spread_pts = spread * tick_sz if spread >= 1 else spread / 1e4 * price.shift(1)
        tick_pct = spread_pts / price.shift(1)
        daily_cost = (pd.Series(tick_pct, index=price.index) * roll_flag).reindex(total_ret.index).fillna(0.0).shift(1)
        rf = load_rf()
        excess = (total_ret - daily_cost - rf.reindex(total_ret.index).ffill().bfill()).astype("float32")
        excess = excess.loc[FILTER_START:FILTER_END]
        excess.name = root
        def winsorise_series_inproc(s: pd.Series, z: float = 5.0, window: int = 2520, min_window: int = 504):
            if s.notna().sum() < 250:
                return s
            effective_min = max(min_window, window // 2)
            cutoff_date = pd.Timestamp("2003-01-01", tz="UTC")
            expanding_stats = s.expanding(min_periods=250)
            mu = pd.Series(index=s.index, dtype='float64')
            sigma = pd.Series(index=s.index, dtype='float64')
            early_mask = s.index < cutoff_date
            if early_mask.any():
                rolling_early = s.rolling(window=window, min_periods=effective_min)
                mu[early_mask] = rolling_early.mean()[early_mask].fillna(expanding_stats.mean()[early_mask])
                sigma[early_mask] = rolling_early.std()[early_mask].fillna(expanding_stats.std()[early_mask])
            late_mask = ~early_mask
            if late_mask.any():
                rolling_strict = s.rolling(window=window, min_periods=window)
                mu[late_mask] = rolling_strict.mean()[late_mask].fillna(expanding_stats.mean()[late_mask])
                sigma[late_mask] = rolling_strict.std()[late_mask].fillna(expanding_stats.std()[late_mask])
            mask = mu.notna() & sigma.notna()
            clipped = s.copy()
            clipped[mask] = s[mask].clip(lower=mu[mask] - z*sigma[mask], upper=mu[mask] + z*sigma[mask])
            return clipped
        excess = winsorise_series_inproc(excess)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        excess.to_csv(OUT_DIR / f"{root}_excess_net_ret.csv", float_format="%.8f")
        price.to_frame("AdjClose").to_csv(OUT_DIR / f"{root}_adj_price.csv", float_format="%.4f")
        ann_vol = float(excess.std() * np.sqrt(252))
        print(f"{root:<14}| rows {len(excess):6,d} | σ {ann_vol:.2%}")
    print("\nDone — cleaned files saved in:", OUT_DIR)

def build_panel():
    SRC_DIR = ADJUSTED_OUT_DIR
    DEST_DIR = PANEL_OUTPUT_DIR
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    cal_path = CACHE_DIR / "btop50_calendar.pkl"
    if cal_path.exists():
        calendar = pd.read_pickle(cal_path)
    else:
        print("Trading calendar not found, using US business days")
        calendar = pd.bdate_range(FILTER_START, FILTER_END, tz="UTC")
    calendar = calendar[(calendar >= FILTER_START) & (calendar <= FILTER_END)]
    MIN_DAYS = 500
    candidates = [
        RB_DIR / "bad_contracts.txt",
        FUTURES_INPUT_DIR / "bad_contracts.txt",
        Path.cwd() / "bad_contracts.txt",
    ]
    bad_contracts = set()
    for c in candidates:
        if c.exists():
            bad_contracts = set(c.read_text().split())
            break
    series, active = {}, {}
    for f in SRC_DIR.glob("*_excess_net_ret.csv"):
        root = f.stem.replace("_excess_net_ret", "")
        if root in bad_contracts:
            continue
        s = (pd.read_csv(f, parse_dates=[0], index_col=0).squeeze("columns").astype("float32"))
        s.index = to_utc(s.index)
        s = s.loc[FILTER_START:FILTER_END]
        series[root] = s.reindex(calendar).ffill(limit=3)
        active[root] = series[root].notna().astype("int8")
    valid = [k for k, s in series.items() if s.notna().sum() >= MIN_DAYS]
    series = {k: series[k] for k in valid}
    active = {k: active[k] for k in valid}
    X = pd.concat(series, axis=1).astype("float32")
    M = pd.concat(active, axis=1)
    assert not X.columns.duplicated().any(), "Duplicate tickers in X panel!"
    assert X.index[0] >= FILTER_START, f"Start {X.index[0]} < {FILTER_START}"
    assert X.index[-1] <= FILTER_END, f"End {X.index[-1]} > {FILTER_END}"
    X.to_parquet(DEST_DIR / "X_panel.parquet", compression="snappy")
    M.to_parquet(DEST_DIR / "M_active_mask.parquet", compression="snappy")
    print(f"Panel & mask written → {DEST_DIR}")
    print(f"   • contracts kept : {len(X.columns)}")
    print(f"   • calendar rows  : {len(X)}")
    print(f"   • date range     : {X.index[0].date()} → {X.index[-1].date()}")
    y_path = CACHE_DIR / "y_excess_BTOP50.pkl"
    if y_path.exists():
        y_excess = pd.read_pickle(y_path)
        common_dates = X.index.intersection(y_excess.index)
        print(f"   • Common dates with BTOP50: {len(common_dates)} ({len(common_dates) / len(X.index) * 100:.1f}%)")
    return X, M

def create_large_universe():
    PANEL_RAW = PANEL_OUTPUT_DIR / "X_panel.parquet"
    X_raw = pd.read_parquet(PANEL_RAW).astype("float32").loc[FILTER_START:FILTER_END]
    ALIASES = {
        "AUD": ["AUD", "6A", "AD"],
        "CAD": ["CAD", "6C", "CD"],
        "CHF": ["CHF", "6S", "SF"],
        "EUR": ["EUR", "EC", "6E"],
        "GBP": ["GBP", "BP", "6B"],
        "JPY": ["JPY", "JY", "6J"],
        "US2Y": ["US2Y", "US2", "TU"],
        "US5Y": ["US5Y", "US5", "FV"],
        "US10Y": ["US10Y", "US10", "TY"],
        "BOBL": ["BOBL"],
        "BUND": ["BUND", "RX", "BTP"],
        "S&P500": ["S&P500", "SP500", "ES"],
        "NASDAQ": ["NASDAQ", "NQ"],
        "DOW": ["DOW", "YM"],
        "EUROSTOXX": ["EUROSTOXX", "EUROSTX", "VG", "FESX", "SX5E"],
        "DAX": ["DAX", "GX", "FDAX"],
        "FTSE": ["FTSE", "FTSE100", "Z"],
        "TOPIX": ["TOPIX", "TPX", "NI"],
        "HSI": ["HSI", "HSI_mini", "HANGSENG"],
        "WTI_CRUDE": ["CRUDE_W", "WTI_CRUDE", "CL", "CL1"],
        "BRENT_CRUDE": ["BRENT_LAST", "BRENT_W", "BRE", "BRN", "CO"],
        "GASOIL": ["GASOIL", "QS", "GO"],
        "GOLD": ["GOLD", "GC", "GC_MINI"],
        "SILVER": ["SILVER", "SI"],
        "COPPER": ["COPPER", "HG", "HG_MINI"],
    }
    MIN_DAYS = 500
    picked, used_lbl = {}, {}
    for friendly, labels in ALIASES.items():
        for lbl in labels:
            s = X_raw.get(lbl)
            if s is not None and pd.Series(s).notna().sum() >= MIN_DAYS:
                picked[friendly] = s
                used_lbl[friendly] = lbl
                if lbl != friendly:
                    print(f"→ {friendly}: using alias '{lbl}'")
                break
        else:
            print(f"{friendly}: no usable history — dropped")
    X_large = (pd.DataFrame(picked).sort_index().astype("float32").dropna(how="all")).loc[FILTER_START:FILTER_END]
    X_large = X_large[[c for c in X_large.columns if c not in AGS_DROP]]
    OUT = PANEL_OUTPUT_DIR / "X_large_universe"
    OUT.mkdir(parents=True, exist_ok=True)
    X_large.to_parquet(OUT / "X_large_universe.parquet", compression="snappy")
    X_large.to_csv(OUT / "X_large_universe.csv", float_format="%.8f")
    pd.Series(used_lbl, name="ColumnUsed").to_csv(OUT / "friendly_to_label_map.csv")
    print("Large-universe panel written →", OUT)
    return X_large

def tune_alphas_kfold_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    lookback: int,
    alpha_grid=None,
    l1_ratio: float = 0.5,
    n_splits: int = 5,
    rng_seed: int = 42,
    min_instruments: int = 5,
    min_val_window: int = 20,
    warm_start_alpha: float | None = None,
    training_window: int = 756
) -> float:
    if lookback > training_window:
        logger.warning(f"Lookback {lookback} > training_window {training_window} — continuing with guard rails")
    common_idx = X_train.index.intersection(y_train.index)
    X, y = X_train.loc[common_idx], y_train.loc[common_idx]
    end = len(X)
    start = max(0, end - training_window)
    X, y = X.iloc[start:end], y.iloc[start:end]
    if len(X) < max(lookback * (n_splits + 1), 120):
        return warm_start_alpha or 1e-3
    if warm_start_alpha is not None:
        alpha_grid = np.logspace(np.log10(warm_start_alpha * 0.5), np.log10(warm_start_alpha * 2.0), 7)
    elif alpha_grid is None:
        alpha_grid = np.logspace(-6, -3, 10)
    cv_results: dict[float, list[float]] = {float(a): [] for a in alpha_grid}
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for _, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        if len(val_idx) < min_val_window:
            continue
        val_start_from_beginning = tr_idx[0] + len(tr_idx) + len(val_idx)
        if val_start_from_beginning > training_window:
            excess = val_start_from_beginning - training_window
            if excess >= len(val_idx):
                continue
            val_idx = val_idx[:-excess]
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        if len(X_tr) < lookback:
            continue
        for alpha in alpha_grid:
            pnl_fold = []
            for t in range(len(X_val)):
                val_pos = val_idx[t]
                lb_start = max(0, val_pos - lookback)
                X_fit, y_fit = X.iloc[lb_start:val_pos], y.iloc[lb_start:val_pos]
                active = X_fit.columns[X_fit.notna().sum() > 0]
                if len(active) < min_instruments:
                    continue
                model = ElasticNet(
                    l1_ratio=l1_ratio, alpha=float(alpha), fit_intercept=False,
                    max_iter=10000, random_state=rng_seed
                )
                model.fit(X_fit[active].fillna(0.0).values, y_fit.values)
                coef = pd.Series(0.0, index=X.columns)
                coef[active] = model.coef_
                pnl_fold.append(float(X_val.iloc[t].fillna(0.0) @ coef))
            if len(pnl_fold) > min_val_window // 2:
                sharpe = (np.mean(pnl_fold) / (np.std(pnl_fold, ddof=1) + 1e-12)) * np.sqrt(252)
                cv_results[float(alpha)].append(float(sharpe))
    best_alpha, best_avg_sharpe = warm_start_alpha or 1e-3, -np.inf
    for alpha, sharpes in cv_results.items():
        if sharpes:
            avg_s = float(np.median(sharpes))
            if avg_s > best_avg_sharpe:
                best_avg_sharpe, best_alpha = avg_s, float(alpha)
    return float(best_alpha)

def tune_alphas_quarterly(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    current_date: pd.Timestamp,
    lookbacks=(20, 25, 30, 35, 40),
    alpha_grid=None,
    l1_ratio: float = 0.5,
    n_splits: int = 5,
    training_window: int = 756,
    rng_seed: int = 42,
    min_instruments: int = 5,
    min_val_window: int = 20,
    previous_alphas: dict | None = None,
    persist_to_disk: bool = True,
    universe_name: str = "large"
) -> dict:
    mask = X_full.index <= current_date
    X_av, y_av = X_full.loc[mask], y_full.loc[mask]
    X_train = X_av.iloc[-training_window:]
    y_train = y_av.loc[X_train.index]
    logger.info(f"Re-optimizing alphas @ {current_date.date()} | window: {X_train.index[0].date()} → {X_train.index[-1].date()}")
    use_parallel = JOBLIB_AVAILABLE and len(lookbacks) > 1 and N_JOBS > 1
    if use_parallel:
        logger.info(f"Parallel alpha tuning for {len(lookbacks)} windows with {N_JOBS} jobs")
        results = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(tune_alphas_kfold_cv)(
                X_train, y_train, w, None, l1_ratio, n_splits, rng_seed + w,
                min_instruments, min_val_window, previous_alphas.get(w) if previous_alphas else None, training_window
            )
            for w in lookbacks
        )
        alpha_star = dict(zip(lookbacks, results))
    else:
        alpha_star = {}
        for w in lookbacks:
            warm = previous_alphas.get(w) if previous_alphas else None
            a = tune_alphas_kfold_cv(
                X_train, y_train, w, None, l1_ratio, n_splits, rng_seed + w,
                min_instruments, min_val_window, warm, training_window
            )
            alpha_star[w] = float(a)
            logger.info(f"  w={w:<2d} → α*={alpha_star[w]:.2e}" + (f" (warm-start {warm:.2e})" if warm else ""))
    if persist_to_disk:
        quarter = f"{current_date.year}Q{(current_date.month - 1)//3 + 1}"
        fn = AUDIT_DIR / f"alphas_{universe_name}_{quarter}_{current_date.strftime('%Y%m%d')}.pkl.gz"
        with gzip.open(fn, "wb") as f:
            pickle.dump({
                'date': current_date, 'alphas': alpha_star, 'lookbacks': lookbacks,
                'training_window': training_window, 'universe': universe_name, 'suite': SUFFIX
            }, f)
        logger.info(f"Saved compressed alphas → {fn}")
    return alpha_star

def _build_single_universe(
    X_raw: pd.DataFrame,
    y: pd.Series,
    lookbacks=(20,25,30,35,40),
    alpha_grid=None,
    l1_ratio=0.5,
    risk_config: RiskConfig = DEFAULT_RISK_CONFIG,
    ann_bdays: int = 252,
    rng_seed: int = 42,
    min_instruments: int = 5,
    initial_training_window: int = 756,
    min_history_pct: float = 0.7,
    blend_win: int = 756,
    oos_start_date: str | None = OOS_START_DATE,
    n_splits: int = 5,
    training_window: int = 756,
    min_val_window: int = 20,
    allow_missing_instruments: bool = False,
    persist_artifacts: bool = True,
    alpha_ema: float = 0.33,
    universe_name: str = "large",
    track_turnover: bool = True
) -> dict:
    X_scaled = pd.DataFrame(index=X_raw.index, columns=X_raw.columns, dtype="float64")
    for col in X_raw.columns:
        X_scaled[col] = rescale_to_target(X_raw[col].astype("float64"), risk_config=risk_config)
    common = X_scaled.index.intersection(y.index)
    X, y = X_scaled.loc[common].copy(), y.loc[common].astype("float64")
    first_tune_date = X.index[min(initial_training_window - 1, len(X) - 1)]
    quarter_ends = pd.date_range(first_tune_date, X.index[-1], freq='Q')
    quarter_ends = quarter_ends[quarter_ends <= X.index[-1]]
    logger.info(f"Quarterly alpha tuning ({universe_name}) | mode: LIVE")
    alpha_history: dict[pd.Timestamp, dict] = {}
    previous_alphas = None
    for qe in tqdm(quarter_ends, desc=f"Tuning ({universe_name})"):
        alpha_history[qe] = tune_alphas_quarterly(
            X_full=X, y_full=y, current_date=qe, lookbacks=lookbacks, alpha_grid=alpha_grid,
            l1_ratio=l1_ratio, n_splits=n_splits, training_window=training_window, rng_seed=rng_seed,
            min_instruments=min_instruments, min_val_window=min_val_window,
            previous_alphas=previous_alphas, persist_to_disk=persist_artifacts, universe_name=universe_name
        )
        previous_alphas = alpha_history[qe]
    if first_tune_date not in alpha_history:
        alpha_history[first_tune_date] = tune_alphas_quarterly(
            X_full=X, y_full=y, current_date=first_tune_date, lookbacks=lookbacks, alpha_grid=alpha_grid,
            l1_ratio=l1_ratio, n_splits=n_splits, training_window=training_window, rng_seed=rng_seed,
            min_instruments=min_instruments, min_val_window=min_val_window,
            previous_alphas=None, persist_to_disk=persist_artifacts, universe_name=universe_name
        )
    predictions_per_model = pd.DataFrame(index=X.index, columns=lookbacks, dtype="float64")
    initial_burn_in = max(lookbacks) + 2
    last_valid_coeffs = {w: pd.Series(0.0, index=X.columns, dtype="float64") for w in lookbacks}
    ema_coef = {w: None for w in lookbacks}
    weight_history = {w: pd.DataFrame(index=X.index, columns=X.columns, dtype="float32") for w in lookbacks} if track_turnover else {}
    persistence_usage = defaultdict(int)
    total_predictions = defaultdict(int)
    for t in tqdm(range(initial_burn_in, len(X)), desc=f"WF ({universe_name})"):
        current_date = X.index[t]
        valid_dates = [d for d in alpha_history.keys() if d <= current_date]
        alpha_star = alpha_history[max(valid_dates)] if valid_dates else alpha_history[first_tune_date]
        for w in lookbacks:
            X_fit = X.iloc[t - w:t]
            y_fit = y.loc[X_fit.index]
            if allow_missing_instruments:
                active = X_fit.columns[X_fit.notna().sum() >= 1]
            else:
                active = X_fit.columns[X_fit.notna().sum() >= int(w * min_history_pct)]
            total_predictions[w] += 1
            if len(active) < min_instruments:
                coef = last_valid_coeffs[w].copy()
                persistence_usage[w] += 1
            else:
                alpha = float(alpha_star.get(w, 1e-3))
                model = ElasticNet(
                    l1_ratio=l1_ratio, alpha=alpha, fit_intercept=False,
                    max_iter=10000, random_state=rng_seed
                )
                model.fit(X_fit[active].fillna(0.0).values, y_fit.values)
                coef = pd.Series(0.0, index=X.columns, dtype="float64"); coef[active] = model.coef_
                last_valid_coeffs[w] = coef.copy()
            ema_coef[w] = coef.copy() if (ema_coef[w] is None) else ema_update(ema_coef[w], coef, alpha_ema)
            smooth_coef = ema_coef[w]
            if track_turnover:
                weight_history[w].loc[current_date] = smooth_coef.astype("float32")
            predictions_per_model.loc[current_date, w] = float(X.iloc[t].fillna(0.0) @ smooth_coef)
    beta_hist = pd.DataFrame(index=predictions_per_model.index, columns=list(lookbacks) + ['intercept'], dtype="float64")
    for i, ts in enumerate(predictions_per_model.index):
        data_for_fit = predictions_per_model.iloc[:i]
        if len(data_for_fit) < blend_win:
            continue
        y_blend = y.loc[data_for_fit.index].tail(blend_win)
        X_blend = data_for_fit.tail(blend_win)
        mask = X_blend.notna().all(axis=1)
        Xb, yb = X_blend.loc[mask], y_blend.loc[mask]
        if len(Xb) < max(30, int(0.5 * blend_win)):
            continue
        reg = LinearRegression(fit_intercept=True).fit(Xb, yb)
        beta_hist.loc[ts, Xb.columns] = reg.coef_
        beta_hist.loc[ts, 'intercept'] = float(reg.intercept_)
    beta_hist = beta_hist.ffill()
    weighted_predictions = (predictions_per_model * beta_hist.drop('intercept', axis=1)).sum(axis=1)
    blended_preds = weighted_predictions.ffill()
    combined_signal = blended_preds + beta_hist['intercept']
    pnl_final = rescale_to_target(combined_signal, target=risk_config.portfolio_vol_target, risk_config=risk_config).dropna().astype("float32")
    turnover_metrics = {}
    weight_paths = {}
    for w in lookbacks:
        dfw = weight_history[w].dropna(how='all')
        if len(dfw) > 1:
            daily_turn = dfw.diff().abs().sum(axis=1)
            turnover_metrics[w] = {
                'mean_daily': float(daily_turn.mean()),
                'std_daily': float(daily_turn.std()),
                'annual': float(daily_turn.mean() * 252)
            }
        if persist_artifacts:
            WEIGHT_HISTORY_DIR.mkdir(exist_ok=True, parents=True)
            path = WEIGHT_HISTORY_DIR / f"weights_large_w{w}.parquet"
            (weight_history[w].astype('float16').to_parquet(path, compression='gzip'))
            weight_paths[w] = path
    if persist_artifacts and weight_paths:
        logger.info(f"Saved weight history → {WEIGHT_HISTORY_DIR}")
    persistence_pct = {w: (persistence_usage[w] / total_predictions[w] * 100) if total_predictions[w] else 0.0 for w in lookbacks}
    avg_persistence_pct = float(np.mean(list(persistence_pct.values()))) if persistence_pct else 0.0
    return {
        'combined': pnl_final,
        'alpha_history': alpha_history,
        'beta_history': beta_hist.astype('float32'),
        'predictions_per_model': predictions_per_model.astype('float32'),
        'persistence_usage_pct': persistence_pct,
        'avg_persistence_pct': avg_persistence_pct,
        'ema_coef_final': {w: coef.astype('float32') for w, coef in ema_coef.items() if coef is not None},
        'turnover_metrics': turnover_metrics,
        'risk_config': risk_config,
        'weight_history_paths': weight_paths,
    }

def monitor_implementation_quality(results: dict, y: pd.Series) -> pd.DataFrame:
    pnl = results['combined']
    common_idx = pnl.index.intersection(y.index)
    risk_config = results.get('risk_config', DEFAULT_RISK_CONFIG)
    metrics = {
        'Sharpe (Model)': float(pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(252)),
        'Correlation with BTOP50': float(pnl.corr(y.loc[common_idx])),
        'Tracking Error (vs BTOP50)': float((pnl - y.loc[common_idx]).std() * np.sqrt(252)),
        'Autocorrelation (1d)': float(pnl.autocorr(lag=1)),
        'Autocorrelation (5d)': float(pnl.autocorr(lag=5)),
        'Max Drawdown': float((pnl.cumsum() - pnl.cumsum().expanding().max()).min()),
        'Hit Rate': float((pnl > 0).mean()),
        'Instrument Vol Target': risk_config.instrument_vol_target,
        'Portfolio Vol Target': risk_config.portfolio_vol_target,
        'Leverage Cap': risk_config.max_leverage_cap,
    }
    return pd.DataFrame(metrics, index=['Value']).T

def calculate_detailed_metrics(returns, benchmark, period_name, apply_burnin=True, burn_in_days=189):
    if apply_burnin and len(returns) > burn_in_days:
        returns = returns.iloc[burn_in_days:]
        benchmark = benchmark.loc[returns.index]
    ann = np.sqrt(252)
    m = {}
    m['Period'] = period_name
    m['Annual Return'] = float(returns.mean() * 252)
    m['Annual Volatility'] = float(returns.std() * ann)
    m['Sharpe Ratio'] = float(m['Annual Return'] / (m['Annual Volatility'] + 1e-12))
    m['Correlation'] = float(returns.corr(benchmark))
    m['R-squared'] = float(m['Correlation'] ** 2)
    m['Tracking Error'] = float((returns - benchmark).std() * ann)
    m['Max Drawdown'] = float((returns.cumsum().cummax() - returns.cumsum()).max())
    m['Skewness'] = float(returns.skew())
    m['Excess Kurtosis'] = float(returns.kurtosis())
    m['95% VaR'] = float(returns.quantile(0.05))
    m['Hit Rate'] = float((returns > 0).mean())
    return m

def create_returns_based_visualizations(
    ret_model: pd.Series,
    pnl_large: pd.Series,
    benchmark: pd.Series,
    burn_in_end_date,
    *,
    BURN_IN_DAYS: int,
    BURN_IN_MONTHS: int,
    OOS_START_DATE: str,
    custom_risk_config: RiskConfig,
    ANALYSIS_OUT: Path = RB_DIR,
    show_plots: bool = SHOW_PLOTS,
):
    suffix = SUFFIX
    print("\n=== Viz A — Cumulative Equity Curves (Returns-based) ===")
    fig, ax = plt.subplots(figsize=(14, 8))
    series_to_plot = []
    if pnl_large is not None and len(pnl_large) > 0:
        series_to_plot.append(("Returns-based (Large Universe)", pnl_large, COLOR_PALETTE["model"]))
    if benchmark is not None and len(benchmark) > 0:
        series_to_plot.append(("BTOP50 (Excess)", benchmark, COLOR_PALETTE["benchmark"]))
    for label, series, color in series_to_plot:
        cum_ret = 100 * (1 + series).cumprod()
        idx = _index_to_naive(cum_ret.index)
        ax.plot(idx, cum_ret.values, lw=1.8, label=label, color=color)
    burn_span_start = pnl_large.index[0] if pnl_large is not None and len(pnl_large) > 0 else None
    if burn_span_start is not None and burn_in_end_date is not None:
        ax.axvspan(to_naive(burn_span_start), to_naive(burn_in_end_date), alpha=0.1, color="gray",
                   label=f"{BURN_IN_MONTHS}-Month Burn-in")
    annotate_crisis_periods(ax)
    ax.axvline(to_naive(OOS_START_DATE), color="black", linestyle="--", alpha=0.7, linewidth=2, label="OOS Start")
    ax.set_title(f"Cumulative Growth of $100",
                 fontsize=14, weight="bold")
    ax.set_ylabel("Portfolio Value ($)"); ax.set_xlabel("Date"); ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3); ax.set_yscale("log")
    for label, series, color in series_to_plot:
        try:
            final_value = float(100 * (1 + series).cumprod().iloc[-1])
            ts = to_naive(series.index[-1])
            ax.text(ts, final_value, f"${final_value:.0f}", va="center", fontsize=9, color=color)
        except Exception:
            pass
    savefig_and_maybe_show(fig, ANALYSIS_OUT / f"A_cumulative_equity_curves_{suffix}.png", show=show_plots)
    print("\n=== Viz B — Rolling Statistics (Returns-based) ===")
    window = 252
    if ret_model is None or benchmark is None or len(ret_model) == 0 or len(benchmark) == 0:
        print("Missing data for rolling stats; skipping.")
    else:
        ret_after = ret_model.loc[burn_in_end_date:]; bench_after = benchmark.loc[burn_in_end_date:]
        if len(ret_after) < window + 10:
            print("Not enough data after burn-in; skipping Chart B.")
        else:
            roll_corr = ret_after.rolling(window).corr(bench_after)
            roll_rs2 = roll_corr ** 2
            roll_te = (ret_after - bench_after).rolling(window).std() * np.sqrt(252)
            roll_ir = ((ret_after - bench_after).rolling(window).mean()
                       / (ret_after - bench_after).rolling(window).std()) * np.sqrt(252)
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
            idx1 = _index_to_naive(roll_corr.index)
            ax1.plot(idx1, roll_corr.values, color=COLOR_PALETTE["normal"], lw=1.5)
            ax1.axhline(y=float(np.nanmean(roll_corr.values)), color="black", linestyle="--", alpha=0.5,
                        label=f"Mean: {float(np.nanmean(roll_corr.values)):.3f}")
            ax1.set_ylabel("Correlation")
            ax1.set_title(f"Rolling 1Y Statistics (Ex-Burn-in) — Returns-based", fontsize=12, weight="bold")
            ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_ylim(0, 1)
            idx2 = _index_to_naive(roll_rs2.index)
            ax2.plot(idx2, roll_rs2.values, color=COLOR_PALETTE["model"], lw=1.5)
            ax2.axhline(y=float(np.nanmean(roll_rs2.values)), color="black", linestyle="--", alpha=0.5,
                        label=f"Mean: {float(np.nanmean(roll_rs2.values)):.3f}")
            ax2.fill_between(idx2, 0, roll_rs2.values, alpha=0.2, color=COLOR_PALETTE["model"])
            ax2.set_ylabel("R-squared"); ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 1)
            ax2.text(0.02, 0.95, f"Avg Variance Explained: {float(np.nanmean(roll_rs2.values))*100:.1f}%",
                     transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                     va="top", fontsize=9)
            idx3 = _index_to_naive(roll_te.index)
            ax3.plot(idx3, roll_te.values, color=COLOR_PALETTE["crisis"], lw=1.5)
            ax3.axhline(y=float(np.nanmean(roll_te.values)), color="black", linestyle="--", alpha=0.5,
                        label=f"Mean: {float(np.nanmean(roll_te.values)):.1%}")
            ax3.set_ylabel("Tracking Error"); ax3.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax3.legend(); ax3.grid(True, alpha=0.3)
            idx4 = _index_to_naive(roll_ir.index)
            ax4.plot(idx4, roll_ir.values, color=COLOR_PALETTE["large"], lw=1.5)
            ax4.axhline(y=0, color="black", linestyle="-", alpha=0.5)
            ax4.axhline(y=float(np.nanmean(roll_ir.values)), color="black", linestyle="--", alpha=0.5,
                        label=f"Mean IR: {float(np.nanmean(roll_ir.values)):.3f}")
            ax4.set_ylabel("Information Ratio"); ax4.set_xlabel("Date"); ax4.legend(); ax4.grid(True, alpha=0.3)
            for ax in (ax1, ax2, ax3, ax4):
                annotate_crisis_periods(ax, y_position="top")
                ax.axvline(to_naive(OOS_START_DATE), color="gray", linestyle="--", alpha=0.7)
            savefig_and_maybe_show(fig, ANALYSIS_OUT / f"B_rolling_statistics_{suffix}.png", show=show_plots)
    print("\n=== Viz C — Regime-Colored Monthly Scatter (Returns-based) ===")
    if ret_model is None or benchmark is None or len(ret_model) == 0 or len(benchmark) == 0:
        print("Missing data for scatter; skipping.")
        return
    monthly_model = ret_model.loc[burn_in_end_date:].resample("M").sum()
    monthly_bench = benchmark.loc[burn_in_end_date:].resample("M").sum()
    rolling_sharpe = benchmark.rolling(252).mean() / (benchmark.rolling(252).std() + 1e-12) * np.sqrt(252)
    monthly_sharpe = rolling_sharpe.resample("M").last()
    common_months = monthly_model.index.intersection(monthly_bench.index).intersection(monthly_sharpe.index)
    if len(common_months) < 6:
        print("Not enough monthly overlap; skipping.")
        return
    x, yv = monthly_bench.loc[common_months]*100, monthly_model.loc[common_months]*100
    sharpe_vals = monthly_sharpe.loc[common_months]
    regimes = sharpe_vals.apply(classify_regime)
    rc = [REGIME_COLORS[r] for r in regimes]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, yv)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(x.values, yv.values, alpha=0.7, s=80, c=rc, edgecolors="black", linewidth=0.5)
    xl = np.linspace(float(x.min()), float(x.max()), 100)
    ax.plot(xl, slope*xl + intercept, color=COLOR_PALETTE["crisis"], lw=2,
            label=f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r_value**2:.3f}")
    ax.plot([float(x.min()), float(x.max())], [float(x.min()), float(x.max())],
            "k--", alpha=0.5, label="45°")
    ax.set_xlabel("BTOP50 Monthly Return (%)"); ax.set_ylabel("Returns-based (Large) Monthly Return (%)")
    ax.set_title(f"Monthly Scatter: Returns-based vs BTOP50 (After {BURN_IN_MONTHS}-Month Burn-in)", fontsize=14, weight="bold")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    patches = [Patch(color=c, label=name.replace("_"," ").title()) for name, c in REGIME_COLORS.items()]
    reg_leg = ax.legend(handles=patches, loc="lower right", title="BTOP50 Regime\n(Rolling Sharpe)", fontsize=8); ax.add_artist(reg_leg)
    stats_text = f"Correlation: {r_value:.3f}\nTracking Error: {(yv-x).std():.1f}%\nMean Alpha: {(yv-x).mean():.1f}%"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            va="top", ha="left", fontsize=10)
    savefig_and_maybe_show(fig, ANALYSIS_OUT / f"C_monthly_scatter_regime_{suffix}.png", show=show_plots)

def compute_drawdown_attribution_single(results: dict, benchmark: pd.Series):
    return {}

def run_extended_diagnostics(
    results: dict,
    benchmark: pd.Series,
    oos_start_date: str,
    analysis_out: Path = RB_DIR,
    *,
    show_plots: bool = SHOW_PLOTS,
):
    ret = results['combined']
    benchmark = benchmark.loc[ret.index]
    suffix = SUFFIX
    def _calculate_drawdown(returns):
        cum = (1 + returns).cumprod()
        run_max = cum.expanding().max()
        return (cum - run_max) / run_max
    try:
        print("\n=== EXT — Turnover & Cost Diagnostics (Large-only) ===")
        t_large = results.get('turnover_metrics', {})
        if t_large:
            fig, ax = plt.subplots(figsize=(8, 6))
            lbs = sorted(t_large.keys())
            ax.bar([f'{w}d' for w in lbs],
                   [t_large[w]['annual'] for w in lbs],
                   color=COLOR_PALETTE.get('large'), alpha=0.7)
            ax.set_title('Large — Annual Turnover (Returns-based)')
            ax.set_xlabel('Lookback'); ax.set_ylabel('Annual Turnover')
            ax.grid(True, alpha=0.3, axis='y')
            savefig_and_maybe_show(fig, analysis_out / f"C_turnover_by_lookback_{suffix}.png", show=show_plots)
            rows = [{'lookback': int(lb),
                     'mean_daily': float(t_large[lb]['mean_daily']),
                     'std_daily': float(t_large[lb]['std_daily']),
                     'annual': float(t_large[lb]['annual'])} for lb in sorted(t_large.keys())]
            pd.DataFrame(rows).to_csv(analysis_out / f"turnover_by_lookback_{suffix}.csv", index=False)
        else:
            print("No turnover metrics found.")
    except Exception as e:
        print(f"Turnover diagnostics failed: {e}")
    try:
        print("\n=== EXT — Lookback Model Blend Weights ===")
        beta_hist = results.get('beta_history', None)
        if beta_hist is not None and not beta_hist.empty:
            beta_no_inter = beta_hist.drop(columns=['intercept'], errors='ignore')
            denom = beta_no_inter.sum(axis=1).replace(0, np.nan)
            beta_norm = beta_no_inter.div(denom, axis=0).fillna(0.0)
            fig, ax = plt.subplots(figsize=(14, 6))
            cols = list(beta_norm.columns)
            idx = _index_to_naive(beta_norm.index)
            ax.stackplot(idx, [beta_norm[c].values for c in cols],
                         labels=[f'{c}d' for c in cols], alpha=0.85)
            ax.set_title('Evolution of Lookback Blend Weights (Returns-based)', fontsize=14, weight='bold')
            ax.set_ylabel('Normalized Weight'); ax.set_xlabel('Date'); ax.set_ylim(0, 1)
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1)); ax.grid(True, alpha=0.3)
            if oos_start_date:
                ax.axvline(to_naive(oos_start_date), color='black', linestyle='--', alpha=0.7, linewidth=2)
            savefig_and_maybe_show(fig, analysis_out / f"E_lookback_blend_weights_{suffix}.png", show=show_plots)
        else:
            print("No beta_history available.")
    except Exception as e:
        print(f"Lookback weights failed: {e}")
    try:
        print("\n=== EXT — Residuals Diagnostics (Model − BTOP50) ===")
        res = (ret - benchmark).dropna()
        res.to_frame("residual").to_csv(analysis_out / f"rb_residuals_daily_{suffix}.csv")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.hist(res.values, bins=60, alpha=0.85, edgecolor='black')
        ax.set_title('Residuals — Daily (Returns-based − BTOP50)', fontsize=14, weight='bold')
        ax.set_xlabel('Residual (daily return)'); ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        savefig_and_maybe_show(fig, analysis_out / f"F_residuals_hist_{suffix}.png", show=show_plots)
        fig, ax = plt.subplots(figsize=(14, 6))
        idx = _index_to_naive(res.index)
        ax.plot(idx, res.cumsum().values, lw=1.8, color=COLOR_PALETTE.get('model', '#d62728'))
        ax.set_title('Cumulative Residual P&L (Alpha over BTOP50)', fontsize=14, weight='bold')
        ax.set_ylabel('Cumulative residual (sum of daily returns)'); ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        if oos_start_date:
            ax.axvline(to_naive(oos_start_date), color='gray', linestyle='--', alpha=0.7)
        annotate_crisis_periods(ax)
        savefig_and_maybe_show(fig, analysis_out / f"F_residuals_cum_{suffix}.png", show=show_plots)
    except Exception as e:
        print(f"Residuals failed: {e}")
    try:
        print("\n=== EXT — Drawdown Analysis ===")
        dd_model = _calculate_drawdown(ret)
        dd_bench = _calculate_drawdown(benchmark)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        ax1.fill_between(_index_to_naive(dd_model.index), 0, dd_model.values*100,
                         color=COLOR_PALETTE['model'], alpha=0.7, label='Returns-based')
        ax1.plot(_index_to_naive(dd_bench.index), dd_bench.values*100,
                 color=COLOR_PALETTE['benchmark'], lw=2, label='BTOP50')
        ax1.set_ylabel('Drawdown (%)'); ax1.set_title('Drawdown: Returns-based vs BTOP50', fontsize=12, weight='bold')
        ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_ylim(top=1)
        dd_diff = dd_model - dd_bench
        idx = _index_to_naive(dd_diff.index)
        ax2.fill_between(idx, 0, dd_diff.values*100, where=(dd_diff.values >= 0), alpha=0.5,
                         color=COLOR_PALETTE['normal'], label='Returns-based Better')
        ax2.fill_between(idx, 0, dd_diff.values*100, where=(dd_diff.values < 0), alpha=0.5,
                         color=COLOR_PALETTE['crisis'], label='BTOP50 Better')
        ax2.set_ylabel('DD Diff (%)'); ax2.set_xlabel('Date'); ax2.legend(); ax2.grid(True, alpha=0.3); ax2.axhline(0, color='black', alpha=0.5)
        for ax in (ax1, ax2):
            annotate_crisis_periods(ax)
            if oos_start_date:
                ax.axvline(to_naive(oos_start_date), color='gray', linestyle='--', alpha=0.7)
        savefig_and_maybe_show(fig, analysis_out / f"H_drawdown_analysis_{suffix}.png", show=show_plots)
    except Exception as e:
        print(f"Drawdown analysis failed: {e}")
    try:
        print("\n=== EXT — Monthly Return Heatmap ===")
        monthly_ret = ret.resample('M').sum() * 100
        years = monthly_ret.index.year
        months = monthly_ret.index.month
        dfp = pd.DataFrame({'Year': years, 'Month': months, 'Return': monthly_ret.values})
        heat = dfp.pivot(index='Year', columns='Month', values='Return')
        colors = [(0.8,0,0), (1,1,1), (0,0.8,0)]
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(heat, annot=True, fmt='.1f', cmap=cmap, center=0,
                    cbar_kws={'label': 'Monthly Return (%)'}, ax=ax,
                    xticklabels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        ax.set_title('Monthly Returns Heatmap — Returns-based', fontsize=14, weight='bold'); ax.set_xlabel('Month'); ax.set_ylabel('Year')
        savefig_and_maybe_show(fig, analysis_out / f"I_monthly_return_heatmap_{suffix}.png", show=show_plots)
    except Exception as e:
        print(f"Monthly heatmap failed: {e}")
    try:
        print("\n=== EXT — Calendar-Year Bars (Model, Benchmark, Alpha) ===")
        ann_model = ret.resample('Y').sum()
        ann_bench = benchmark.resample('Y').sum().reindex(ann_model.index)
        ann_alpha = ann_model - ann_bench
        out = pd.DataFrame({
            'model': ann_model,
            'benchmark': ann_bench,
            'alpha': ann_alpha
        })
        out.index = out.index.year
        out.index.name = 'year'
        out.to_csv(analysis_out / f"Y_calendar_year_bars_{suffix}.csv")
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(out))
        width = 0.27
        ax.bar(x - width, out['model'].to_numpy(), width, label='Returns-based',
               alpha=0.9, color=COLOR_PALETTE.get('model'))
        ax.bar(x,         out['benchmark'].to_numpy(), width, label='BTOP50',
               alpha=0.9, color=COLOR_PALETTE.get('benchmark'))
        ax.bar(x + width, out['alpha'].to_numpy(), width, label='Alpha (Model − BTOP50)',
               alpha=0.9, color=COLOR_PALETTE.get('normal'))
        ax.set_xticks(x); ax.set_xticklabels(out.index.astype(int), rotation=0)
        ax.set_title('Calendar-Year Returns', fontsize=14, weight='bold')
        ax.set_xlabel('Year'); ax.set_ylabel('Return')
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, alpha=0.3, axis='y'); ax.legend()
        savefig_and_maybe_show(fig, analysis_out / f"Y_calendar_year_bars_{suffix}.png", show=show_plots)
    except Exception as e:
        print(f"Calendar-year bars failed: {e}")
    try:
        print("\n=== EXT — Blender Drift vs Residual TE ===")
        beta_hist = results.get('beta_history', None)
        if beta_hist is not None and not beta_hist.empty:
            res = (ret - benchmark).dropna()
            beta_no_inter = beta_hist.drop(columns=['intercept'], errors='ignore').fillna(0.0)
            drift = beta_no_inter.diff().abs().sum(axis=1).reindex(res.index).fillna(0.0)
            te63 = (res.rolling(63).std() * np.sqrt(252))
            fig, ax1 = plt.subplots(figsize=(14,6))
            ax1.plot(_index_to_naive(te63.index), te63.values, label='63d Residual TE (annualized)', lw=1.5)
            ax1.yaxis.set_major_formatter(PercentFormatter(1.0)); ax1.set_ylabel('Residual TE'); ax1.grid(True, alpha=0.3)
            ax2 = ax1.twinx()
            ax2.plot(_index_to_naive(drift.index), drift.values, label='Daily Blender Drift (Σ|Δβ|)', lw=1.0, alpha=0.7, color=COLOR_PALETTE.get('normal'))
            ax1.set_title('Residual TE vs Lookback Blender Drift — Returns-based', fontsize=14, weight='bold')
            ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
            savefig_and_maybe_show(fig, analysis_out / f"RB_blender_weight_vs_residuals_{suffix}.png", show=show_plots)
        else:
            print("No beta_history found — skipped blender drift.")
    except Exception as e:
        print(f"Blender vs residuals failed: {e}")
    try:
        print("\n=== EXT — Lookback Contribution to Alpha (monthly) ===")
        preds = results.get('predictions_per_model', None)
        if preds is not None and not preds.empty:
            bh = results.get('beta_history', pd.DataFrame(index=preds.index, columns=list(preds.columns)+['intercept'])).fillna(0.0)
            res = (ret - benchmark).reindex(preds.index).dropna()
            preds = preds.reindex(res.index).fillna(0.0)
            bh = bh.reindex(res.index).fillna(0.0)
            raw = (preds.abs() * bh.drop(columns=['intercept'], errors='ignore').abs())
            share = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
            daily_contrib = share.mul(res, axis=0)
            monthly = daily_contrib.resample('M').sum()
            fig, ax = plt.subplots(figsize=(14, 7))
            labels = [f'{c}d' for c in monthly.columns]
            ax.stackplot(_index_to_naive(monthly.index), [monthly[c].values for c in monthly.columns], labels=labels, alpha=0.9)
            ax.set_title('Lookback Contribution to Alpha (monthly sum) — Returns-based', fontsize=14, weight='bold')
            ax.set_ylabel('Alpha (monthly sum of daily residual)'); ax.set_xlabel('Date'); ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
            savefig_and_maybe_show(fig, analysis_out / f"RB_lookback_alpha_contribution_{suffix}.png", show=show_plots)
        else:
            print("No predictions_per_model — skipped alpha contribution.")
    except Exception as e:
        print(f"Alpha contribution failed: {e}")
    try:
        print("\n=== EXT — Gross Exposure by Lookback (NO combined line) ===")
        weight_paths = results.get('weight_history_paths', {})
        if not weight_paths:
            print("No weight history paths available; exposures chart skipped.")
        else:
            fig, ax = plt.subplots(figsize=(14, 6))
            for w in sorted(weight_paths):
                dfw = pd.read_parquet(weight_paths[w]).astype('float32')
                gross = dfw.abs().sum(axis=1).astype('float32')
                ax.plot(_index_to_naive(gross.index), gross.values, lw=1.3, label=f'{w}d')
            ax.set_title('Gross Exposure by Lookback — Returns-based (Large Only)', fontsize=14, weight='bold')
            ax.set_ylabel('Σ |weights|'); ax.set_xlabel('Date'); ax.grid(True, alpha=0.3)
            if oos_start_date:
                ax.axvline(to_naive(oos_start_date), color='gray', linestyle='--', alpha=0.7)
            annotate_crisis_periods(ax)
            ax.legend(ncol=3, fontsize=9)
            savefig_and_maybe_show(fig, analysis_out / f"J_exposures_by_lookback_{suffix}.png", show=show_plots)
    except Exception as e:
        print(f"Exposures plot failed: {e}")
    print("\nExtended diagnostics complete.")

def run_btop50_validation_large_only(X_large_full: pd.DataFrame, y_full: pd.Series):
    BURN_IN_DAYS = 189
    BURN_IN_MONTHS = 9
    TRAIN_VAL_END_DATE = to_utc(pd.to_datetime(OOS_START_DATE)) - pd.Timedelta(days=1)
    custom_risk_config = RiskConfig(
        instrument_vol_target=0.12,
        portfolio_vol_target=0.07,
        max_leverage_cap=1.5,
        vol_halflife=40
    )
    print(f"--- BTOP50 Validation | Mode: LARGE_ONLY (Returns-based) ---")
    print(f"Train/Validation ends: {TRAIN_VAL_END_DATE.date()} | OOS starts: {OOS_START_DATE}")
    print(f"Burn-in: {BURN_IN_MONTHS} months ({BURN_IN_DAYS} trading days)")
    print(f"Risk: inst_vol={custom_risk_config.instrument_vol_target:.1%}, port_vol={custom_risk_config.portfolio_vol_target:.1%}, cap={custom_risk_config.max_leverage_cap:.1f}, hl={custom_risk_config.vol_halflife}")
    results_large = _build_single_universe(
        X_raw=X_large_full, y=y_full,
        lookbacks=(20,25,30,35,40), alpha_grid=None, l1_ratio=0.5,
        risk_config=custom_risk_config, ann_bdays=252, rng_seed=42,
        min_instruments=5, initial_training_window=756, min_history_pct=0.7,
        blend_win=756, oos_start_date=OOS_START_DATE,
        n_splits=5, training_window=756, min_val_window=20,
        allow_missing_instruments=False, persist_artifacts=True, universe_name="large",
        alpha_ema=0.33, track_turnover=True
    )
    pnl_large = results_large['combined']
    benchmark = y_full.loc[pnl_large.index]
    burn_in_end_date = pnl_large.index[BURN_IN_DAYS] if len(pnl_large) > BURN_IN_DAYS else pnl_large.index[-1]
    quality = monitor_implementation_quality(results_large, y_full)
    print("\n--- Implementation Quality (Returns-based, Large Only) ---")
    print(quality)
    metrics_oos = calculate_detailed_metrics(pnl_large.loc[OOS_START_DATE:], benchmark.loc[OOS_START_DATE:], "OOS", apply_burnin=False)
    print("\n--- Large OOS Performance (Returns-based) ---")
    for k, v in metrics_oos.items():
        if k != "Period": print(f"{k}: {v:.3f}")
    quality.to_csv(RB_DIR / f"implementation_quality_{SUFFIX}.csv")
    pd.DataFrame([metrics_oos]).to_csv(RB_DIR / f"oos_performance_{SUFFIX}.csv", index=False)
    create_returns_based_visualizations(
        ret_model=pnl_large,
        pnl_large=pnl_large,
        benchmark=benchmark,
        burn_in_end_date=burn_in_end_date,
        BURN_IN_DAYS=BURN_IN_DAYS,
        BURN_IN_MONTHS=BURN_IN_MONTHS,
        OOS_START_DATE=OOS_START_DATE,
        custom_risk_config=custom_risk_config,
        ANALYSIS_OUT=RB_DIR,
        show_plots=SHOW_PLOTS
    )
    try:
        run_extended_diagnostics(
            results=results_large, benchmark=benchmark, oos_start_date=OOS_START_DATE,
            analysis_out=RB_DIR, show_plots=SHOW_PLOTS
        )
    except Exception as e:
        print(f"Extended diagnostics failed: {e}")
    return results_large, pnl_large

def run():
    logger.info("Starting BTOP50 Returns-Based Pipeline (Large-only visuals; returns_based filenames)")
    try:
        logger.info("Step 1: BTOP50 data…")
        y_excess = process_btop50_data()
        logger.info("Step 1a: Auditing BTOP50 excess returns…")
        audit_btop50_excess(y_excess, save_dir=RB_DIR, show_plots=SHOW_PLOTS)
        logger.info("Step 2: Futures data…")
        process_futures_data()
        logger.info("Step 3: Panel build…")
        X_panel, M_mask = build_panel()
        logger.info("Step 4: Large universe…")
        X_large = create_large_universe()
        logger.info("Step 5: Load full data for validation…")
        X_large_full = pd.read_parquet(PANEL_OUTPUT_DIR / "X_large_universe/X_large_universe.parquet").astype("float32")
        y_full = pd.read_pickle(CACHE_DIR / "y_excess_BTOP50.pkl").astype("float32")
        logger.info("Step 6: Validation (large_only) + visuals…")
        results_out, ret_out = run_btop50_validation_large_only(X_large_full, y_full)
        logger.info("Pipeline complete.")
        logger.info(f"All outputs → {RB_DIR.resolve()}")
        if SHOW_PLOTS:
            print("\nShowing any generated figures. Close the windows to end the run...")
            plt.show()
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        logger.error("Ensure the following input paths exist with data:")
        logger.error(f"  • BTOP50 Excel: {BTOP50_EXCEL_PATH}")
        logger.error(f"  • US3MT Excel:  {US3MT_EXCEL_PATH}")
        logger.error(f"  • FUTURES_INPUT_DIR (with subfolders multiple_prices_csv/, roll_calendars_csv/, csvconfig/): {FUTURES_INPUT_DIR}")
        raise
    except Exception as e:
        logger.error(f"Error in BTOP50 analysis: {e}")
        logger.error("Full traceback:", exc_info=True)
        raise

if __name__ == "__main__":
    run()