from __future__ import annotations
import os
import json
import warnings
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
from matplotlib import colors as mcolors
from . import config

PROJECT_ROOT = getattr(config, "PROJECT_ROOT", Path(__file__).resolve().parents[1])
DATA_DIR = getattr(config, "DATA_DIR", PROJECT_ROOT / "data")
FIGURES_DIR = getattr(config, "FIGURES_DIR", PROJECT_ROOT / "figures")
VIZ_DIR_ROOT = FIGURES_DIR / "btop50_visualizations" / "process_based_results"
VIZ_DIR = VIZ_DIR_ROOT / "BTOP50"
VIZ_DIR.mkdir(parents=True, exist_ok=True)
PANEL_OUTPUT_DIR = VIZ_DIR_ROOT / "panel"
DEFAULT_XLARGE_PARQUET = PANEL_OUTPUT_DIR / "X_large_universe" / "X_large_universe.parquet"
ADJUSTED_OUT_DIR = VIZ_DIR_ROOT / "adjusted_prices_csv"
CARRY_OUT_DIR = VIZ_DIR_ROOT / "carry_signals_csv"
TBILL_XLSX = getattr(config, "TBILL_XLSX", DATA_DIR / "US3MT=RR.xlsx")
BTOP_XLSX = getattr(config, "BTOP50_INPUT_PATH_V6", DATA_DIR / "BTOP50_Index_historical_data(6).xlsx")

warnings.filterwarnings("ignore")
if not logging.getLogger().handlers:
    logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"), format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("btop_process")
plt.rcParams.update({"figure.figsize": (14, 8), "savefig.dpi": 300, "figure.dpi": 110, "font.family": "serif", "axes.titlesize": 14, "axes.labelsize": 11, "legend.fontsize": 9})
SHOW_FIGS = bool(int(os.getenv("SHOW_FIGS", "1")))

def _save_only(fig, path: Path):
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def _to_utc_index(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(idx))
    return idx.tz_convert("UTC") if getattr(idx, "tz", None) else idx.tz_localize("UTC")

def _to_naive(dt_or_idx):
    if isinstance(dt_or_idx, pd.DatetimeIndex):
        return dt_or_idx.tz_convert(None) if getattr(dt_or_idx, 'tz', None) else dt_or_idx
    dt = pd.to_datetime(dt_or_idx)
    return dt.tz_convert(None) if getattr(dt, 'tzinfo', None) else dt

def ann(x: pd.Series) -> float:
    return float(x.std() * np.sqrt(252))

def safe_corr(a: pd.Series, b: pd.Series) -> float:
    a, b = a.align(b, join="inner")
    a, b = a.dropna(), b.dropna()
    common = a.index.intersection(b.index)
    if len(common) < 30:
        return np.nan
    aa, bb = a.loc[common], b.loc[common]
    if aa.std() < 1e-12 or bb.std() < 1e-12:
        return np.nan
    return float(aa.corr(bb))

def align_to_common_span(y: pd.Series, X: pd.DataFrame) -> Tuple[pd.DatetimeIndex, pd.Series, pd.DataFrame]:
    y_nonan = y.dropna()
    X_nonan = X.dropna(how="all")
    if y_nonan.empty or X_nonan.empty:
        cal = pd.date_range(y.index.min(), y.index.max(), freq="B", tz="UTC")
        return cal, y, X
    start = max(y_nonan.index.min(), X_nonan.index.min())
    end = min(y_nonan.index.max(), X_nonan.index.max())
    cal2 = pd.date_range(start, end, freq="B", tz="UTC")
    return cal2, y.reindex(cal2), X.reindex(cal2)

def build_tbill_rf_daily(tbill_xlsx: Path, sheet: Optional[str] = None, col: Optional[str] = None) -> pd.Series:
    assert Path(tbill_xlsx).exists(), f"Missing T-bill file: {tbill_xlsx}"
    def _pick_date_column(df: pd.DataFrame) -> Optional[str]:
        for c in df.columns:
            cl = str(c).strip().lower()
            if "date" in cl or "time" in cl:
                return c
        dtc = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
        if len(dtc) > 0:
            return dtc[0]
        cand = df.columns[0]
        parsed = pd.to_datetime(df[cand], errors="coerce")
        if parsed.notna().sum() >= max(30, int(0.3*len(parsed))):
            df[cand] = parsed
            return cand
        return None
    def _pick_yield_column(df: pd.DataFrame, date_col: str) -> Optional[Tuple[str, pd.Series]]:
        best = None
        BEST_TOKENS = ("yield", "rate", "last", "px_last", "value", "close", "price")
        for c in df.columns:
            if c == date_col:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            nn = int(s.notna().sum())
            if nn < 30:
                continue
            name = str(c).lower()
            bonus = 1000 if any(tok in name for tok in BEST_TOKENS) else 0
            score = nn + bonus
            if (best is None) or (score > best[0]):
                best = (score, c, s)
        if best is None:
            return None
        return best[1], best[2]
    xls = pd.ExcelFile(tbill_xlsx)
    candidate_sheets = []
    if sheet and sheet in xls.sheet_names:
        candidate_sheets = [sheet]
    else:
        for nm in ("Table Data", "Data", "DATA", "Sheet1"):
            if nm in xls.sheet_names:
                candidate_sheets.append(nm)
        candidate_sheets += [s for s in xls.sheet_names if s not in candidate_sheets]
    chosen = None
    chosen_meta = None
    for sh in candidate_sheets:
        df0 = pd.read_excel(xls, sheet_name=sh)
        if df0 is None or df0.empty:
            continue
        date_col = _pick_date_column(df0)
        if not date_col:
            continue
        df = df0.dropna(subset=[date_col]).copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        if df.empty:
            continue
        if col and col in df.columns:
            y_raw = pd.to_numeric(df[col], errors="coerce")
            if y_raw.notna().sum() < 30:
                picked = _pick_yield_column(df, date_col)
                if picked is None:
                    continue
                col_name, y_raw = picked
            else:
                col_name = col
        else:
            picked = _pick_yield_column(df, date_col)
            if picked is None:
                continue
            col_name, y_raw = picked
        idx = _to_utc_index(df[date_col])
        s = pd.Series(y_raw.values, index=idx).sort_index()
        med = float(np.nanmedian(s.values))
        unit = "decimal"
        if med > 100:
            s = s / 10000.0
            unit = "bps"
        elif med > 1.5:
            s = s / 100.0
            unit = "percent"
        cal = pd.date_range(s.index.min(), s.index.max(), freq="B", tz="UTC")
        rf_daily = (s.reindex(cal).ffill() / 360.0).astype("float64")
        chosen = rf_daily
        chosen_meta = (sh, col_name, unit, cal[0].date(), cal[-1].date())
        break
    if chosen is None:
        raise KeyError("Could not find a usable numeric yield series in the T-bill workbook.")
    sh, col_name, unit, d0, d1 = chosen_meta
    print(f"T-bill autodetect → sheet='{sh}', column='{col_name}', units='{unit}', span {d0} → {d1}")
    return chosen

def build_btop50_excess_from_excel(btop_xlsx: Path, rf_daily: pd.Series, use_returns_directly: bool = True, filter_start=pd.Timestamp("2010-01-01", tz="UTC"), filter_end=pd.Timestamp("2024-03-29", tz="UTC")) -> pd.Series:
    assert btop_xlsx.exists(), f"Missing BTOP50 Excel: {btop_xlsx}"
    df = pd.read_excel(btop_xlsx)
    date_cands = [c for c in df.columns if str(c).strip().lower() in ("dstamp", "date", "dates", "time", "timestamp")]
    date_col = date_cands[0] if date_cands else df.columns[0]
    df = df.dropna(subset=[date_col]).copy()
    df["Date"] = _to_utc_index(pd.to_datetime(df[date_col], errors="coerce"))
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df = df.loc[filter_start:filter_end]
    if use_returns_directly and ("ROR" in df.columns):
        ret = pd.to_numeric(df["ROR"], errors="coerce")
    else:
        price_col = 100 if 100 in df.columns else next((c for c in ("Last Price","PX_LAST","Close","Price","CLOSE","Last") if c in df.columns), None)
        if price_col is None:
            raise KeyError("BTOP50 Excel does not contain 'ROR' or a recognizable price/index column.")
        px = pd.to_numeric(df[price_col], errors="coerce").dropna()
        ret = px.pct_change()
    idx_b = pd.date_range(ret.index.min(), ret.index.max(), freq="B", tz="UTC")
    ret = ret.reindex(idx_b).asfreq("B").ffill(limit=1).bfill(limit=1)
    mu, sg = ret.mean(), ret.std()
    ret = ret.clip(lower=mu - 5*sg, upper=mu + 5*sg)
    rf_aligned = rf_daily.reindex(ret.index).ffill().bfill()
    excess = (ret - rf_aligned).astype("float32")
    return excess

@dataclass
class CFG:
    OUT_DIR: Path = VIZ_DIR
    FUT_PRICE_DIR: Path | None = None
    FUT_ROLL_DIR: Path | None = None
    FUT_CFG_DIR: Path | None = None
    FUT_OUT_DIR: Path | None = None
    CARRY_SIGNALS_DIR: Path | None = None
    XLARGE_PATH: Path = DEFAULT_XLARGE_PARQUET
    FORCE_REBUILD_XLARGE: bool = False
    LOOKBACKS: np.ndarray = field(default_factory=lambda: np.array([5,10,15,16,20,30,32,40,60,80,90,120,150,160,180,220,260], dtype=int))
    KEEP_BREAKOUT: Tuple[int, ...] = (80, 160)
    KEEP_SKEWABS: Tuple[int, ...] = (180,)
    KEEP_ACCEL: Tuple[int, ...] = (16, 32)
    EWMA_SPAN_VOL: int = 40
    Z_CAP: float = 3.0
    PRE_SCALE_TARGET: float = 0.20
    PRE_SCALE_CAP: float = 3.0
    PRE_SCALE_FLOOR: float = 0.30
    FINAL_VOL_TARGET: float = 0.085
    FINAL_SCALE_CAP: float = 2.0
    FINAL_SCALE_FLOOR: float = 0.25
    FINAL_SCALE_WINSOR: float = 0.001
    ANN_DAYS: int = 252
    LAMBDA_GRID: np.ndarray = field(default_factory=lambda: np.logspace(-2, 0, 31))
    LAMBDA_FIX_DATE: Optional[pd.Timestamp] = None
    CV_FOLDS: int = 5
    PURGE_GAP_DAYS: int = 21
    POSITIVE: bool = True
    L1_TARGET: float = 1.0
    MAX_SINGLE_WEIGHT: float = 0.05
    WEIGHT_EMA_SPAN: int = 5
    STARTUP_DELAY_DAYS: int = 20
    ROLL_CORR_WIN: int = 252
    ROLL_TE_WIN: int = 252
    REPL_TRADE_COST_BPS: float = 1.0
    OVERLAY_TRADE_COST_BPS: float = 1.0
    SOFT_START_DAYS: int = 5
    FILTER_START: pd.Timestamp = pd.Timestamp("2000-01-03", tz="UTC")
    FILTER_END: pd.Timestamp = pd.Timestamp("2024-03-29", tz="UTC")
    ROLL_OFFSET_BD: int = 5
    OOS_START: pd.Timestamp = pd.Timestamp("2021-11-01", tz="UTC")
    BURN_IN_DAYS: int = 189
    BURN_IN_MONTHS: int = 9
    USE_CARRY: bool = True
    KEEP_CARRY: Tuple[int, ...] = (5, 20, 60, 120)
    CARRY_CLIP_ABS_ANNUAL: float = 2.0
    GATE_OVERRIDE: Optional[pd.Timestamp] = pd.Timestamp("2013-06-30", tz="UTC")
    def __post_init__(self):
        self.OUT_DIR.mkdir(parents=True, exist_ok=True)
        (self.OUT_DIR / "diagnostics").mkdir(exist_ok=True)
        self._discover_or_set_futures_dirs()
    def _discover_or_set_futures_dirs(self):
        BASE_DIR = PROJECT_ROOT
        if all([self.FUT_PRICE_DIR, self.FUT_ROLL_DIR, self.FUT_CFG_DIR]):
            for p in [self.FUT_PRICE_DIR, self.FUT_ROLL_DIR, self.FUT_CFG_DIR]:
                assert Path(p).exists(), f"Missing: {p}"
            if not self.CARRY_SIGNALS_DIR:
                self.CARRY_SIGNALS_DIR = Path(self.FUT_CFG_DIR).parent / "carry_signals_csv"
        else:
            search_roots = [BASE_DIR, DATA_DIR, BASE_DIR / "data", BASE_DIR.parent if BASE_DIR.parent.exists() else BASE_DIR]
            need = {"multiple_prices_csv": None, "roll_calendars_csv": None, "csvconfig": None, "carry_signals_csv": None}
            for root in search_roots:
                if not root.exists():
                    continue
                for sub in list(need.keys()):
                    if need[sub] is None:
                        try:
                            hit = next((p for p in root.rglob(sub) if p.is_dir()), None)
                        except Exception:
                            hit = None
                        if hit is not None:
                            need[sub] = hit
            self.FUT_PRICE_DIR = need["multiple_prices_csv"]
            self.FUT_ROLL_DIR = need["roll_calendars_csv"]
            self.FUT_CFG_DIR = need["csvconfig"]
            self.CARRY_SIGNALS_DIR = need["carry_signals_csv"] or (self.FUT_CFG_DIR.parent / "carry_signals_csv" if self.FUT_CFG_DIR else None)
            missing = [("FUT_PRICE_DIR", self.FUT_PRICE_DIR), ("FUT_ROLL_DIR", self.FUT_ROLL_DIR), ("FUT_CFG_DIR", self.FUT_CFG_DIR)]
            missing = [k for k, v in missing if (v is None or not Path(v).exists())]
            if missing:
                print("Could not auto-discover required futures input folders: " + ", ".join(missing))
                print("Expected to find subfolders somewhere under your project tree:")
                print("  'multiple_prices_csv', 'roll_calendars_csv', 'csvconfig'")
                print(f"If you have a prebuilt parquet, ensure it exists at:\n  {self.XLARGE_PATH}")
            else:
                print("Futures inputs:")
                print("  prices :", self.FUT_PRICE_DIR)
                print("  rolls  :", self.FUT_ROLL_DIR)
                print("  cfg    :", self.FUT_CFG_DIR)
        self.FUT_OUT_DIR = ADJUSTED_OUT_DIR
        self.FUT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        if self.CARRY_SIGNALS_DIR is None or not Path(self.CARRY_SIGNALS_DIR).exists():
            self.CARRY_SIGNALS_DIR = CARRY_OUT_DIR
        Path(self.CARRY_SIGNALS_DIR).mkdir(parents=True, exist_ok=True)
        print("Per-instrument adjusted CSVs →", self.FUT_OUT_DIR)
        print("Carry signals directory     →", self.CARRY_SIGNALS_DIR)

CFG = CFG()

def rescale_series_to_target(ret: pd.Series, target: float, span: int = CFG.EWMA_SPAN_VOL, cap: float = np.inf, floor: float = 0.0, keep_na: bool = True) -> pd.Series:
    sigma = ret.ewm(span=span, adjust=False).std().shift(1) * np.sqrt(CFG.ANN_DAYS)
    sigma = sigma.replace(0, np.nan)
    scale = (target / sigma).clip(lower=floor, upper=cap)
    out = ret * scale
    return out if keep_na else out.fillna(0.0)

def renorm_with_cap(beta: pd.Series, cap: float, l1_target: float) -> pd.Series:
    w = beta.clip(lower=0.0).astype(float)
    s = w.sum()
    if s <= 0:
        return w
    w /= s
    for _ in range(200):
        over = w > cap + 1e-15
        if over.any():
            w.loc[over] = cap
        total = float(w.sum())
        if abs(total - l1_target) < 1e-12:
            break
        if total > l1_target + 1e-12:
            w *= (l1_target / total)
            continue
        residual = l1_target - total
        if residual <= 1e-12:
            break
        under = w < cap - 1e-15
        if not under.any():
            w *= (l1_target / float(w.sum()))
            break
        base = beta.clip(lower=0.0).astype(float).loc[under]
        if base.sum() <= 1e-15:
            w.loc[under] += residual / int(under.sum())
        else:
            w.loc[under] += residual * (base / float(base.sum()))
    w = w.clip(lower=0.0, upper=cap)
    s = float(w.sum())
    if s > 0:
        w *= min(1.0, l1_target / s)
    return w

def availability_adjust_weights(W: pd.DataFrame, R: pd.DataFrame, cap: float = CFG.MAX_SINGLE_WEIGHT, l1_target: float = CFG.L1_TARGET) -> pd.DataFrame:
    W = W.reindex(columns=R.columns, fill_value=0.0)
    live = R.notna()
    out = pd.DataFrame(0.0, index=W.index, columns=W.columns)
    for t in W.index:
        m = live.loc[t] if t in live.index else pd.Series(False, index=W.columns)
        if not m.any():
            continue
        w = W.loc[t].where(m, 0.0)
        n_live = int(m.sum())
        cap_eff = max(cap, (l1_target / max(n_live, 1)) + 1e-12)
        out.loc[t] = renorm_with_cap(w, cap_eff, l1_target)
    return out

def winsorise_series(s: pd.Series, z: float = 5.0, window: int = 2520, min_window: int = 504) -> Tuple[pd.Series, int, float]:
    if s.notna().sum() < 250:
        return s, 0, 0.0
    cutoff_date = pd.Timestamp("2003-01-01", tz="UTC")
    mu = pd.Series(index=s.index, dtype="float64")
    sg = pd.Series(index=s.index, dtype="float64")
    early = s.index < cutoff_date
    if early.any():
        roll_early = s.rolling(window=window, min_periods=max(min_window, window//2))
        mu[early] = roll_early.mean()[early].fillna(s.expanding(min_periods=250).mean()[early])
        sg[early] = roll_early.std()[early].fillna(s.expanding(min_periods=250).std()[early])
    late = ~early
    if late.any():
        roll_late = s.rolling(window=window, min_periods=window)
        mu[late] = roll_late.mean()[late].fillna(s.expanding(min_periods=250).mean()[late])
        sg[late] = roll_late.std()[late].fillna(s.expanding(min_periods=250).std()[late])
    mask = mu.notna() & sg.notna()
    clipped = s.copy()
    lower = mu[mask] - z * sg[mask]
    upper = mu[mask] + z * sg[mask]
    before = s[mask]
    clipped.loc[mask] = before.clip(lower=lower, upper=upper)
    n_clip = int(((before < lower) | (before > upper)).sum())
    pct = float((n_clip / max(mask.sum(), 1) * 100)) if mask.sum() else 0.0
    if pct > 1.0 and s.name:
        log.info(f"Winsorization: {s.name} had {pct:.1f}% clipped")
    return clipped, n_clip, pct

def reshape_to_long(path: Path, root: str) -> pd.DataFrame:
    wide = pd.read_csv(path)
    def pick_first(cands):
        for c in cands:
            if c in wide.columns:
                return c
        return None
    date_col = pick_first(["DATETIME","DATE_TIME","DATE","Date"])
    price_col = pick_first(["PRICE","PX_LAST","CLOSE","Close","Last Price"])
    pcon_col = pick_first(["PRICE_CONTRACT","CONTRACT","PriceContract","Contract"])
    if not (date_col and price_col and pcon_col):
        raise KeyError(f"Missing required columns in {path.name} (need DATE/PRICE/PRICE_CONTRACT variants)")
    carry_col = pick_first(["CARRY","CARRY_LAST","Carry"])
    ccon_col = pick_first(["CARRY_CONTRACT","CarryContract"])
    fwd_col = pick_first(["FORWARD","FORWARD_LAST","Forward"])
    fcon_col = pick_first(["FORWARD_CONTRACT","ForwardContract"])
    df = wide.copy()
    UNIT_FIX = {"Silver": .01, "GAS_": .01, "Corn_mini": .01, "Wheat_mini": .01, "Sugar": .01, "HG1": .01, "RB": .01}
    fac = next((UNIT_FIX[p] for p in UNIT_FIX if root.startswith(p.rstrip("_"))), 1)
    price_rows = (df[[date_col, pcon_col, price_col]].rename(columns={date_col:"Date", pcon_col:"Contract", price_col:"Close"}).assign(Close=lambda d: pd.to_numeric(d["Close"], errors="coerce") * fac, Kind="Price").dropna(subset=["Close"]))
    carry_rows = pd.DataFrame(columns=["Date","Contract","Close","Kind"])
    if carry_col and ccon_col and (carry_col in df.columns) and (ccon_col in df.columns):
        carry_rows = (df[[date_col, ccon_col, carry_col]].rename(columns={date_col:"Date", ccon_col:"Contract", carry_col:"Close"}).assign(Close=lambda d: pd.to_numeric(d["Close"], errors="coerce") * fac, Kind="Carry").dropna(subset=["Close"]))
    fwd_rows = pd.DataFrame(columns=["Date","Contract","Close","Kind"])
    if fwd_col and fcon_col and (fwd_col in df.columns) and (fcon_col in df.columns):
        fwd_rows = (df[[date_col, fcon_col, fwd_col]].rename(columns={date_col:"Date", fcon_col:"Contract", fwd_col:"Close"}).assign(Close=lambda d: pd.to_numeric(d["Close"], errors="coerce") * fac, Kind="Forward").dropna(subset=["Close"]))
    long = (pd.concat([price_rows, carry_rows, fwd_rows], ignore_index=True).assign(Date=lambda d: _to_utc_index(pd.to_datetime(d["Date"]).dt.normalize())).sort_values(["Date","Contract"]))
    long = long[(long["Date"] >= CFG.FILTER_START) & (long["Date"] <= CFG.FILTER_END)]
    return long

def build_xlarge_from_csv() -> pd.DataFrame:
    if not CFG.FUT_PRICE_DIR or not Path(CFG.FUT_PRICE_DIR).exists():
        raise FileNotFoundError(f"Missing FUT_PRICE_DIR ({CFG.FUT_PRICE_DIR})")
    if not CFG.FUT_ROLL_DIR or not Path(CFG.FUT_ROLL_DIR).exists():
        raise FileNotFoundError(f"Missing FUT_ROLL_DIR ({CFG.FUT_ROLL_DIR})")
    if not CFG.FUT_CFG_DIR or not Path(CFG.FUT_CFG_DIR).exists():
        raise FileNotFoundError(f"Missing FUT_CFG_DIR ({CFG.FUT_CFG_DIR})")
    inst = pd.read_csv(Path(CFG.FUT_CFG_DIR) / "instrumentconfig.csv").set_index("Instrument")
    cost = pd.read_csv(Path(CFG.FUT_CFG_DIR) / "spreadcosts.csv").set_index("Instrument")
    rf = pd.Series(0.0, index=pd.date_range(CFG.FILTER_START, CFG.FILTER_END, freq="B", tz="UTC"))
    print("Building per-instrument excess returns from CSVs ...")
    SKIP = {"HANGTECH","IBEX"} | {p.stem for p in Path(CFG.FUT_PRICE_DIR).glob("*_micro.csv")} | {p.stem for p in Path(CFG.FUT_PRICE_DIR).glob("*_mini.csv")}
    series = {}
    for csv_path in sorted(Path(CFG.FUT_PRICE_DIR).glob("*.csv")):
        root = csv_path.stem
        if root in SKIP:
            continue
        long = reshape_to_long(csv_path, root)
        roll_file = Path(CFG.FUT_ROLL_DIR) / f"{root}.csv"
        if not roll_file.exists():
            print(f"⚠ {root}: missing roll calendar")
            continue
        roll = (pd.read_csv(roll_file, parse_dates=["DATE_TIME"]).rename(columns={"DATE_TIME":"RollDate","current_contract":"Current","next_contract":"Next"}))
        roll["ExecDate"] = (pd.to_datetime(roll["RollDate"]) - BDay(CFG.ROLL_OFFSET_BD))
        roll["ExecDate"] = _to_utc_index(roll["ExecDate"])
        roll = roll[(roll["ExecDate"] >= CFG.FILTER_START) & (roll["ExecDate"] <= CFG.FILTER_END)]
        adj_factor, factor = {}, 1.0
        for cur, nxt, ex in zip(roll["Current"][::-1], roll["Next"][::-1], roll["ExecDate"][::-1]):
            adj_factor[cur] = factor
            p_cur = long.query("Contract == @cur & Date == @ex & Kind == 'Price'")["Close"]
            p_nxt = long.query("Contract == @nxt & Date == @ex & Kind == 'Price'")["Close"]
            if not p_cur.empty and not p_nxt.empty:
                ratio = p_nxt.iat[-1] / p_cur.iat[0]
                factor *= ratio if ratio < 5 else 1/ratio
        long["AdjPrice"] = long["Close"] * long["Contract"].map(adj_factor).fillna(1.0)
        price_series = long.query("Kind == 'Price'").set_index("Date")["AdjPrice"]
        price_daily = price_series.groupby(price_series.index.normalize()).last()
        price_daily.index = _to_utc_index(price_daily.index)
        gap_mask = price_daily.index.to_series().diff().dt.days.gt(7)
        price_daily.loc[gap_mask] = np.nan
        price_daily = price_daily.ffill(limit=3)
        price_daily = price_daily.loc[CFG.FILTER_START:CFG.FILTER_END]
        has_carry = not long.query("Kind == 'Carry'").empty
        if has_carry:
            carry_df = (long.query("Kind == 'Carry'").pivot_table(index="Date", columns="Contract", values="Close", aggfunc="last"))
            carry_daily = carry_df.groupby(carry_df.index.normalize()).last()
            carry_daily.index = _to_utc_index(carry_daily.index)
            carry_ret_daily = carry_daily.diff().stack().groupby(level=0).sum()
            carry_ret_daily.index = _to_utc_index(carry_ret_daily.index)
            carry_ret_daily = (carry_ret_daily / price_daily.shift(1)).reindex(price_daily.index).fillna(0.0)
        else:
            jumps = {}
            for cur, nxt, ex in zip(roll["Current"], roll["Next"], roll["ExecDate"]):
                p_cur = long.query("Contract == @cur & Date == @ex & Kind == 'Price'")["Close"]
                p_nxt = long.query("Contract == @nxt & Date == @ex & Kind == 'Price'")["Close"]
                if not p_cur.empty and not p_nxt.empty:
                    jumps[ex.normalize()] = p_nxt.iat[0] / p_cur.iat[0] - 1
            carry_ret_daily = pd.Series(jumps, dtype=float)
            carry_ret_daily.index = _to_utc_index(carry_ret_daily.index)
            carry_ret_daily = carry_ret_daily.reindex(price_daily.index).fillna(0.0)
        price_ret_daily = price_daily.pct_change()
        total_ret = (price_ret_daily + carry_ret_daily).dropna()
        try:
            row_ic = inst.loc[root]
            tick_sz = row_ic.get("TickSize", 0.01)
            spread = float(cost.loc[root, "SpreadCost"]) if root in cost.index else 0.0
        except Exception:
            tick_sz, spread = 0.01, 0.0
        roll_days = pd.Index(sorted(set(roll["ExecDate"].dt.normalize().tolist())))
        roll_flag = pd.Series(0, index=price_daily.index, dtype=int)
        roll_flag.loc[price_daily.index.isin(roll_days)] = 1
        if spread >= 1:
            spread_pts = spread * tick_sz
            tick_pct = (spread_pts / price_daily.shift(1))
        else:
            tick_pct = pd.Series(spread/1e4, index=price_daily.index)
        daily_cost = (tick_pct.fillna(0.0) * roll_flag).reindex(total_ret.index).fillna(0.0).shift(1).fillna(0.0)
        rf2 = pd.Series(0.0, index=total_ret.index)
        excess = (total_ret - daily_cost - rf2.reindex(total_ret.index).fillna(0.0)).astype("float64")
        excess = excess.loc[CFG.FILTER_START:CFG.FILTER_END]
        excess.name = root
        excess, _, pct_c = winsorise_series(excess)
        ann_vol = excess.std()*np.sqrt(252)
        print(f"✔ {root:<14}| rows {len(excess):6,d} | σ {ann_vol:.2%} | clipped {pct_c:.1f}%")
        Path(CFG.FUT_OUT_DIR).mkdir(parents=True, exist_ok=True)
        excess.to_csv(Path(CFG.FUT_OUT_DIR) / f"{root}_excess_net_ret.csv", float_format="%.8f")
        series[root] = excess
    if not series:
        raise RuntimeError("No futures series built. Check CSV inputs / roll calendars.")
    X = pd.DataFrame(series).sort_index()
    X.index = _to_utc_index(X.index)
    Path(CFG.XLARGE_PATH).parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(CFG.XLARGE_PATH, compression="snappy")
    log.info(f"Saved X_large_universe to {CFG.XLARGE_PATH} with shape {X.shape}")
    return X

def load_xlarge() -> pd.DataFrame:
    if (not CFG.FORCE_REBUILD_XLARGE) and Path(CFG.XLARGE_PATH).exists():
        X = pd.read_parquet(CFG.XLARGE_PATH)
        X.index = _to_utc_index(X.index)
        X = X[~X.index.duplicated(keep="last")].sort_index()
        ann_vol = (X.std() * np.sqrt(252)).describe(percentiles=[.25,.5,.75])
        print("ℹ Sanity: per-contract ann vol (from parquet)\n", ann_vol.to_string())
        return X
    return build_xlarge_from_csv()

PAPER_27 = ["AD","BP","CD","EC","JY","SF","TU","FV","TY","US","G","RX","H","ES","NQ","YM","Z","VG","AP","NKD","GC","SI","HG","CL","HO","NG","XB"]
PB_TO_RB_MAPPING = {
    "AD":"AUD","BP":"GBP","CD":"CAD","EC":"EUR","JY":"JPY","SF":"CHF",
    "TU":"US2Y","FV":"US5Y","TY":"US10Y","US":"US","G":"GILT","RX":"BUND","H":"BOBL",
    "ES":"S&P500","NQ":"NASDAQ","YM":"DOW","Z":"FTSE","VG":"EUROSTOXX","AP":"ASX200","NKD":"TOPIX",
    "GC":"GOLD","SI":"SILVER","HG":"COPPER","CL":"WTI_CRUDE","HO":"HEATOIL","NG":"NATGAS","XB":"RBOB"
}
ALIAS = {
    "AD": ["AUD","AD","6A"],
    "BP": ["GBP","BP","6B"],
    "CD": ["CAD","CD","6C"],
    "EC": ["EUR","EC","6E"],
    "JY": ["JPY","JY","6J"],
    "SF": ["CHF","SF","6S"],
    "TU": ["US2Y","US2","TU","SHATZ"],
    "FV": ["US5Y","US5","FV","BOBL"],
    "TY": ["US10Y","US10","TY","OAT"],
    "US": ["US","US30","US30Y","US20","BUXL"],
    "G":  ["GILT","LONG_GILT","UK_GILT","G"],
    "RX": ["BUND","RX"],
    "H":  ["BOBL","SHATZ","H"],
    "ES": ["S&P500","SP500","SPX","ES"],
    "NQ": ["NASDAQ","NQ"],
    "YM": ["DOW","DJI","YM"],
    "Z":  ["FTSE","FTSE100","Z"],
    "VG": ["EUROSTOXX","EUROSTX","EURO600","VG"],
    "AP": ["ASX200","SPI200","AP"],
    "NKD": ["TOPIX","NIKKEI","NI","JP-REALESTATE","NKD"],
    "GC": ["GOLD","GOLD-mini","GC"],
    "SI": ["SILVER","SI"],
    "HG": ["COPPER","COPPER-mini","COPPER-micro","COPPER_LME","HG"],
    "CL": ["WTI_CRUDE","CRUDE_W","CRUDE_ICE","BRENT_W","BRENT_CRUDE","BRENT-LAST","BRE","CL"],
    "HO": ["HEATOIL","GASOIL","HEATOIL-ICE","HO"],
    "NG": ["NATGAS","GAS_US","GAS-LAST","GAS-PEN","NG"],
    "XB": ["RBOB","GASOLINE","GASOILINE","GASOILINE_ICE","RB","XB"]
}

def pick_alias_columns(X_all: pd.DataFrame) -> Tuple[Dict[str,str], List[str], List[str]]:
    chosen, hits, misses = {}, [], []
    present = set(X_all.columns)
    for code in PAPER_27:
        rb = PB_TO_RB_MAPPING.get(code, code)
        sel = next((a for a in ALIAS.get(code, []) if a in present), None)
        if sel is None:
            misses.append(code)
        else:
            chosen[rb] = sel
            hits.append(rb)
    return chosen, hits, misses

def load_carry_signals(carry_dir: Path | None, chosen_map: Dict[str, str], keep_spans: Tuple[int, ...], cal: pd.DatetimeIndex) -> pd.DataFrame:
    if carry_dir is None or not Path(carry_dir).exists():
        return pd.DataFrame(index=cal)
    data: Dict[Tuple[str, int], pd.Series] = {}
    for rb_name, sym in chosen_map.items():
        path = Path(carry_dir) / f"{sym}_carry_signals.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "Date" not in df.columns:
            continue
        idx = _to_utc_index(pd.to_datetime(df["Date"], errors="coerce"))
        for span in keep_spans:
            cand = [f"Carry{int(span)}", f"carry{int(span)}", f"carry_{int(span)}", f"CARRY{int(span)}", f"CARRY_{int(span)}"]
            col = next((c for c in cand if c in df.columns), None)
            if col:
                s = pd.to_numeric(df[col], errors="coerce")
                ser = pd.Series(s.values, index=idx).reindex(cal)
                data[(rb_name, int(span))] = ser
    if not data:
        return pd.DataFrame(index=cal)
    cols = pd.MultiIndex.from_tuples(list(data.keys()), names=["contract","lookback"])
    C = pd.DataFrame(data, index=cal)
    C.columns = cols
    return C

def _build_carry_for_instrument(root: str, price_csv_dir: Path, roll_csv_dir: Path, spans: Tuple[int, ...], filter_start: pd.Timestamp, filter_end: pd.Timestamp) -> Optional[pd.DataFrame]:
    price_path = Path(price_csv_dir) / f"{root}.csv"
    roll_path = Path(roll_csv_dir) / f"{root}.csv"
    if (not price_path.exists()) or (not roll_path.exists()):
        return None
    long = reshape_to_long(price_path, root)
    P = (long.query("Kind == 'Price'").pivot(index="Date", columns="Contract", values="Close").sort_index())
    P.index = _to_utc_index(P.index)
    cal = pd.date_range(max(P.index.min(), filter_start), min(P.index.max(), filter_end), freq="B", tz="UTC")
    P = P.reindex(cal).ffill(limit=3)
    roll = (pd.read_csv(roll_path, parse_dates=["DATE_TIME"]).rename(columns={"DATE_TIME":"RollDate","current_contract":"Current","next_contract":"Next"}))
    roll["ExecDate"] = _to_utc_index(pd.to_datetime(roll["RollDate"]) - BDay(CFG.ROLL_OFFSET_BD))
    roll = roll[(roll["ExecDate"] >= filter_start) & (roll["ExecDate"] <= filter_end)].sort_values("ExecDate")
    if roll.empty:
        return None
    starts: List[pd.Timestamp] = []
    ends: List[pd.Timestamp] = []
    curs: List[str] = []
    nxts: List[str] = []
    execs = roll["ExecDate"].to_list()
    for i in range(len(roll)):
        end = roll.iloc[i]["ExecDate"]
        start = cal.min() if i == 0 else (roll.iloc[i-1]["ExecDate"] + BDay(1))
        start = _to_utc_index(start)[0]
        end = _to_utc_index(end)[0]
        if end < cal.min():
            continue
        starts.append(max(start, cal.min()))
        ends.append(min(end, cal.max()))
        curs.append(str(roll.iloc[i]["Current"]))
        nxts.append(str(roll.iloc[i]["Next"]))
    carry_ann = pd.Series(np.nan, index=cal, dtype="float64")
    for start, end, curc, nextc in zip(starts, ends, curs, nxts):
        if curc not in P.columns or nextc not in P.columns:
            continue
        dates = cal[(cal >= start) & (cal <= end)]
        if len(dates) == 0:
            continue
        p_cur = P[curc].reindex(dates)
        p_nxt = P[nextc].reindex(dates)
        rel = (p_nxt / p_cur) - 1.0
        dtr = (end - dates) / np.timedelta64(1, "D")
        dtr = np.maximum(dtr.astype(float), 1.0)
        ann_yield = (rel.values * (365.0 / dtr)).astype(float)
        s = pd.Series(ann_yield, index=dates).clip(lower=-CFG.CARRY_CLIP_ABS_ANNUAL, upper=+CFG.CARRY_CLIP_ABS_ANNUAL)
        carry_ann.loc[dates] = s
    if carry_ann.notna().sum() == 0:
        return None
    out = pd.DataFrame(index=cal)
    for span in sorted(set(int(s) for s in spans)):
        out[f"Carry{span}"] = carry_ann.ewm(span=span, adjust=False, min_periods=span).mean()
    out = out.reset_index().rename(columns={"index":"Date"})
    return out

def build_carry_signals_for_aliases(chosen_map: Dict[str, str], rebuild: bool = False) -> None:
    Path(CFG.CARRY_SIGNALS_DIR).mkdir(parents=True, exist_ok=True)
    targets = sorted(set(chosen_map.values()))
    built, skipped = 0, 0
    for sym in targets:
        out_path = Path(CFG.CARRY_SIGNALS_DIR) / f"{sym}_carry_signals.csv"
        if out_path.exists() and not rebuild:
            skipped += 1
            continue
        df = _build_carry_for_instrument(sym, price_csv_dir=Path(CFG.FUT_PRICE_DIR), roll_csv_dir=Path(CFG.FUT_ROLL_DIR), spans=CFG.KEEP_CARRY, filter_start=CFG.FILTER_START, filter_end=CFG.FILTER_END)
        if df is None or df.empty:
            print(f"⚠ carry-builder: could not build for {sym} (missing data).")
            continue
        try:
            df.to_csv(out_path, index=False, float_format="%.8f")
            built += 1
        except Exception as e:
            print(f"⚠ carry-builder: failed to write {out_path.name}: {e}")
    print(f"Carry-builder summary → built: {built}, skipped(existing): {skipped}, dir={CFG.CARRY_SIGNALS_DIR}")

def _risk_scale_to_pre_target(strat_pre: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=strat_pre.index, columns=strat_pre.columns, dtype=float)
    for col in strat_pre.columns:
        out[col] = rescale_series_to_target(strat_pre[col], target=CFG.PRE_SCALE_TARGET, span=CFG.EWMA_SPAN_VOL, cap=CFG.PRE_SCALE_CAP, floor=CFG.PRE_SCALE_FLOOR, keep_na=True)
    return out

def build_strategy_families(returns_panel: pd.DataFrame, trend_lbs: np.ndarray, breakout_lbs: Tuple[int, ...], skewabs_lbs: Tuple[int, ...], accel_lbs: Tuple[int, ...], carry_panel: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    r = returns_panel.copy().fillna(0.0)
    sigma40_ann = r.ewm(span=CFG.EWMA_SPAN_VOL, adjust=False).std() * np.sqrt(CFG.ANN_DAYS)
    sigma40_ann = sigma40_ann.shift(1).replace(0, np.nan)
    pieces = []
    for lb in trend_lbs:
        mu = r.ewm(span=int(lb), adjust=False, min_periods=int(lb)).mean()
        sd = r.ewm(span=int(lb), adjust=False, min_periods=int(lb)).std().clip(lower=1e-8)
        z = (mu / sd).clip(-CFG.Z_CAP, CFG.Z_CAP).ewm(span=max(2, lb // 4), adjust=False).mean()
        w = z.shift(1) / sigma40_ann
        strat_pre = w * r
        strat_scaled = _risk_scale_to_pre_target(strat_pre)
        strat_scaled.columns = pd.MultiIndex.from_product([["TREND"], list(returns_panel.columns), [lb]], names=["family","contract","lookback"])
        pieces.append(strat_scaled)
    if breakout_lbs:
        price_idx = (1.0 + r.fillna(0.0)).cumprod()
        for lb in breakout_lbs:
            roll_max = price_idx.shift(1).rolling(lb, min_periods=lb).max()
            roll_min = price_idx.shift(1).rolling(lb, min_periods=lb).min()
            sig = pd.DataFrame(np.where(price_idx.gt(roll_max), 1.0, np.where(price_idx.lt(roll_min), -1.0, 0.0)), index=r.index, columns=r.columns)
            z = sig.ewm(span=max(2, lb // 4), adjust=False).mean().clip(-1.0, 1.0)
            w = z.shift(1) / sigma40_ann
            strat_pre = w * r
            strat_scaled = _risk_scale_to_pre_target(strat_pre)
            strat_scaled.columns = pd.MultiIndex.from_product([["BREAKOUT"], list(returns_panel.columns), [lb]], names=["family","contract","lookback"])
            pieces.append(strat_scaled)
    if skewabs_lbs:
        for lb in skewabs_lbs:
            sk = r.rolling(lb, min_periods=lb).skew()
            z = sk.clip(-CFG.Z_CAP, CFG.Z_CAP).ewm(span=max(2, lb // 4), adjust=False).mean()
            w = z.shift(1) / sigma40_ann
            strat_pre = w * r
            strat_scaled = _risk_scale_to_pre_target(strat_pre)
            strat_scaled.columns = pd.MultiIndex.from_product([["SKEWABS"], list(returns_panel.columns), [lb]], names=["family","contract","lookback"])
            pieces.append(strat_scaled)
    if accel_lbs:
        for lb in accel_lbs:
            short = r.ewm(span=int(lb), adjust=False, min_periods=int(lb)).mean()
            long = r.ewm(span=int(4*lb), adjust=False, min_periods=int(4*lb)).mean()
            sd_s = r.ewm(span=int(lb), adjust=False, min_periods=int(lb)).std().clip(lower=1e-8)
            z = ((short - long) / sd_s).clip(-CFG.Z_CAP, CFG.Z_CAP).ewm(span=max(2, lb // 4), adjust=False).mean()
            w = z.shift(1) / sigma40_ann
            strat_pre = w * r
            strat_scaled = _risk_scale_to_pre_target(strat_pre)
            strat_scaled.columns = pd.MultiIndex.from_product([["ACCEL"], list(returns_panel.columns), [lb]], names=["family","contract","lookback"])
            pieces.append(strat_scaled)
    if carry_panel is not None and not carry_panel.empty:
        carry_pre: Dict[Tuple[str, str, int], pd.Series] = {}
        for (contract, lb) in carry_panel.columns:
            if contract not in r.columns:
                continue
            sig = carry_panel[(contract, lb)].astype(float)
            w = sig.shift(1) / sigma40_ann[contract]
            carry_pre[("CARRY", contract, int(lb))] = (w * r[contract])
        if carry_pre:
            df = pd.DataFrame(carry_pre, index=r.index)
            df.columns = pd.MultiIndex.from_tuples(df.columns, names=["family","contract","lookback"])
            carry_scaled = _risk_scale_to_pre_target(df)
            pieces.append(carry_scaled)
            print("   CARRY family added.")
    Rfam = pd.concat(pieces, axis=1).sort_index(axis=1)
    return Rfam

def cv_lambda_on_corr(R: pd.DataFrame, y: pd.Series, lambdas: np.ndarray, n_folds: int = 5, end_date: Optional[pd.Timestamp] = None, purge_gap_days: int = CFG.PURGE_GAP_DAYS) -> float:
    common = R.index.intersection(y.index)
    R0, y0 = R.loc[common].fillna(0.0), y.loc[common].astype(float)
    if end_date is not None:
        mask = R0.index <= end_date
        if not mask.any():
            end_date = None
        else:
            R0, y0 = R0.loc[mask], y0.loc[mask]
    if y0.isna().any():
        good = y0.notna()
        R0, y0 = R0.loc[good], y0.loc[good]
    N = len(y0)
    if N < 500:
        log.warning("Very short history for CV; falling back to smallest lambda")
        return float(lambdas[0])
    fold_edges = np.linspace(0, N, n_folds+1, dtype=int)[1:]
    scores: Dict[float, float] = {}
    for lam in lambdas:
        corr_list = []
        for i in range(len(fold_edges)-1):
            train_end = fold_edges[i]
            test_end = fold_edges[i+1]
            purge_end = min(train_end + purge_gap_days, test_end)
            if purge_end >= test_end or train_end < 252 or (test_end - purge_end) < 63:
                continue
            R_tr, y_tr = R0.iloc[:train_end], y0.iloc[:train_end]
            R_te, y_te = R0.iloc[purge_end:test_end], y0.iloc[purge_end:test_end]
            if len(R_te) < 30 or len(R_tr) < 252:
                continue
            model = Ridge(alpha=float(lam), fit_intercept=False, positive=CFG.POSITIVE)
            model.fit(R_tr.values, y_tr.values)
            beta_raw = pd.Series(model.coef_, index=R_tr.columns)
            beta = renorm_with_cap(beta_raw, CFG.MAX_SINGLE_WEIGHT, CFG.L1_TARGET)
            y_hat = R_te @ beta
            c = safe_corr(y_hat, y_te)
            if np.isfinite(c):
                corr_list.append(c)
        scores[float(lam)] = np.nanmean(corr_list) if corr_list else -np.inf
    if any(np.isfinite(v) and v>-np.inf for v in scores.values()):
        best = max(scores, key=lambda k: scores[k])
    else:
        best = float(lambdas[0])
        log.warning("CV produced no finite scores; falling back to smallest lambda.")
    log.info(f"CV best lambda={best:.6f} (CV corr={scores.get(best, np.nan):.3f})")
    return float(best)

def fit_quarterly_betas(R: pd.DataFrame, y: pd.Series, lam: float, fix_after: Optional[pd.Timestamp]) -> pd.DataFrame:
    q_start = R.index.min()
    q_end = R.index.max()
    quarters = pd.date_range(q_start, q_end, freq="Q", tz="UTC")
    beta_hist: Dict[pd.Timestamp, pd.Series] = {}
    lam_used = lam
    Rz, yz = R.fillna(0.0), y.astype(float)
    for q in quarters:
        mask = Rz.index <= q
        R_tr, y_tr = Rz.loc[mask], yz.loc[mask]
        if y_tr.isna().any():
            good = y_tr.notna()
            R_tr, y_tr = R_tr.loc[good], y_tr.loc[good]
        if len(R_tr) < 252:
            continue
        if fix_after is not None and q <= fix_after:
            if q == quarters[quarters <= fix_after][-1]:
                lam_used = cv_lambda_on_corr(Rz, yz, CFG.LAMBDA_GRID, CFG.CV_FOLDS, end_date=fix_after)
                log.info(f"Lambda fixed at {lam_used:.6f} after {fix_after.date()}")
        model = Ridge(alpha=float(lam_used), fit_intercept=False, positive=CFG.POSITIVE)
        model.fit(R_tr.values, y_tr.values)
        beta_raw = pd.Series(model.coef_, index=R_tr.columns)
        beta = renorm_with_cap(beta_raw, CFG.MAX_SINGLE_WEIGHT, CFG.L1_TARGET)
        beta_hist[q] = beta
    if not beta_hist:
        raise RuntimeError("No quarterly fits produced.")
    B = pd.DataFrame(beta_hist).T
    daily = pd.date_range(B.index.min(), B.index.max(), freq="B", tz="UTC")
    B = B.reindex(daily).ffill().reindex(R.index).ffill()
    B = B.ewm(span=CFG.WEIGHT_EMA_SPAN, adjust=False).mean()
    first = B.index[0]
    B.loc[B.index < first + pd.Timedelta(days=CFG.STARTUP_DELAY_DAYS)] = 0.0
    return B

def winsor_scale(s: pd.Series, w: float = CFG.FINAL_SCALE_WINSOR) -> pd.Series:
    if w <= 0:
        return s
    med = s.rolling(21, min_periods=5).median()
    lo = med * (1 - w)
    hi = med * (1 + w)
    return s.clip(lower=lo, upper=hi).where(med.notna(), s)

def generate_replication(R: pd.DataFrame, beta_daily: pd.DataFrame, final_target: float, cost_bps: float, overlay_cost_bps: float):
    beta_daily = beta_daily.reindex(columns=R.columns, fill_value=0.0)
    raw = (R.fillna(0.0) * beta_daily).sum(axis=1)
    sigma = raw.ewm(span=CFG.EWMA_SPAN_VOL, adjust=False, min_periods=CFG.STARTUP_DELAY_DAYS).std().shift(1) * np.sqrt(CFG.ANN_DAYS)
    sigma = sigma.replace([0, np.inf], np.nan).bfill()
    scale = (final_target / sigma).clip(lower=CFG.FINAL_SCALE_FLOOR, upper=CFG.FINAL_SCALE_CAP).fillna(method="bfill").fillna(method="ffill")
    scale = winsor_scale(scale, CFG.FINAL_SCALE_WINSOR)
    final_gross = (raw * scale).fillna(0.0)
    delta_w = beta_daily.diff().abs().sum(axis=1)
    one_way_turnover_weights = 0.5 * delta_w
    l1_prev = beta_daily.abs().sum(axis=1).shift(1).fillna(0.0)
    one_way_turnover_overlay = (scale.diff().abs().fillna(0.0) * l1_prev)
    cost_weights = (one_way_turnover_weights * (cost_bps / 1e4)).fillna(0.0)
    cost_overlay = (one_way_turnover_overlay * (overlay_cost_bps / 1e4)).fillna(0.0)
    final_net = final_gross - cost_weights - cost_overlay
    return final_net, scale, one_way_turnover_weights, one_way_turnover_overlay

def performance_table(rep: pd.Series, y: pd.Series, start="2001-01-02") -> pd.DataFrame:
    idx = rep.index.intersection(y.index)
    idx = idx[idx >= pd.Timestamp(start, tz="UTC")]
    r, b = rep.loc[idx], y.loc[idx]
    out = pd.DataFrame(index=["Replication","Benchmark"], columns=["Annualized Return","Annualized Volatility","Sharpe Ratio","Max Drawdown","Correlation","Tracking Error"])
    def ann_ret(x): return (1+ x).prod()**(252/max(len(x),1)) - 1.0 if len(x)>0 else np.nan
    def dd(x): c = (1+x).cumprod(); peak = c.cummax(); return (c/peak - 1.0).min()
    out.loc["Replication","Annualized Return"] = ann_ret(r)
    out.loc["Benchmark","Annualized Return"] = ann_ret(b)
    out.loc["Replication","Annualized Volatility"] = r.std()*np.sqrt(252)
    out.loc["Benchmark","Annualized Volatility"] = b.std()*np.sqrt(252)
    out.loc["Replication","Sharpe Ratio"] = (r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else np.nan
    out.loc["Benchmark","Sharpe Ratio"] = (b.mean()/b.std()*np.sqrt(252)) if b.std()>0 else np.nan
    c = r.corr(b)
    out.loc["Replication","Correlation"] = c
    out.loc["Benchmark","Correlation"] = c
    out.loc["Replication","Tracking Error"] = (r-b).std()*np.sqrt(252)
    out.loc["Benchmark","Tracking Error"] = (r-b).std()*np.sqrt(252)
    out.loc["Replication","Max Drawdown"] = dd(r)
    out.loc["Benchmark","Max Drawdown"] = dd(b)
    return out.astype(float)

COLOR_PALETTE = {'model': '#d62728', 'benchmark': '#9467bd', 'normal': '#2c3e50', 'crisis': '#e74c3c'}
CRISIS_PERIODS = [('2007-07-01', '2009-03-31', 'GFC'), ('2011-05-01', '2011-10-31', 'EU Debt'), ('2015-08-01', '2016-02-29', 'China/Oil'), ('2020-02-15', '2020-04-30', 'COVID-19'), ('2022-02-15', '2022-10-31', 'Ukraine/Inflation')]

def annotate_crisis_periods(ax):
    xlim = ax.get_xlim()
    for start, end, label in CRISIS_PERIODS:
        sdt, edt = pd.to_datetime(start), pd.to_datetime(end)
        sn, en = mdates.date2num(sdt), mdates.date2num(edt)
        if en < xlim[0] or sn > xlim[1]:
            continue
        ax.axvspan(sdt, edt, alpha=0.15, color=COLOR_PALETTE['crisis'], zorder=0)
        ax.text(sdt + (edt - sdt)/2, ax.get_ylim()[1]*0.94, label, ha='center', va='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

def classify_regime(rolling_sharpe):
    if rolling_sharpe < -2:   return 'Strong Bear'
    if rolling_sharpe < -1:   return 'Bear'
    if rolling_sharpe < -0.5: return 'Weak Bear'
    if rolling_sharpe < 0.5:  return 'Neutral'
    if rolling_sharpe < 1:    return 'Weak Bull'
    if rolling_sharpe < 2:    return 'Bull'
    return 'Strong Bull'

def make_rev26h_style_figures_process(rep: pd.Series, y: pd.Series, beta_daily: pd.DataFrame, weight_by_lb_daily: pd.DataFrame, turnover_portfolio: pd.Series):
    try:
        rep = rep.dropna()
        y = y.reindex(rep.index).dropna()
        rep = rep.reindex(y.index)
        if len(rep) < 300 or len(y) < 300:
            print("Not enough data for full viz pack — continuing with what we can.")
        burn_in_end = rep.index[min(CFG.BURN_IN_DAYS, len(rep)-1)]
        oos_start = CFG.OOS_START
        fig, ax = plt.subplots(figsize=(14, 8))
        rep_c = 100 * (1 + rep).cumprod()
        y_c = 100 * (1 + y).cumprod()
        ax.plot(_to_naive(rep_c.index), rep_c.values, label="Process-Based Replication (net)", color=COLOR_PALETTE['model'], lw=1.8)
        ax.plot(_to_naive(y_c.index), y_c.values, label="BTOP50 (excess)", color=COLOR_PALETTE['benchmark'], lw=1.5)
        ax.set_yscale("log"); ax.grid(True, alpha=0.3)
        ax.set_title(f"Cumulative Growth of $100 (Process-Based, target {CFG.FINAL_VOL_TARGET:.1%})")
        ax.set_ylabel("Portfolio Value ($)"); ax.set_xlabel("Date")
        ax.axvspan(_to_naive(rep.index[0]), _to_naive(burn_in_end), alpha=0.1, color="gray", label=f"{CFG.BURN_IN_MONTHS}-Month Burn-in")
        ax.axvline(_to_naive(oos_start), color="black", ls="--", lw=2, alpha=0.7, label="OOS Start")
        annotate_crisis_periods(ax); ax.legend(loc="upper left", framealpha=0.9)
        plt.tight_layout(); plt.savefig(VIZ_DIR / "A_cumulative_equity_curves_process_v6_btop50.png", bbox_inches="tight")
        if SHOW_FIGS: plt.show(); plt.close(fig)
        else: plt.close(fig)
        w = CFG.ROLL_CORR_WIN
        rep_b = rep.loc[burn_in_end:]
        y_b = y.loc[rep_b.index]
        if len(rep_b) >= w + 20:
            roll_corr = rep_b.rolling(w).corr(y_b)
            roll_rs2 = (roll_corr**2)
            roll_te = (rep_b - y_b).rolling(w).std() * np.sqrt(252)
            roll_ir = ((rep_b - y_b).rolling(w).mean() / ((rep_b - y_b).rolling(w).std() + 1e-12)) * np.sqrt(252)
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize=(14, 13), sharex=True)
            ax1.plot(_to_naive(roll_corr.index), roll_corr.values, color=COLOR_PALETTE['normal'], lw=1.5)
            ax1.axhline(float(roll_corr.mean()), color="black", ls="--", alpha=0.5, label=f"Mean {roll_corr.mean():.3f}")
            ax1.set_ylabel("Corr"); ax1.legend(); ax1.grid(True, alpha=0.3)
            ax1.set_title("Rolling 1Y Statistics (Ex-Burn-in)")
            ax2.plot(_to_naive(roll_rs2.index), roll_rs2.values, color=COLOR_PALETTE['model'], lw=1.5)
            ax2.fill_between(_to_naive(roll_rs2.index), 0, roll_rs2.values, color=COLOR_PALETTE['model'], alpha=0.2)
            ax2.set_ylabel("R²"); ax2.grid(True, alpha=0.3); ax2.axhline(float(roll_rs2.mean()), color="black", ls="--", alpha=0.5)
            ax3.plot(_to_naive(roll_te.index), 100*roll_te.values, color=COLOR_PALETTE['crisis'], lw=1.5)
            ax3.set_ylabel("TE (%)"); ax3.grid(True, alpha=0.3)
            ax3.axhline(float((100*roll_te).mean()), color="black", ls="--", alpha=0.5, label=f"Mean {float((100*roll_te).mean()):.1f}%")
            ax3.legend()
            ax4.plot(_to_naive(roll_ir.index), roll_ir.values, color=COLOR_PALETTE['normal'], lw=1.5)
            ax4.axhline(0, color="black", lw=1, alpha=0.5)
            ax4.axhline(float(roll_ir.mean()), color="black", ls="--", alpha=0.5, label=f"Mean IR {float(roll_ir.mean()):.3f}")
            ax4.set_ylabel("IR"); ax4.set_xlabel("Date"); ax4.legend(); ax4.grid(True, alpha=0.3)
            for ax in (ax1, ax2, ax3, ax4):
                annotate_crisis_periods(ax)
                ax.axvline(_to_naive(oos_start), color='gray', ls='--', lw=2, alpha=0.7)
            plt.tight_layout(); plt.savefig(VIZ_DIR / "B_rolling_statistics_process_v6_btop50.png", bbox_inches="tight")
            if SHOW_FIGS: plt.show(); plt.close(fig)
            else: plt.close(fig)
        if isinstance(beta_daily.columns, pd.MultiIndex) and 'lookback' in beta_daily.columns.names:
            w_by_lb_abs = beta_daily.abs().groupby(axis=1, level='lookback').sum()
        else:
            w_by_lb_abs = weight_by_lb_daily.abs()
        denom = w_by_lb_abs.sum(axis=1).replace(0, np.nan)
        lb_share = w_by_lb_abs.div(denom, axis=0).fillna(0.0)
        fig, ax = plt.subplots(figsize=(14, 6))
        lbs = list(lb_share.columns)
        ax.stackplot(_to_naive(lb_share.index), [lb_share[c].values for c in lbs], labels=[f"{c}d" for c in lbs], alpha=0.9)
        ax.set_ylim(0,1); ax.set_ylabel('Share of |weights|'); ax.set_xlabel('Date'); ax.grid(True, alpha=0.3)
        ax.set_title('Lookback Share of Absolute Weights (normalized daily)')
        ax.legend(loc='upper left', ncol=5, bbox_to_anchor=(1.0, 1.0))
        ax.axvline(_to_naive(CFG.OOS_START), color='black', ls='--', lw=2, alpha=0.7)
        plt.tight_layout(); plt.savefig(VIZ_DIR / "E_lookback_weight_share_process_v6_btop50.png", bbox_inches="tight")
        if SHOW_FIGS: plt.show(); plt.close(fig)
        else: plt.close(fig)
        print(f"Base figures A/B/E saved to: {VIZ_DIR}")
    except Exception as e:
        print(f"REV-26h visuals failed: {e}")

def make_rev26h_extras(rep: pd.Series, y: pd.Series, scale: pd.Series, beta_daily: pd.DataFrame, R_eff: pd.DataFrame):
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    start = max(rep.first_valid_index(), y.first_valid_index())
    r = rep.loc[start:]; b = y.loc[start:]
    try:
        fig, ax = plt.subplots(figsize=(14,4))
        ax.plot(_to_naive(scale.index), scale.values, lw=1.2)
        ax.axhline(CFG.FINAL_SCALE_CAP, ls="--", alpha=0.5)
        ax.axhline(CFG.FINAL_SCALE_FLOOR, ls="--", alpha=0.5)
        ax.set_title("Overlay Scale (×)"); ax.set_ylabel("×"); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(VIZ_DIR / "D_overlay_scale_process_v6_btop50.png", bbox_inches="tight")
        if SHOW_FIGS: plt.show(); plt.close(fig)
        else: plt.close(fig)
    except Exception as e:
        print(f"overlay scale plot failed: {e}")
    try:
        res = (r - b).dropna()
        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(res.values*100, bins=50, alpha=0.85)
        ax.set_title("Residuals Histogram (rep - bench, daily %)"); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(VIZ_DIR / "F_residuals_hist_process_v6_btop50.png", bbox_inches="tight")
        if SHOW_FIGS: plt.show(); plt.close(fig)
        else: plt.close(fig)
        fig, ax = plt.subplots(figsize=(14,4))
        ax.plot(_to_naive(res.index), (res.cumsum()*100).values, lw=1.4)
        ax.set_title("Cumulative Residual P&L (rep - bench, % points)"); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(VIZ_DIR / "F_residuals_cum_process_v6_btop50.png", bbox_inches="tight")
        if SHOW_FIGS: plt.show(); plt.close(fig)
        else: plt.close(fig)
    except Exception as e:
        print(f"residual plots failed: {e}")

def _monthly_return(s: pd.Series) -> pd.Series:
    return s.resample('M').apply(lambda x: (1.0 + x).prod() - 1.0)

def _calendar_year_return(s: pd.Series) -> pd.Series:
    return s.resample('Y').apply(lambda x: x.add(1.0).prod() - 1.0)

def export_turnover_by_lookback(W: pd.DataFrame):
    if not (isinstance(W.columns, pd.MultiIndex) and 'lookback' in W.columns.names):
        print("export_turnover_by_lookback: columns do not have a 'lookback' level; skipping.")
        return
    by_lb_daily = 0.5 * W.diff().abs().groupby(level='lookback', axis=1).sum()
    by_lb_daily.to_csv(VIZ_DIR / "turnover_by_lookback_daily_process_v6_btop50.csv", float_format="%.8f")
    mean_d = by_lb_daily.mean(axis=0); std_d = by_lb_daily.std(axis=0)
    summary = pd.DataFrame({"lookback": mean_d.index.astype(int), "mean_daily": mean_d.values, "std_daily": std_d.values, "annual": (mean_d * 252).values}).sort_values("lookback")
    summary.to_csv(VIZ_DIR / "turnover_by_lookback_process_v6_btop50.csv", index=False, float_format="%.8f")
    pretty = summary.copy()
    pretty["mean_daily"] = 100 * pretty["mean_daily"]; pretty["annual"] = 100 * pretty["annual"]
    print("\nTurnover by lookback — annualized % of notional")
    print(pretty.to_string(index=False, formatters={"mean_daily": "{:.4f}%".format, "std_daily": "{:.6f}".format, "annual": "{:.2f}%".format}))
    fig, ax = plt.subplots(figsize=(14, 8))
    xs = np.arange(len(summary))
    ax.bar(xs, 100.0 * summary["annual"].values)
    ax.set_xticks(xs)
    ax.set_xticklabels(summary["lookback"].astype(int).astype(str))
    ax.yaxis.set_major_formatter(PercentFormatter(100.0))
    ax.set_title("Turnover by Lookback (annualized, % of notional)")
    ax.set_xlabel("Lookback (days)")
    ax.set_ylabel("% / year")
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "C_turnover_by_lookback_process_v6_btop50.png", bbox_inches="tight")
    if SHOW_FIGS: plt.show(); plt.close(fig)
    else: plt.close(fig)

def export_implementation_quality(rep: pd.Series, y: pd.Series, scale: pd.Series, turn_w: pd.Series, turn_s: pd.Series, W_eff_adj: Optional[pd.DataFrame]):
    idx = rep.index.intersection(y.index)
    r, b = rep.loc[idx], y.loc[idx]
    res = r - b
    te = res.std() * np.sqrt(252)
    corr = r.corr(b)
    r2 = corr**2 if pd.notna(corr) else np.nan
    ir = (res.mean() / (res.std() + 1e-12)) * np.sqrt(252)
    ann_r = (1.0 + r).prod()**(252/len(r)) - 1.0 if len(r) else np.nan
    ann_b = (1.0 + b).prod()**(252/len(b)) - 1.0 if len(b) else np.nan
    cap_frac = (scale >= CFG.FINAL_SCALE_CAP - 1e-12).mean()
    floor_frac = (scale <= CFG.FINAL_SCALE_FLOOR + 1e-12).mean()
    live_l1 = W_eff_adj.abs().sum(axis=1).mean() if W_eff_adj is not None else np.nan
    cost_w_bps = (turn_w * (CFG.REPL_TRADE_COST_BPS / 1e4)).mean() * 1e4
    cost_s_bps = (turn_s * (CFG.OVERLAY_TRADE_COST_BPS / 1e4)).mean() * 1e4
    out = pd.DataFrame([{"start": str(idx.min().date()) if len(idx) else None, "end": str(idx.max().date()) if len(idx) else None, "days": int(len(idx)), "corr": corr, "R2": r2, "TE": te, "IR": ir, "rep_ann_return": ann_r, "bench_ann_return": ann_b, "mean_overlay": float(scale.mean()), "pct_days_at_cap": float(cap_frac), "pct_days_at_floor": float(floor_frac), "mean_live_L1": float(live_l1) if pd.notna(live_l1) else np.nan, "avg_replication_cost_bps_per_day": float(cost_w_bps), "avg_overlay_cost_bps_per_day": float(cost_s_bps)}])
    out.to_csv(CFG.OUT_DIR / "implementation_quality_process_v6_btop50.csv", index=False, float_format="%.8f")
    print("\nImplementation quality\n", out.to_string(index=False))

def export_oos_performance(rep: pd.Series, y: pd.Series):
    def stats(r, b):
        res = r - b
        d = {"corr": r.corr(b), "TE": res.std() * np.sqrt(252), "SR": (r.mean()/(r.std() + 1e-12)) * np.sqrt(252), "vol": r.std() * np.sqrt(252), "CAGR": (1.0 + r).prod()**(252/max(len(r),1)) - 1.0 if len(r) else np.nan}
        return d
    oos = CFG.OOS_START
    idx = rep.index.intersection(y.index)
    r_all, b_all = rep.loc[idx], y.loc[idx]
    r_is, b_is = r_all.loc[:oos - pd.Timedelta(days=1)], b_all.loc[:oos - pd.Timedelta(days=1)]
    r_oos, b_oos = r_all.loc[oos:], b_all.loc[oos:]
    rows = []
    rows.append({"segment": "ALL", **stats(r_all, b_all)})
    rows.append({"segment": "IS", **stats(r_is, b_is)})
    rows.append({"segment": "OOS", **stats(r_oos, b_oos)})
    df = pd.DataFrame(rows)
    df.to_csv(CFG.OUT_DIR / "oos_performance_process_v6_btop50.csv", index=False, float_format="%.8f")
    print("\nIS / OOS performance\n", df.to_string(index=False))

def make_calendar_year_bars(rep: pd.Series, y: pd.Series):
    r_y = _calendar_year_return(rep)
    b_y = _calendar_year_return(y.reindex(r_y.index))
    a_y = (1.0 + (rep - y.reindex(rep.index))).resample('Y').apply(lambda x: x.prod() - 1.0)
    tbl = pd.DataFrame({"model": r_y, "benchmark": b_y, "alpha": a_y})
    tbl.index = tbl.index.year
    tbl.to_csv(VIZ_DIR / "calendar_year_table_process_v6_btop50.csv", float_format="%.8f")
    fig, ax = plt.subplots(figsize=(14, 8))
    xs = np.arange(len(tbl)); width = 0.28
    ax.bar(xs - width, 100*tbl["model"].values, width, label="Model")
    ax.bar(xs, 100*tbl["benchmark"].values, width, label="Benchmark")
    ax.bar(xs + width, 100*tbl["alpha"].values, width, label="Alpha")
    ax.set_xticks(xs); ax.set_xticklabels(tbl.index.astype(int).astype(str))
    ax.yaxis.set_major_formatter(PercentFormatter(100.0))
    ax.set_title("Calendar-Year Returns")
    ax.grid(True, axis='y', alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(VIZ_DIR / "Y_calendar_year_bars_process_v6_btop50.png", bbox_inches="tight")
    if SHOW_FIGS: plt.show(); plt.close(fig)
    else: plt.close(fig)

def make_monthly_heatmap(rep: pd.Series, y: pd.Series, annotate: bool = True, fmt: str = "{:.1f}", diverging: bool = True):
    m_model = _monthly_return(rep)
    m_bench = _monthly_return(y.reindex(m_model.index))
    m_alpha = _monthly_return(rep - y.reindex(rep.index))
    df = pd.DataFrame({"model": m_model, "benchmark": m_bench, "alpha": m_alpha})
    df.to_csv(VIZ_DIR / "monthly_returns_process_v6_btop50.csv", float_format="%.8f")
    mat = df["model"].copy()
    mat.index = pd.to_datetime(mat.index)
    heat = mat.to_frame("ret")
    heat["year"] = heat.index.year
    heat["month"] = heat.index.month
    pivot = heat.pivot(index="year", columns="month", values="ret").sort_index()
    data_pct = 100.0 * pivot.values
    fig, ax = plt.subplots(figsize=(14, 8))
    if diverging:
        vmin = np.nanmin(data_pct); vmax = np.nanmax(data_pct); bound = max(abs(vmin), abs(vmax))
        norm = mcolors.TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
        im = ax.imshow(data_pct, aspect="auto", interpolation="nearest", cmap=plt.cm.RdYlGn, norm=norm)
    else:
        im = ax.imshow(data_pct, aspect="auto", interpolation="nearest")
    ax.set_title("Monthly Returns — Model (%)")
    ax.set_yticks(np.arange(len(pivot.index))); ax.set_yticklabels(pivot.index.astype(int).astype(str))
    ax.set_xticks(np.arange(12)); ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    cbar = plt.colorbar(im, ax=ax); cbar.ax.set_ylabel("%", rotation=270, labelpad=15)
    ax.grid(False)
    if annotate:
        def _best_text_color(val: float) -> str:
            rgba = im.cmap(im.norm(val)) if hasattr(im, "norm") else im.cmap(val)
            r, g, b = rgba[:3]; luminance = 0.299*r + 0.587*g + 0.114*b
            return "black" if luminance > 0.6 else "white"
        n_rows, n_cols = data_pct.shape
        for i in range(n_rows):
            for j in range(n_cols):
                v = data_pct[i, j]
                if np.isnan(v): continue
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8, color=_best_text_color(v))
    plt.tight_layout(); plt.savefig(VIZ_DIR / "I_monthly_return_heatmap_process_v6_btop50.png", bbox_inches="tight")
    if SHOW_FIGS: plt.show(); plt.close(fig)
    else: plt.close(fig)

def make_monthly_scatter_regime(rep: pd.Series, y: pd.Series):
    m_model = _monthly_return(rep)
    m_bench = _monthly_return(y.reindex(m_model.index))
    rs_daily = (y.rolling(63).mean() / (y.rolling(63).std() + 1e-12)) * np.sqrt(252)
    rs_m_ends = rs_daily.reindex(m_bench.index, method="ffill")
    reg = rs_m_ends.apply(classify_regime)
    tmp = pd.DataFrame({"x": m_bench.values, "y": m_model.values, "regime": reg.values}, index=m_model.index)
    tmp.to_csv(VIZ_DIR / "monthly_scatter_data_process_v6_btop50.csv", float_format="%.8f")
    colors = {"Strong Bear":"#b2182b", "Bear":"#ef8a62", "Weak Bear":"#fddbc7", "Neutral":"#cccccc", "Weak Bull":"#d1e5f0", "Bull":"#67a9cf", "Strong Bull":"#2166ac"}
    fig, ax = plt.subplots(figsize=(14, 8))
    for g, sub in tmp.groupby("regime"):
        ax.scatter(100*sub["x"], 100*sub["y"], s=18, alpha=0.85, label=g, edgecolors="none", c=colors.get(g, "#333333"))
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1], 5)
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=1, alpha=0.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Benchmark monthly (%)"); ax.set_ylabel("Model monthly (%)")
    ax.set_title("Monthly Return Scatter (regime‑colored)")
    ax.grid(True, alpha=0.3); ax.legend(ncol=3, fontsize=9)
    plt.tight_layout(); plt.savefig(VIZ_DIR / "C_monthly_scatter_regime_process_v6_btop50.png", bbox_inches="tight")
    if SHOW_FIGS: plt.show(); plt.close(fig)
    else: plt.close(fig)

def make_drawdown_analysis(rep: pd.Series, y: pd.Series):
    def dd(s):
        c = (1.0 + s).cumprod(); peak = c.cummax()
        return c/peak - 1.0
    r = rep.dropna(); b = y.reindex(r.index).dropna()
    r = r.reindex(b.index)
    dd_r, dd_b = dd(r), dd(b)
    diff = dd_r - dd_b
    dd_tbl = pd.DataFrame({"stat": ["min_model_dd","min_bench_dd","min_diff_dd"], "value": [dd_r.min(), dd_b.min(), diff.min()]})
    dd_tbl.to_csv(VIZ_DIR / "drawdown_stats_process_v6_btop50.csv", index=False, float_format="%.8f")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.plot(_to_naive(dd_b.index), 100*dd_b, label="Benchmark"); ax1.plot(_to_naive(dd_r.index), 100*dd_r, label="Model")
    ax1.set_title("Drawdowns (%)"); ax1.grid(True, alpha=0.3); ax1.legend()
    ax2.plot(_to_naive(diff.index), 100*diff, label="Model − Benchmark")
    ax2.set_title("Drawdown Difference (pp)"); ax2.grid(True, alpha=0.3)
    for ax in (ax1, ax2):
        ax.yaxis.set_major_formatter(PercentFormatter(100.0))
        ax.axvline(_to_naive(CFG.OOS_START), color="black", ls="--", lw=2, alpha=0.7)
        annotate_crisis_periods(ax)
    plt.tight_layout(); plt.savefig(VIZ_DIR / "H_drawdown_analysis_process_v6_btop50.png", bbox_inches="tight")
    if SHOW_FIGS: plt.show(); plt.close(fig)
    else: plt.close(fig)

def export_timeavg_lookback_weights_csv(W_eff_adj: pd.DataFrame):
    w_by_lb = W_eff_adj.abs().groupby(axis=1, level="lookback").sum()
    share = (w_by_lb.T / w_by_lb.sum(axis=1).replace(0, np.nan)).T
    timeavg = share.mean().sort_index()
    out = pd.DataFrame({"lookback": timeavg.index.astype(int), "share": timeavg.values})
    out.to_csv(VIZ_DIR / "pb_timeavg_lookback_weights_process_v6_btop50.csv", index=False, float_format="%.8f")

def export_library_coverage_heatmap(R_eff: pd.DataFrame):
    live = R_eff.notna().mean(axis=0)
    meta = pd.DataFrame({"family": R_eff.columns.get_level_values("family"), "contract": R_eff.columns.get_level_values("contract"), "lookback": R_eff.columns.get_level_values("lookback").astype(int), "live_frac": live.values})
    agg = (meta.groupby(["family", "lookback"]).agg(n_contracts=("contract", "nunique"), avg_live=("live_frac", "mean")).reset_index())
    agg["effective_series"] = agg["n_contracts"] * agg["avg_live"]
    agg.to_csv(VIZ_DIR / "pb_library_summary_process_v6_btop50.csv", index=False, float_format="%.6f")
    pivot = agg.pivot(index="family", columns="lookback", values="effective_series")
    pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(pivot.fillna(0.0).values, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns))); ax.set_xticklabels(pivot.columns.astype(int).astype(str))
    ax.set_title("Strategy‑Library Coverage / Density (effective series)")
    cbar = plt.colorbar(im, ax=ax); cbar.ax.set_ylabel("effective series", rotation=270, labelpad=15)
    plt.tight_layout(); plt.savefig(VIZ_DIR / "PB_library_coverage_heatmap_process_v6_btop50.png", bbox_inches="tight")
    if SHOW_FIGS: plt.show(); plt.close(fig)
    else: plt.close(fig)

def export_cv_lambda_path(R: pd.DataFrame, y: pd.Series):
    common = R.index.intersection(y.index)
    R0, y0 = R.loc[common].fillna(0.0), y.loc[common].astype(float)
    if y0.isna().any():
        good = y0.notna()
        R0, y0 = R0.loc[good], y0.loc[good]
    N = len(y0)
    if N < 500:
        return
    fold_edges = np.linspace(0, N, CFG.CV_FOLDS + 1, dtype=int)[1:]
    scores = {}
    for lam in CFG.LAMBDA_GRID:
        lam = float(lam)
        corr_list = []
        for i in range(len(fold_edges)-1):
            train_end = fold_edges[i]
            test_end = fold_edges[i+1]
            purge_end = min(train_end + CFG.PURGE_GAP_DAYS, test_end)
            if purge_end >= test_end or train_end < 252 or (test_end - purge_end) < 63:
                continue
            R_tr, y_tr = R0.iloc[:train_end], y0.iloc[:train_end]
            R_te, y_te = R0.iloc[purge_end:test_end], y0.iloc[purge_end:test_end]
            model = Ridge(alpha=lam, fit_intercept=False, positive=CFG.POSITIVE)
            model.fit(R_tr.values, y_tr.values)
            beta_raw = pd.Series(model.coef_, index=R_tr.columns)
            beta = renorm_with_cap(beta_raw, CFG.MAX_SINGLE_WEIGHT, CFG.L1_TARGET)
            y_hat = R_te @ beta
            c = safe_corr(y_hat, y_te)
            if np.isfinite(c):
                corr_list.append(c)
        scores[lam] = np.nanmean(corr_list) if corr_list else np.nan
    df = pd.DataFrame({"lambda": list(scores.keys()), "cv_corr": list(scores.values())}).dropna().sort_values("lambda")
    df.to_csv(VIZ_DIR / "pb_cv_lambda_summary_process_v6_btop50.csv", index=False, float_format="%.6f")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["lambda"], df["cv_corr"], lw=1.8)
    ax.set_xscale("log"); ax.set_xlabel("λ (Ridge)"); ax.set_ylabel("Mean OOS corr (CV)")
    ax.set_title("CV λ‑Path (blocked, purged)")
    ax.grid(True, alpha=0.3)
    _save_only(fig, VIZ_DIR / "PB_cv_lambda_path_process_v6_btop50.png")

def export_cost_decomposition(turnover_w: pd.Series, turnover_s: pd.Series):
    cw = (turnover_w * (CFG.REPL_TRADE_COST_BPS / 1e4)).resample("M").sum() * 1e4
    cs = (turnover_s * (CFG.OVERLAY_TRADE_COST_BPS / 1e4)).resample("M").sum() * 1e4
    df = pd.DataFrame({"replication_bps_per_month": cw, "overlay_bps_per_month": cs})
    df.to_csv(VIZ_DIR / "pb_cost_decomposition_process_v6_btop50.csv", float_format="%.6f")
    fig, ax = plt.subplots(figsize=(14,8))
    ax.bar(_to_naive(df.index), df["replication_bps_per_month"], label="Replication trading (bps/mo)")
    ax.bar(_to_naive(df.index), df["overlay_bps_per_month"], bottom=df["replication_bps_per_month"], alpha=0.35, label="Overlay rebalancing (bps/mo)")
    ax.set_title("Cost Decomposition (monthly bps, one-way)")
    ax.grid(True, axis='y', alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(VIZ_DIR / "PB_cost_decomposition_process_v6_btop50.png", bbox_inches="tight")
    if SHOW_FIGS: plt.show(); plt.close(fig)
    else: plt.close(fig)

def plot_asset_class_rolling_contrib(rolling_ac: pd.DataFrame, outfile: Path):
    fig, ax = plt.subplots(figsize=(14, 8))
    for col in rolling_ac.columns:
        ax.plot(_to_naive(rolling_ac.index), 100*rolling_ac[col], label=col, lw=1.4)
    ax.set_title("Asset-Class Rolling 3m Contribution (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(100.0))
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=9, bbox_to_anchor=(1.0, 1.02), loc="upper right")
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    if SHOW_FIGS: plt.show(); plt.close(fig)
    else: plt.close(fig)

def export_signal_bars_and_csv(R_eff: pd.DataFrame, W_eff_adj: pd.DataFrame, scale: pd.Series, y_eff: pd.Series):
    contrib_daily = R_eff.mul(W_eff_adj, fill_value=0.0)
    scale = scale.reindex(contrib_daily.index).astype(float)
    contrib_daily = (contrib_daily.T * scale).T
    fam_daily = contrib_daily.groupby(axis=1, level='family').sum()
    fam_m = fam_daily.resample("M").sum()
    fam_m.to_csv(VIZ_DIR / "pb_signal_family_contrib_monthly_process_v6_btop50.csv", float_format="%.8f")
    if len(fam_m) > 0:
        start_bar = fam_m.index.max() - pd.DateOffset(years=10)
        fmb = fam_m.loc[start_bar:]
        fig, ax = plt.subplots(figsize=(14, 8))
        bottom = np.zeros(len(fmb))
        for col in fmb.columns:
            ax.bar(_to_naive(fmb.index), 100*fmb[col].values, bottom=bottom, label=col)
            bottom += 100*fmb[col].values
        ax.set_title("Family Contribution — Monthly (last 10y, %)")
        ax.yaxis.set_major_formatter(PercentFormatter(100.0))
        ax.grid(True, axis='y', alpha=0.3); ax.legend(ncol=5, fontsize=8, bbox_to_anchor=(1.0, 1.02), loc="upper right")
        ax.axvline(_to_naive(CFG.OOS_START), color="black", ls="--", lw=2, alpha=0.7)
        _save_only(fig, VIZ_DIR / "PB_signal_family_contrib_monthly_process_v6_btop50.png")
        fam_y = fam_daily.resample("Y").sum()
        fam_y.index = fam_y.index.year
        fam_y.to_csv(VIZ_DIR / "pb_signal_family_contrib_calendar_process_v6_btop50.csv", float_format="%.8f")
        fig, ax = plt.subplots(figsize=(14, 8))
        xs = np.arange(len(fam_y))
        bottoms = np.zeros(len(fam_y))
        for col in fam_y.columns:
            ax.bar(xs, 100*fam_y[col].values, bottom=bottoms, label=col)
            bottoms += 100*fam_y[col].values
        ax.set_xticks(xs); ax.set_xticklabels(fam_y.index.astype(int).astype(str))
        ax.yaxis.set_major_formatter(PercentFormatter(100.0))
        ax.set_title("Family Contribution — Calendar‑Year (%)")
        ax.grid(True, axis='y', alpha=0.3); ax.legend(ncol=5, fontsize=8, bbox_to_anchor=(1.0, 1.02), loc="upper right")
        plt.tight_layout(); plt.savefig(VIZ_DIR / "PB_signal_family_contrib_calendar_bars_process_v6_btop50.png", bbox_inches="tight")
        if SHOW_FIGS: plt.show(); plt.close(fig)
        else: plt.close(fig)
    lb_daily = contrib_daily.groupby(axis=1, level='lookback').sum()
    lb_m = lb_daily.resample("M").sum()
    lb_m.to_csv(VIZ_DIR / "pb_signal_lookback_contrib_monthly_process_v6_btop50.csv", float_format="%.8f")
    if len(lb_m) > 0:
        start_bar = lb_m.index.max() - pd.DateOffset(years=10)
        lbm = lb_m.loc[start_bar:]
        fig, ax = plt.subplots(figsize=(14, 8))
        bottom = np.zeros(len(lbm))
        for col in sorted(lbm.columns):
            ax.bar(_to_naive(lbm.index), 100*lbm[col].values, bottom=bottom, label=f"{int(col)}d")
            bottom += 100*lbm[col].values
        ax.set_title("Lookback Contribution — Monthly (last 10y, %)")
        ax.yaxis.set_major_formatter(PercentFormatter(100.0))
        ax.grid(True, axis='y', alpha=0.3); ax.legend(ncol=6, fontsize=8, bbox_to_anchor=(1.0, 1.02), loc="upper right")
        ax.axvline(_to_naive(CFG.OOS_START), color="black", ls="--", lw=2, alpha=0.7)
        _save_only(fig, VIZ_DIR / "PB_signal_lookback_contrib_monthly_process_v6_btop50.png")
        lb_y = lb_daily.resample("Y").sum()
        lb_y.index = lb_y.index.year
        fig, ax = plt.subplots(figsize=(14, 8))
        xs = np.arange(len(lb_y))
        bottoms = np.zeros(len(lb_y))
        for col in sorted(lb_y.columns):
            ax.bar(xs, 100*lb_y[col].values, bottom=bottoms, label=f"{int(col)}d")
            bottoms += 100*lb_y[col].values
        ax.set_xticks(xs); ax.set_xticklabels(lb_y.index.astype(int).astype(str))
        ax.yaxis.set_major_formatter(PercentFormatter(100.0))
        ax.set_title("Lookback Contribution — Calendar‑Year (%)")
        ax.grid(True, axis='y', alpha=0.3); ax.legend(ncol=6, fontsize=8, bbox_to_anchor=(1.0, 1.02), loc="upper right")
        plt.tight_layout(); plt.savefig(VIZ_DIR / "PB_signal_lookback_contrib_calendar_bars_process_v6_btop50.png", bbox_inches="tight")
        if SHOW_FIGS: plt.show(); plt.close(fig)
        else: plt.close(fig)
    ctrt_daily = contrib_daily.groupby(axis=1, level='contract').sum()
    ctrt_daily.to_csv(VIZ_DIR / "pb_signal_contract_contrib_daily_process_v6_btop50.csv", float_format="%.8f")
    avg_abs = ctrt_daily.abs().mean().sort_values(ascending=False)
    top = list(avg_abs.head(10).index)
    if top:
        cm = ctrt_daily[top].resample("M").sum()
        mean_m = 100*cm.mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(14, 8))
        xs = np.arange(len(mean_m))
        ax.bar(xs, mean_m.values)
        ax.set_xticks(xs); ax.set_xticklabels(mean_m.index, rotation=45, ha='right')
        ax.set_ylabel("% per month")
        ax.set_title("Top‑10 Contracts — Mean Monthly Contribution (post‑overlay, %)")
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout(); plt.savefig(VIZ_DIR / "PB_signal_contract_top10_contrib_bars_process_v6_btop50.png", bbox_inches="tight")
        if SHOW_FIGS: plt.show(); plt.close(fig)
        else: plt.close(fig)
        rc = ctrt_daily[top].rolling(63).sum()
        fig, ax = plt.subplots(figsize=(14, 8))
        for c in top:
            ax.plot(_to_naive(rc.index), 100*rc[c], label=c, lw=1.2)
        ax.set_title("Top‑10 Contracts — Rolling 63‑day Contribution (%)")
        ax.yaxis.set_major_formatter(PercentFormatter(100.0))
        ax.grid(True, alpha=0.3); ax.legend(ncol=5)
        plt.tight_layout(); plt.savefig(VIZ_DIR / "PB_signal_contract_top10_contrib_process_v6_btop50.png", bbox_inches="tight")
        if SHOW_FIGS: plt.show(); plt.close(fig)
        else: plt.close(fig)
    bench = y_eff.reindex(fam_daily.index)
    fam_te = {}
    base_res = (fam_daily.sum(axis=1) - bench)
    base_te = base_res.rolling(252).std() * np.sqrt(252)
    for fam in fam_daily.columns:
        without = (fam_daily.drop(columns=[fam]).sum(axis=1) - bench)
        te_wo = without.rolling(252).std() * np.sqrt(252)
        fam_te[fam] = (base_te - te_wo).mean()
    fam_te = pd.Series(fam_te).sort_values(ascending=False)
    fam_te.to_frame("te_contribution").to_csv(VIZ_DIR / "pb_signal_family_TE_contrib_process_v6_btop50.csv", float_format="%.8f")
    fig, ax = plt.subplots(figsize=(12, 6))
    xs = np.arange(len(fam_te))
    ax.bar(xs, 100*fam_te.values)
    ax.set_xticks(xs); ax.set_xticklabels(fam_te.index)
    ax.set_ylabel("Avg reduction in TE (pp)")
    ax.set_title("Family Contribution to Tracking Error (higher = more TE explained)")
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(VIZ_DIR / "PB_signal_family_TE_contrib_process_v6_btop50.png", bbox_inches="tight")
    if SHOW_FIGS: plt.show(); plt.close(fig)
    else: plt.close(fig)
    ASSET_CLASS_MAP = {
        "AUD":"FX","CAD":"FX","CHF":"FX","EUR":"FX","GBP":"FX","JPY":"FX",
        "US2Y":"Bond","US5Y":"Bond","US10Y":"Bond","US":"Bond","GILT":"Bond","BUND":"Bond","BOBL":"Bond",
        "S&P500":"Equity","NASDAQ":"Equity","DOW":"Equity","FTSE":"Equity","EUROSTOXX":"Equity","ASX200":"Equity","TOPIX":"Equity",
        "GOLD":"Metals","SILVER":"Metals","COPPER":"Metals",
        "WTI_CRUDE":"Energy","HEATOIL":"Energy","NATGAS":"Energy","RBOB":"Energy","BRENT_CRUDE":"Energy","CRUDE_W":"Energy","CRUDE_ICE":"Energy","GASOIL":"Energy"
    }
    contrib_by_ctrt = contrib_daily.groupby(axis=1, level="contract").sum()
    asset_classes = sorted(set(ASSET_CLASS_MAP.get(c, "Other") for c in contrib_by_ctrt.columns))
    contrib_daily_ac_df = pd.DataFrame(0.0, index=contrib_by_ctrt.index, columns=asset_classes)
    for c in contrib_by_ctrt.columns:
        ac = ASSET_CLASS_MAP.get(c, "Other")
        contrib_daily_ac_df[ac] = contrib_daily_ac_df[ac] + contrib_by_ctrt[c]
    rolling_ac = contrib_daily_ac_df.rolling(63).sum().dropna()
    if len(rolling_ac) > 0:
        plot_asset_class_rolling_contrib(rolling_ac, VIZ_DIR / "PB_asset_class_rolling_contrib_process_v6_btop50.png")

def export_timeavg_lookback_weights_csv(W_eff_adj: pd.DataFrame):
    w_by_lb = W_eff_adj.abs().groupby(axis=1, level="lookback").sum()
    share = (w_by_lb.T / w_by_lb.sum(axis=1).replace(0, np.nan)).T
    timeavg = share.mean().sort_index()
    out = pd.DataFrame({"lookback": timeavg.index.astype(int), "share": timeavg.values})
    out.to_csv(VIZ_DIR / "pb_timeavg_lookback_weights_process_v6_btop50.csv", index=False, float_format="%.8f")

def export_library_coverage_heatmap(R_eff: pd.DataFrame):
    live = R_eff.notna().mean(axis=0)
    meta = pd.DataFrame({"family": R_eff.columns.get_level_values("family"), "contract": R_eff.columns.get_level_values("contract"), "lookback": R_eff.columns.get_level_values("lookback").astype(int), "live_frac": live.values})
    agg = (meta.groupby(["family", "lookback"]).agg(n_contracts=("contract", "nunique"), avg_live=("live_frac", "mean")).reset_index())
    agg["effective_series"] = agg["n_contracts"] * agg["avg_live"]
    agg.to_csv(VIZ_DIR / "pb_library_summary_process_v6_btop50.csv", index=False, float_format="%.6f")
    pivot = agg.pivot(index="family", columns="lookback", values="effective_series")
    pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(pivot.fillna(0.0).values, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns))); ax.set_xticklabels(pivot.columns.astype(int).astype(str))
    ax.set_title("Strategy‑Library Coverage / Density (effective series)")
    cbar = plt.colorbar(im, ax=ax); cbar.ax.set_ylabel("effective series", rotation=270, labelpad=15)
    plt.tight_layout(); plt.savefig(VIZ_DIR / "PB_library_coverage_heatmap_process_v6_btop50.png", bbox_inches="tight")
    if SHOW_FIGS: plt.show(); plt.close(fig)
    else: plt.close(fig)

def export_cv_lambda_path(R: pd.DataFrame, y: pd.Series):
    common = R.index.intersection(y.index)
    R0, y0 = R.loc[common].fillna(0.0), y.loc[common].astype(float)
    if y0.isna().any():
        good = y0.notna()
        R0, y0 = R0.loc[good], y0.loc[good]
    N = len(y0)
    if N < 500:
        return
    fold_edges = np.linspace(0, N, CFG.CV_FOLDS + 1, dtype=int)[1:]
    scores = {}
    for lam in CFG.LAMBDA_GRID:
        lam = float(lam)
        corr_list = []
        for i in range(len(fold_edges)-1):
            train_end = fold_edges[i]
            test_end = fold_edges[i+1]
            purge_end = min(train_end + CFG.PURGE_GAP_DAYS, test_end)
            if purge_end >= test_end or train_end < 252 or (test_end - purge_end) < 63:
                continue
            R_tr, y_tr = R0.iloc[:train_end], y0.iloc[:train_end]
            R_te, y_te = R0.iloc[purge_end:test_end], y0.iloc[purge_end:test_end]
            model = Ridge(alpha=lam, fit_intercept=False, positive=CFG.POSITIVE)
            model.fit(R_tr.values, y_tr.values)
            beta_raw = pd.Series(model.coef_, index=R_tr.columns)
            beta = renorm_with_cap(beta_raw, CFG.MAX_SINGLE_WEIGHT, CFG.L1_TARGET)
            y_hat = R_te @ beta
            c = safe_corr(y_hat, y_te)
            if np.isfinite(c):
                corr_list.append(c)
        scores[lam] = np.nanmean(corr_list) if corr_list else np.nan
    df = pd.DataFrame({"lambda": list(scores.keys()), "cv_corr": list(scores.values())}).dropna().sort_values("lambda")
    df.to_csv(VIZ_DIR / "pb_cv_lambda_summary_process_v6_btop50.csv", index=False, float_format="%.6f")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["lambda"], df["cv_corr"], lw=1.8)
    ax.set_xscale("log"); ax.set_xlabel("λ (Ridge)"); ax.set_ylabel("Mean OOS corr (CV)")
    ax.set_title("CV λ‑Path (blocked, purged)")
    ax.grid(True, alpha=0.3)
    _save_only(fig, VIZ_DIR / "PB_cv_lambda_path_process_v6_btop50.png")

def plot_frozen_vs_live_oos(rep_live: pd.Series, rep_frozen_scaled_net: pd.Series, bench: pd.Series):
    idx = rep_frozen_scaled_net.index.intersection(rep_live.index).intersection(bench.index)
    if len(idx) == 0:
        return
    r_live = rep_live.loc[idx]; r_froz = rep_frozen_scaled_net.loc[idx]; b = bench.loc[idx]
    fig, ax = plt.subplots(figsize=(14,8))
    for ser, label in [(r_live, "Live (net)"), (r_froz, "Frozen (overlay-scaled, net)"), (b, "Benchmark")]:
        ax.plot(_to_naive((1+ser).cumprod().index), 100*(1+ser).cumprod(), label=label, lw=1.6)
    ax.set_yscale("log"); ax.set_title("Frozen vs Live — OOS (log)")
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(VIZ_DIR / "PB_frozen_vs_live_oos_process_v6_btop50.png", bbox_inches="tight")
    if SHOW_FIGS: plt.show(); plt.close(fig)
    else: plt.close(fig)

def main_btop50():
    print("\n==================================================================")
    print("PROCESS-BASED BTOP50")
    print("==================================================================\n")
    print("Building BTOP50 excess returns from Excel + 3m T-bill ...")
    rf = build_tbill_rf_daily(TBILL_XLSX)
    y = build_btop50_excess_from_excel(BTOP_XLSX, rf)
    cal = pd.date_range(y.index.min(), y.index.max(), freq="B", tz="UTC")
    y = y.reindex(cal)
    print(f"BTOP50 realized vol (excess): {ann(y):.1%}\n")
    X_all = load_xlarge()
    X_all = X_all.reindex(cal).astype("float64")
    cal2, y2, X_all2 = align_to_common_span(y=y, X=X_all)
    print(f"Aligned span: {cal2[0].date()} → {cal2[-1].date()}  (y last = {y2.dropna().index.max().date() if y2.dropna().size else 'n/a'}, panel last = {X_all2.dropna(how='all').index.max().date() if not X_all2.dropna(how='all').empty else 'n/a'})")
    cal, y, X_all = cal2, y2, X_all2
    print("Selecting ...")
    chosen, hits, misses = pick_alias_columns(X_all)
    sel = list(chosen.values())
    X27 = X_all[sel].copy()
    rename = {v:k for k,v in chosen.items() if v in X27.columns}
    X27.rename(columns=rename, inplace=True)
    print(f"   Found {len(hits)}/{len(PAPER_27)} contracts.")
    if misses:
        print(f" Missing: {misses}  (consider adding better proxies)")
    if CFG.USE_CARRY and CFG.KEEP_CARRY:
        print("Building carry signals (Carver-style) for selected aliases ...")
        build_carry_signals_for_aliases(chosen_map=chosen, rebuild=False)
    carry_panel = pd.DataFrame(index=cal)
    if CFG.USE_CARRY and CFG.KEEP_CARRY:
        print("Loading Carver-style carry signals ...")
        carry_panel = load_carry_signals(CFG.CARRY_SIGNALS_DIR, chosen_map=chosen, keep_spans=CFG.KEEP_CARRY, cal=cal)
        if carry_panel.empty:
            print("No carry signals found — proceeding without CARRY family.")
        else:
            (CFG.OUT_DIR / "library").mkdir(exist_ok=True)
            carry_panel.to_parquet(CFG.OUT_DIR / "library" / "carry_signals_panel_btop50.parquet", compression="snappy")
            loaded = sorted(set(carry_panel.columns.get_level_values("contract")))
            print(f"   Carry contracts loaded: {len(loaded)} ({', '.join(loaded)})")
    print("Building strategy library (multi-family) + per-strategy pre-scale (t-1) ...")
    R = build_strategy_families(returns_panel=X27, trend_lbs=CFG.LOOKBACKS, breakout_lbs=CFG.KEEP_BREAKOUT, skewabs_lbs=CFG.KEEP_SKEWABS, accel_lbs=CFG.KEEP_ACCEL, carry_panel=carry_panel)
    (CFG.OUT_DIR / "library").mkdir(exist_ok=True)
    R.to_parquet(CFG.OUT_DIR / "library" / "strategy_library_multifamily_btop50.parquet", compression="snappy")
    print("Saved strategy library\n")
    print("Cross-validating lambda (purged blocks; maximize correlation) ...")
    lam0 = cv_lambda_on_corr(R, y, CFG.LAMBDA_GRID, CFG.CV_FOLDS, end_date=CFG.LAMBDA_FIX_DATE)
    print("\n Fitting quarterly betas (positive, L1=1, 5% cap + renorm) ...")
    B = fit_quarterly_betas(R, y, lam0, fix_after=CFG.LAMBDA_FIX_DATE)
    B.to_parquet(CFG.OUT_DIR / "weights_daily_btop50.parquet", compression="snappy")
    print("Saved daily weights")
    first_w = B.dropna(how="all").index.min()
    if pd.isna(first_w):
        raise RuntimeError("No non-NaN weights were produced — check the quarterly betas step.")
    W_eff = B.loc[first_w:].ffill()
    R_eff = R.loc[first_w:]
    y_eff = y.reindex(R_eff.index)
    print("\n--- Sanity checks ---")
    print("First weight date          :", first_w.date())
    print("First non-NaN in y_eff     :", y_eff.first_valid_index().date() if y_eff.first_valid_index() is not None else None)
    print("Library shape (R)          :", R.shape)
    print("\n Availability-aware renormalization of weights ...")
    W_eff_adj = availability_adjust_weights(W_eff, R_eff, cap=CFG.MAX_SINGLE_WEIGHT, l1_target=CFG.L1_TARGET)
    live_mass_under_cap = (W_eff.where(R_eff.notna(), 0.0)).sum(axis=1).clip(upper=1.0)
    gate_thresh = 0.80
    gate_auto = live_mass_under_cap[live_mass_under_cap >= gate_thresh].index.min()
    if pd.isna(gate_auto):
        gate_auto = live_mass_under_cap[live_mass_under_cap > 0].index.min()
    gate = gate_auto
    if CFG.GATE_OVERRIDE is not None and pd.notna(gate):
        gate = max(gate, CFG.GATE_OVERRIDE)
    if pd.notna(gate):
        msg = f"Gating start at {gate.date()} (live L1 ≥ {int(gate_thresh*100)}%)"
        if CFG.GATE_OVERRIDE is not None and pd.notna(gate_auto):
            msg += f" | auto={gate_auto.date()} | override={CFG.GATE_OVERRIDE.date()}"
        print(msg)
        R_eff = R_eff.loc[gate:]
        W_eff_adj = W_eff_adj.loc[gate:]
        y_eff = y_eff.loc[gate:]
        if CFG.SOFT_START_DAYS > 0:
            end_ix = min(CFG.SOFT_START_DAYS, len(W_eff_adj))
            ramp = np.linspace(1/(end_ix), 1.0, end_ix)
            W_eff_adj.iloc[:end_ix,:] = (W_eff_adj.iloc[:end_ix,:].T * ramp).T
        W_eff_adj.to_parquet(CFG.OUT_DIR / "weights_daily_availability_adjusted_gated_btop50.parquet", compression="snappy")
    else:
        raise RuntimeError("No date with live mass > 0 — check strategy library build.")
    print("\n Generating replication + final overlay (t-1) ...")
    rep, scale, turnover_w, turnover_s = generate_replication(R_eff, W_eff_adj, final_target=CFG.FINAL_VOL_TARGET, cost_bps=CFG.REPL_TRADE_COST_BPS, overlay_cost_bps=CFG.OVERLAY_TRADE_COST_BPS)
    rep.to_csv(CFG.OUT_DIR / "replication_returns_net_btop50.csv", float_format="%.8f")
    scale.to_csv(CFG.OUT_DIR / "scale_factor_btop50.csv", float_format="%.6f")
    turnover_w.to_csv(CFG.OUT_DIR / "turnover_series_weights_btop50.csv", float_format="%.6f")
    turnover_s.to_csv(CFG.OUT_DIR / "turnover_series_overlay_btop50.csv", float_format="%.6f")
    print(" Saved replication returns (net), overlay scale, and turnover series\n")
    perf_start = str(R_eff.index[0].date())
    print(" Performance (from gated start) ...")
    perf = performance_table(rep, y_eff, start=perf_start)
    print(perf.to_string())
    make_rev26h_style_figures_process(rep=rep, y=y_eff, beta_daily=W_eff_adj, weight_by_lb_daily=W_eff_adj.groupby(level="lookback", axis=1).sum(), turnover_portfolio=turnover_w)
    make_rev26h_extras(rep=rep, y=y_eff, scale=scale, beta_daily=W_eff_adj, R_eff=R_eff)
    export_turnover_by_lookback(W_eff_adj)
    export_implementation_quality(rep, y_eff, scale, turnover_w, turnover_s, W_eff_adj)
    export_oos_performance(rep, y_eff)
    make_calendar_year_bars(rep, y_eff)
    make_monthly_heatmap(rep, y_eff)
    make_monthly_scatter_regime(rep, y_eff)
    make_drawdown_analysis(rep, y_eff)
    export_signal_bars_and_csv(R_eff, W_eff_adj, scale, y_eff)
    export_timeavg_lookback_weights_csv(W_eff_adj)
    export_library_coverage_heatmap(R_eff)
    export_cv_lambda_path(R, y)
    export_cost_decomposition(turnover_w, turnover_s)
    print("\n=== IN-SAMPLE / OUT-OF-SAMPLE SPLIT (effective, gated span) ===")
    oos_start = CFG.OOS_START
    rep_is = rep.loc[: oos_start - pd.Timedelta(days=1)]
    y_is = y_eff.loc[rep_is.index]
    rep_oos = rep.loc[oos_start:]
    y_oos = y_eff.loc[rep_oos.index]
    def brief_stats(r, b):
        corr = r.corr(b)
        te = (r.sub(b, fill_value=0)).std() * np.sqrt(252)
        sr = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else np.nan
        vol = r.std() * np.sqrt(252)
        return corr, te, sr, vol
    is_corr, is_te, is_sr, is_vol = brief_stats(rep_is, y_is)
    oos_corr, oos_te, oos_sr, oos_vol = brief_stats(rep_oos, y_oos)
    print(f"IS  (<= {(oos_start - pd.Timedelta(days=1)).date()}): corr={is_corr:.3f} | TE={is_te:.2%} | SR={is_sr:.2f} | vol={is_vol:.2%}")
    print(f"OOS (>= {oos_start.date()}): corr={oos_corr:.3f} | TE={oos_te:.2%} | SR={oos_sr:.2f} | vol={oos_vol:.2%}")
    pd.DataFrame({"segment": ["IS","OOS"], "corr": [is_corr, oos_corr], "TE": [is_te, oos_te], "SR": [is_sr, oos_sr], "vol": [is_vol, oos_vol]}).to_csv(CFG.OUT_DIR / "oos_split_process_v6_btop50.csv", index=False, float_format="%.6f")
    print("\n Frozen-beta OOS (fit once on pre-OOS; fixed thereafter) ...")
    cut = CFG.OOS_START - pd.Timedelta(days=1)
    R_pre = R.loc[:cut]; y_pre = y.reindex(R_pre.index)
    row_mask = y_pre.notna() & ~R_pre.isna().all(axis=1)
    R_fit = R_pre.loc[row_mask].fillna(0.0)
    y_fit = y_pre.loc[row_mask].astype(float)
    if len(R_fit) < 200:
        raise ValueError(f"Not enough pre-OOS observations after alignment (have {len(R_fit)}).")
    model = Ridge(alpha=float(lam0), fit_intercept=False, positive=CFG.POSITIVE)
    model.fit(R_fit.values, y_fit.values)
    beta0_raw = pd.Series(model.coef_, index=R.columns)
    beta0 = renorm_with_cap(beta0_raw, CFG.MAX_SINGLE_WEIGHT, CFG.L1_TARGET)
    W_frozen = pd.DataFrame(0.0, index=R.index, columns=R.columns)
    W_frozen.loc[CFG.OOS_START:, :] = beta0.values
    rep_oos_frozen = (R.fillna(0.0).mul(W_frozen, axis=0)).sum(axis=1).loc[CFG.OOS_START:]
    bench_oos = y.reindex(rep_oos_frozen.index)
    frozen_sigma = rep_oos_frozen.ewm(span=CFG.EWMA_SPAN_VOL, adjust=False, min_periods=CFG.STARTUP_DELAY_DAYS).std().shift(1) * np.sqrt(CFG.ANN_DAYS)
    frozen_sigma = frozen_sigma.replace([0, np.inf], np.nan).bfill()
    frozen_scale = (CFG.FINAL_VOL_TARGET / frozen_sigma).clip(lower=CFG.FINAL_SCALE_FLOOR, upper=CFG.FINAL_SCALE_CAP).fillna(method="bfill").fillna(method="ffill")
    if CFG.FINAL_SCALE_WINSOR > 0:
        med = frozen_scale.rolling(21, min_periods=5).median()
        lo, hi = med*(1-CFG.FINAL_SCALE_WINSOR), med*(1+CFG.FINAL_SCALE_WINSOR)
        frozen_scale = frozen_scale.clip(lower=lo, upper=hi).where(med.notna(), frozen_scale)
    rep_oos_frozen_scaled_gross = rep_oos_frozen * frozen_scale
    delta_w_frozen = W_frozen.diff().abs().sum(axis=1).loc[CFG.OOS_START:]
    one_way_turnover_w_frozen = 0.5 * delta_w_frozen
    cost_w_frozen = one_way_turnover_w_frozen * (CFG.REPL_TRADE_COST_BPS / 1e4)
    l1_prev_frozen = W_frozen.abs().sum(axis=1).loc[frozen_scale.index].shift(1).fillna(0.0)
    one_way_turnover_s_frozen = frozen_scale.diff().abs().fillna(0.0) * l1_prev_frozen
    cost_s_frozen = one_way_turnover_s_frozen * (CFG.OVERLAY_TRADE_COST_BPS / 1e4)
    rep_oos_frozen_scaled_net = rep_oos_frozen_scaled_gross - cost_w_frozen - cost_s_frozen
    rep_oos_frozen_scaled_net.to_csv(CFG.OUT_DIR / "replication_returns_frozen_oos_net_btop50.csv", float_format="%.8f")
    bench_oos = y.reindex(rep_oos_frozen_scaled_net.index)
    fo_corr_s = float(rep_oos_frozen_scaled_net.corr(bench_oos))
    fo_te_s = float((rep_oos_frozen_scaled_net - bench_oos).std() * np.sqrt(252))
    fo_sr_s = float(rep_oos_frozen_scaled_net.mean() / (rep_oos_frozen_scaled_net.std() + 1e-12) * np.sqrt(252))
    fo_vol_s = float(rep_oos_frozen_scaled_net.std() * np.sqrt(252))
    print(f"Frozen OOS (overlay-scaled, net): corr={fo_corr_s:.3f} | TE={fo_te_s:.2%} | SR={fo_sr_s:.2f} | vol={fo_vol_s:.2%}")
    plot_frozen_vs_live_oos(rep.loc[rep_oos_frozen_scaled_net.index], rep_oos_frozen_scaled_net, bench_oos)
    cfg_dump = {"LOOKBACKS": list(map(int, CFG.LOOKBACKS)), "KEEP_BREAKOUT": list(map(int, CFG.KEEP_BREAKOUT)), "KEEP_SKEWABS": list(map(int, CFG.KEEP_SKEWABS)), "KEEP_ACCEL": list(map(int, CFG.KEEP_ACCEL)), "EWMA_SPAN_VOL": CFG.EWMA_SPAN_VOL, "Z_CAP": CFG.Z_CAP, "PRE_SCALE_TARGET": CFG.PRE_SCALE_TARGET, "PRE_SCALE_CAP": CFG.PRE_SCALE_CAP, "PRE_SCALE_FLOOR": CFG.PRE_SCALE_FLOOR, "FINAL_VOL_TARGET": CFG.FINAL_VOL_TARGET, "FINAL_SCALE_CAP": CFG.FINAL_SCALE_CAP, "FINAL_SCALE_FLOOR": CFG.FINAL_SCALE_FLOOR, "FINAL_SCALE_WINSOR": CFG.FINAL_SCALE_WINSOR, "LAMBDA_GRID": [float(x) for x in CFG.LAMBDA_GRID], "LAMBDA_FIX_DATE": str(CFG.LAMBDA_FIX_DATE.date()) if CFG.LAMBDA_FIX_DATE else None, "CV_FOLDS": CFG.CV_FOLDS, "PURGE_GAP_DAYS": CFG.PURGE_GAP_DAYS, "MAX_SINGLE_WEIGHT": CFG.MAX_SINGLE_WEIGHT, "WEIGHT_EMA_SPAN": CFG.WEIGHT_EMA_SPAN, "FORCE_REBUILD_XLARGE": CFG.FORCE_REBUILD_XLARGE, "REPL_TRADE_COST_BPS": CFG.REPL_TRADE_COST_BPS, "OVERLAY_TRADE_COST_BPS": CFG.OVERLAY_TRADE_COST_BPS, "SOFT_START_DAYS": CFG.SOFT_START_DAYS, "OOS_START": str(CFG.OOS_START.date()), "BURN_IN_DAYS": CFG.BURN_IN_DAYS, "USE_CARRY": CFG.USE_CARRY, "KEEP_CARRY": list(map(int, CFG.KEEP_CARRY)), "CARRY_CLIP_ABS_ANNUAL": CFG.CARRY_CLIP_ABS_ANNUAL, "FUT_PRICE_DIR": str(CFG.FUT_PRICE_DIR), "FUT_ROLL_DIR": str(CFG.FUT_ROLL_DIR), "FUT_CFG_DIR": str(CFG.FUT_CFG_DIR), "CARRY_SIGNALS_DIR": str(CFG.CARRY_SIGNALS_DIR), "XLARGE_PATH": str(CFG.XLARGE_PATH), "OUT_DIR": str(CFG.OUT_DIR), "DATA_DIR": str(DATA_DIR), "FIGURES_DIR": str(FIGURES_DIR), "BTOP_XLSX": str(BTOP_XLSX), "TBILL_XLSX": str(TBILL_XLSX)}
    (CFG.OUT_DIR / "config_btop50.json").write_text(json.dumps(cfg_dump, indent=2))
    print("Configuration saved")
    print(f"Results saved to: {CFG.OUT_DIR}")

def run(show_figs: bool | None = None, force_rebuild: bool | None = None):
    global SHOW_FIGS
    if show_figs is not None:
        SHOW_FIGS = bool(show_figs)
    if force_rebuild is not None:
        CFG.FORCE_REBUILD_XLARGE = bool(force_rebuild)
    print(f"Output directory: {CFG.OUT_DIR}")
    print(f"Final volatility target: {CFG.FINAL_VOL_TARGET:.1%}")
    print(f"Using trend lookbacks: {CFG.LOOKBACKS}")
    print(f"Breakout LBs: {CFG.KEEP_BREAKOUT} | SkewAbs LBs: {CFG.KEEP_SKEWABS} | Accel LBs: {CFG.KEEP_ACCEL}\n")
    main_btop50()

if __name__ == "__main__":
    run()
