from __future__ import annotations
import os, json, warnings, logging, re
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

DATA_DIR = Path(config.DATA_DIR)
FIGURES_DIR = Path(config.FIGURES_DIR)
PANEL_OUTPUT_DIR = Path(config.PANEL_OUTPUT_DIR)
SG_TREND_INPUT_DIR = Path(config.SG_TREND_INPUT_DIR)
FUTURES_INPUT_DIR = Path(config.FUTURES_INPUT_DIR)
TBILL_XLSX = DATA_DIR / "US3MT=RR.xlsx"
SG_XLSX = DATA_DIR / "SG CTA Indexes Correct.xlsx"
DEFAULT_XLARGE_PARQUET = PANEL_OUTPUT_DIR / "X_large_universe" / "X_large_universe.parquet"
VIZ_DIR = Path(config.FIGURES_DIR) / "sg_cta_visualizations" / "process_based_results"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")
if not logging.getLogger().handlers:
    logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"), format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("sg_cta_process")

plt.rcParams.update({
    "figure.figsize": (14, 8),
    "savefig.dpi": 300,
    "figure.dpi": 110,
    "font.family": "serif",
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "legend.fontsize": 9
})
SHOW_FIGS = bool(int(os.getenv("SHOW_FIGS", "1")))

def _save_only(fig, path: Path):
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def _to_utc_index(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(idx))
    return idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")

def _to_naive(dt_or_idx):
    if isinstance(dt_or_idx, pd.DatetimeIndex):
        return dt_or_idx.tz_convert(None) if getattr(dt_or_idx, 'tz', None) else dt_or_idx
    dt = pd.to_datetime(dt_or_idx)
    return dt.tz_convert(None) if getattr(dt, 'tzinfo', None) else dt

def build_all_sg_excess_from_excels(data_dir: Path, tbill_xlsx: Path, sg_xlsx: Path) -> Dict[str, pd.Series]:
    assert tbill_xlsx.exists(), f"Missing T-bill file: {tbill_xlsx}"
    assert sg_xlsx.exists(), f"Missing SG workbook: {sg_xlsx}"
    tb = pd.read_excel(tbill_xlsx, sheet_name="Table Data", parse_dates=["Date"]).dropna(subset=["Date"])
    yield_col = "US3MT=RR (BID_YIELD)"
    if yield_col not in tb.columns:
        numeric_cols = [c for c in tb.columns if c.lower() != "date" and pd.api.types.is_numeric_dtype(tb[c])]
        if not numeric_cols:
            raise KeyError("Could not find a numeric yield column in the T-bill workbook.")
        yield_col = numeric_cols[0]
    tb = tb.rename(columns={yield_col: "y"}).set_index("Date").sort_index()
    tb.index = _to_utc_index(tb.index)
    rf_daily = tb["y"]/100.0/360.0
    rf_daily = rf_daily.reindex(pd.date_range(rf_daily.index.min(), rf_daily.index.max(), freq="B", tz="UTC")).ffill()
    book = pd.ExcelFile(sg_xlsx)
    out: Dict[str, pd.Series] = {}
    price_cands = ["Last Price", "PX_LAST", "Close", "Price", "Last"]
    for sheet in book.sheet_names:
        df = pd.read_excel(book, sheet_name=sheet, parse_dates=["Date"])
        price_col = next((c for c in price_cands if c in df.columns), None)
        if (price_col is None) or ("Date" not in df.columns):
            continue
        idx = df[["Date", price_col]].dropna().assign(Date=lambda d: _to_utc_index(d["Date"])).set_index("Date").sort_index().asfreq("B")
        total = idx[price_col].pct_change()
        common = total.index.intersection(rf_daily.index)
        excess = (total.loc[common] - rf_daily.loc[common]).astype(np.float32)
        slug = sheet.replace(" ", "_")
        out[slug] = excess.dropna()
        try:
            out[slug].to_pickle(data_dir / f"{slug}_Excess_Returns.pkl")
        except Exception:
            pass
    if out:
        combined = pd.concat(out, axis=1).sort_index().astype(np.float32)
        try:
            combined.to_pickle(data_dir / "All_SG_Indices_Excess_Returns.pkl")
        except Exception:
            pass
    return out

def load_calendar_and_neixcta(data_dir: Path, tbill_xlsx: Path, sg_xlsx: Path) -> Tuple[pd.DatetimeIndex, pd.Series]:
    excess = build_all_sg_excess_from_excels(data_dir, tbill_xlsx, sg_xlsx)
    def _norm(s: str) -> str:
        return "".join(ch for ch in str(s).lower() if ch.isalnum())
    neixcta_aliases = ["NEIXCTA", "SG_CTA", "SGCTA", "SG CTA", "NEIX CTA"]
    neixctat_aliases = ["NEIXCTAT", "SG_CTA_Trend", "SGCTA_TREND", "SG CTA TREND", "SG TREND", "SG TREND INDEX"]
    key = None; chosen_label = None
    for alias in neixcta_aliases:
        key = next((k for k in excess.keys() if _norm(k) == _norm(alias)), None)
        if key:
            chosen_label = "NEIXCTA"; break
    if key is None:
        for alias in neixctat_aliases:
            key = next((k for k in excess.keys() if _norm(k) == _norm(alias)), None)
            if key:
                chosen_label = "NEIXCTAT (fallback)"; break
    if key is None:
        raise RuntimeError("Could not find NEIXCTA/Trend in SG workbook.")
    print(f"Using sheet '{key}' as {chosen_label} target.")
    y = excess[key].copy()
    y.index = _to_utc_index(y.index)
    y = y.sort_index()
    cal = pd.date_range(y.index.min(), y.index.max(), freq="B", tz="UTC")
    y = y.reindex(cal)
    return cal, y

def ensure_neixcta_pickles():
    SG_TREND_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    y_pkl = SG_TREND_INPUT_DIR / "y_excess_NEIXCTA.pkl"
    cal_pkl = SG_TREND_INPUT_DIR / "sg_trend_calendar.pkl"
    if y_pkl.exists() and cal_pkl.exists():
        return
    cal, y = load_calendar_and_neixcta(DATA_DIR, TBILL_XLSX, SG_XLSX)
    y.to_pickle(y_pkl)
    pd.to_pickle(cal, cal_pkl)
    print("Saved NEIXCTA inputs:")
    print("  ", y_pkl)
    print("  ", cal_pkl)

@dataclass
class CFG:
    OUT_DIR: Path = VIZ_DIR
    FUT_PRICE_DIR: Path = field(init=False)
    FUT_ROLL_DIR: Path = field(init=False)
    FUT_CFG_DIR: Path = field(init=False)
    FUT_OUT_DIR: Path = field(init=False)
    CARRY_SIGNALS_DIR: Path = field(init=False)
    XLARGE_PATH: Path = DEFAULT_XLARGE_PARQUET
    FORCE_REBUILD_XLARGE: bool = True
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
    LAMBDA_GRID: np.ndarray = field(default_factory=lambda: np.logspace(-3, 0, 41))
    LAMBDA_FIX_DATE: pd.Timestamp = pd.Timestamp("2007-12-31", tz="UTC")
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
    BUILD_CARRY_INLINE: bool = True
    CARRY_FORCE_REBUILD: bool = False
    CARRY_CAP: float = 20.0
    CARRY_TARGET_ABS: float = 10.0
    CARRY_TRIM_Q: float = 0.995
    CARRY_VOL_DENOM: float = 16.0
    CARRY_SIGMA_FLOOR_PCT: float = 0.05
    CARRY_SIGMA_FLOOR_MIN: float = 1e-9
    CARRY_VOL_SHORT: int = 32
    CARRY_VOL_LONG: int = 252
    CARRY_W_SHORT: float = 0.7
    CARRY_K_MAX: float = 120.0
    CARRY_K_MAX_EQ: float = 400.0
    CARRY_MIN_OBS_FOR_K: int = 250
    CARRY_MIN_OBS_HARD: int = 250
    CARRY_FALLBACK_ALLOW_ANY_FRONT: bool = True
    CARRY_APPLY_COMMODITY_SIGN_FLIP: bool = True
    CARRY_MIN_ROWS_OUTPUT: int = 10
    def __post_init__(self):
        fut_base = Path(config.FUTURES_INPUT_DIR)
        self.FUT_PRICE_DIR = fut_base / "multiple_prices_csv"
        self.FUT_ROLL_DIR = fut_base / "roll_calendars_csv"
        self.FUT_CFG_DIR = fut_base / "csvconfig"
        self.FUT_OUT_DIR = fut_base / "adjusted_prices_csv"
        self.CARRY_SIGNALS_DIR = fut_base / "carry_signals_csv"
        for p in [self.FUT_PRICE_DIR, self.FUT_ROLL_DIR, self.FUT_CFG_DIR]:
            if not p.exists():
                raise FileNotFoundError(f"Missing required futures folder: {p}")
        self.FUT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CARRY_SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
        self.OUT_DIR.mkdir(parents=True, exist_ok=True)
        (self.OUT_DIR / "diagnostics").mkdir(exist_ok=True)

CFG = CFG()

def ann(x: pd.Series) -> float:
    return float(x.std() * np.sqrt(CFG.ANN_DAYS))

def safe_corr(a: pd.Series, b: pd.Series) -> float:
    a, b = a.align(b, join="inner")
    a, b = a.dropna(), b.dropna()
    common = a.index.intersection(b.index)
    if len(common) < 30: return np.nan
    aa, bb = a.loc[common], b.loc[common]
    if aa.std() < 1e-12 or bb.std() < 1e-12: return np.nan
    return float(aa.corr(bb))

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
    price_rows = df[[date_col, pcon_col, price_col]].rename(columns={date_col:"Date", pcon_col:"Contract", price_col:"Close"}).assign(Close=lambda d: pd.to_numeric(d["Close"], errors="coerce") * fac, Kind="Price").dropna(subset=["Close"])
    carry_rows = pd.DataFrame(columns=["Date","Contract","Close","Kind"])
    if carry_col and ccon_col and (carry_col in df.columns) and (ccon_col in df.columns):
        carry_rows = df[[date_col, ccon_col, carry_col]].rename(columns={date_col:"Date", ccon_col:"Contract", carry_col:"Close"}).assign(Close=lambda d: pd.to_numeric(d["Close"], errors="coerce") * fac, Kind="Carry").dropna(subset=["Close"])
    fwd_rows = pd.DataFrame(columns=["Date","Contract","Close","Kind"])
    if fwd_col and fcon_col and (fwd_col in df.columns) and (fcon_col in df.columns):
        fwd_rows = df[[date_col, fcon_col, fwd_col]].rename(columns={date_col:"Date", fcon_col:"Contract", fwd_col:"Close"}).assign(Close=lambda d: pd.to_numeric(d["Close"], errors="coerce") * fac, Kind="Forward").dropna(subset=["Close"])
    long = pd.concat([price_rows, carry_rows, fwd_rows], ignore_index=True).assign(Date=lambda d: _to_utc_index(pd.to_datetime(d["Date"]).dt.normalize())).sort_values(["Date","Contract"])
    long = long[(long["Date"] >= CFG.FILTER_START) & (long["Date"] <= CFG.FILTER_END)]
    return long

def build_xlarge_from_csv() -> pd.DataFrame:
    if not Path(CFG.FUT_PRICE_DIR).exists():
        raise FileNotFoundError(f"Missing FUT_PRICE_DIR ({CFG.FUT_PRICE_DIR})")
    if not Path(CFG.FUT_ROLL_DIR).exists():
        raise FileNotFoundError(f"Missing FUT_ROLL_DIR ({CFG.FUT_ROLL_DIR})")
    if not Path(CFG.FUT_CFG_DIR).exists():
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
        roll = pd.read_csv(roll_file, parse_dates=["DATE_TIME"]).rename(columns={"DATE_TIME":"RollDate","current_contract":"Current","next_contract":"Next"})
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
            carry_df = long.query("Kind == 'Carry'").pivot_table(index="Date", columns="Contract", values="Close", aggfunc="last")
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
    "AD":["AUD","AD","6A"],
    "BP":["GBP","BP","6B"],
    "CD":["CAD","CD","6C"],
    "EC":["EUR","EC","6E"],
    "JY":["JPY","JY","6J"],
    "SF":["CHF","SF","6S"],
    "TU":["US2Y","US2","TU","SHATZ"],
    "FV":["US5Y","US5","FV","BOBL"],
    "TY":["US10Y","US10","TY","OAT"],
    "US":["US","US30","US30Y","US20","BUXL"],
    "G":["GILT","LONG_GILT","UK_GILT","G"],
    "RX":["BUND","RX"],
    "H":["BOBL","SHATZ","H"],
    "ES":["S&P500","SP500","SPX","ES"],
    "NQ":["NASDAQ","NQ"],
    "YM":["DOW","DJI","YM"],
    "Z":["FTSE","FTSE100","Z"],
    "VG":["EUROSTOXX","EUROSTX","EURO600","VG"],
    "AP":["ASX200","SPI200","AP"],
    "NKD":["TOPIX","NIKKEI","NI","JP-REALESTATE","NKD"],
    "GC":["GOLD","GOLD-mini","GC"],
    "SI":["SILVER","SI"],
    "HG":["COPPER","COPPER-mini","COPPER-micro","COPPER_LME","HG"],
    "CL":["WTI_CRUDE","CRUDE_W","CRUDE_ICE","BRENT_W","BRENT_CRUDE","BRENT-LAST","BRE","CL"],
    "HO":["HEATOIL","GASOIL","HEATOIL-ICE","HO"],
    "NG":["NATGAS","GAS_US","GAS-LAST","GAS-PEN","NG"],
    "XB":["RBOB","GASOLINE","GASOILINE","GASOILINE_ICE","RB","XB"]
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

def _tokenize(sym: str) -> list[str]:
    return [t for t in re.split(r'[^A-Z0-9]+', str(sym).upper()) if t]

_COMMODITY_STEMS = {"BRENT","WTI","CRUDE","OIL","GASOIL","HEATOIL","RBOB","GASOLINE","NATGAS","GOLD","SILVER","PLATIN","PALLAD","ALUMIN","ALUMINIUM","COPPER","ZINC","TIN","LEAD","NICKEL","COAL","IRON","STEEL","COFFEE","ROBUSTA","COCOA","COTTON","SUGAR","CORN","WHEAT","SOY","OATS","RICE","CATTLE","HOG","BUTTER","CHEESE","MILK","WHEY","RUBBER","ETHANOL","EUA"}

SIGN_MAP = {
    "BRENT_W": -1, "BRENT-LAST": -1, "CRUDE_ICE": -1, "CRUDE_W": -1,
    "GAS-LAST": -1, "GAS-PEN": -1, "GAS_US": -1, "GAS_US_mini": -1,
    "GASOIL": -1, "GASOILINE": -1, "GASOILINE_ICE": -1,
    "HEATOIL": -1, "HEATOIL-ICE": -1, "NATGAS": -1, "EUA": -1, "COAL": -1, "COAL-GEORDIE": -1,
    "ALUMINIUM": -1, "ALUMINIUM_LME": -1, "GOLD": -1, "GOLD-mini": -1, "SILVER": -1,
    "COPPER": -1, "COPPER-mini": -1, "COPPER-micro": -1, "COPPER_LME": -1,
    "LEAD_LME": -1, "NICKEL_LME": -1, "ZINC_LME": -1, "TIN_LME": -1,
    "PALLAD": -1, "PLAT": -1, "IRON": -1, "STEEL": -1,
    "CORN": -1, "WHEAT": -1, "WHEAT_ICE": -1, "REDWHEAT": -1, "RICE": -1,
    "SOY": -1, "SOYBEAN": -1, "SOYMEAL": -1, "SOYOIL": -1, "OATIES": -1,
    "CANOLA": -1, "COTTON2": -1,
    "SUGAR11": -1, "SUGAR16": -1, "SUGAR_WHITE": -1,
    "COCOA": -1, "COCOA_LDN": -1, "COFFEE": -1, "ROBUSTA": -1, "OJ": -1,
    "FEEDCOW": -1, "LIVECOW": -1, "LEANHOG": -1,
    "BUTTER": -1, "CHEESE": -1, "MILK": -1, "MILKDRY": -1, "MILKWET": -1, "WHEY": -1,
    "RUBBER": -1, "ETHANOL": -1,
    "SWISSLEAD": +1, "EU-OIL": +1, "EU-DJ-OIL": +1, "BBCOMM": +1,
}
NON_COMMODITY_OVERRIDES = {"SWISSLEAD", "EU-OIL", "EU-DJ-OIL", "BBCOMM"}

def _looks_like_commodity_symbol(sym: str) -> bool:
    s = str(sym).upper()
    if s in NON_COMMODITY_OVERRIDES:
        return False
    parts = _tokenize(s)
    if any(p in _COMMODITY_STEMS for p in parts):
        return True
    for p in parts:
        for stem in _COMMODITY_STEMS:
            if len(stem) >= 4 and stem in p:
                return True
    return False

def _cap_for_symbol(sym: Optional[str]) -> float:
    s = (sym or "").upper()
    if s in NON_COMMODITY_OVERRIDES:
        return CFG.CARRY_K_MAX_EQ
    if s in SIGN_MAP and SIGN_MAP.get(s, +1) == -1:
        return CFG.CARRY_K_MAX
    return CFG.CARRY_K_MAX if _looks_like_commodity_symbol(s) else CFG.CARRY_K_MAX_EQ

def _read_csv_fast(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, engine="pyarrow", **kwargs)
    except Exception:
        return pd.read_csv(path, **kwargs)

def _parse_dt_mixed(x) -> pd.DatetimeIndex:
    s = pd.to_datetime(pd.Series(x), errors="coerce", utc=False)
    try:
        s = s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return s.dt.normalize()

_MONTH_LETTER = {'F':'01','G':'02','H':'03','J':'04','K':'05','M':'06','N':'07','Q':'08','U':'09','V':'10','X':'11','Z':'12'}
_MONTH_NAME = {'JAN':'01','FEB':'02','MAR':'03','APR':'04','MAY':'05','JUN':'06','JUL':'07','AUG':'08','SEP':'09','SEPT':'09','OCT':'10','NOV':'11','DEC':'12'}

def _to_four_digit_year(y: str) -> str:
    if len(y) == 4 and y.isdigit():
        return y
    if len(y) == 2 and y.isdigit():
        yy = int(y)
        return f"19{y}" if yy >= 70 else f"20{y}"
    return ""

def _canon_yyyymm(x: object):
    s = str(x).strip().upper()
    if not s:
        return np.nan
    m = re.search(r'(19|20)\d{2}(0[1-9]|1[0-2])', s)
    if m: return m.group(0)
    m = re.search(r'(19|20)\d{2}(0[1-9]|1[0-2])\d{2}', s)
    if m: return m.group(0)[:6]
    m = re.search(r'([FGHJKMNQUVXZ])\s*(\d{2,4})', s) or re.search(r'(\d{2,4})\s*([FGHJKMNQUVXZ])', s)
    if m:
        g1, g2 = m.group(1), m.group(2)
        if g1.isalpha():
            mon = _MONTH_LETTER.get(g1, ""); year = _to_four_digit_year(g2)
        else:
            year = _to_four_digit_year(g1); mon = _MONTH_LETTER.get(g2, "")
        ym = (year + mon) if (year and mon) else ""
        return ym if len(ym) == 6 else np.nan
    m = re.search(r'(' + '|'.join(_MONTH_NAME.keys()) + r')\s*[-/]?\s*(\d{2,4})', s) or re.search(r'(\d{2,4})\s*[-/]?\s*(' + '|'.join(_MONTH_NAME.keys()) + r')', s)
    if m:
        if m.group(1) in _MONTH_NAME:
            mon = _MONTH_NAME[m.group(1)]; year = _to_four_digit_year(m.group(2))
        else:
            year = _to_four_digit_year(m.group(1)); mon = _MONTH_NAME[m.group(2)]
        ym = (year + mon) if (year and mon) else ""
        return ym if len(ym) == 6 else np.nan
    m = re.search(r'(19|20)\d{2}', s)
    if m:
        year = m.group(0); m2 = re.search(r'(0[1-9]|1[0-2])', s[m.end():m.end()+4])
        if m2: return year + m2.group(1)
    return np.nan

def _looks_like_roll_header(cols: List[str]) -> bool:
    L = [c.lower() for c in cols]
    has_date = any(c in L for c in ("date_time","datetime","date"))
    has_curr = any(("current" in c) and ("next" not in c) for c in L)
    has_carry = any("carry" in c for c in L)
    looks_price = (("price" in L) and ("forward" in L)) or (any("price_contract" in c for c in L) and any("forward_contract" in c for c in L))
    return has_date and has_curr and has_carry and not looks_price

def _looks_like_price_header(cols: List[str]) -> bool:
    L = [c.lower() for c in cols]
    req = {"datetime","price","price_contract","forward","forward_contract"}
    return req.issubset(set(L))

def _classify_csv(path: Path) -> str:
    try:
        head = _read_csv_fast(path, nrows=32); cols = list(head.columns)
    except Exception:
        return "unknown"
    if _looks_like_roll_header(cols):  return "roll"
    if _looks_like_price_header(cols): return "price"
    return "unknown"

def _price_path_for_symbol(symbol: str) -> Optional[Path]:
    p = Path(CFG.FUT_PRICE_DIR) / f"{symbol}.csv"
    if p.exists() and _classify_csv(p) == "price":
        return p
    log.warning(f"{symbol}: no exact price file found in {CFG.FUT_PRICE_DIR}")
    return None

def _load_roll_calendar(csv_path: Path) -> pd.DataFrame:
    df = _read_csv_fast(csv_path)
    cols = {c.lower(): c for c in df.columns}
    dcol = next((cols[k] for k in ("date_time","datetime","date") if k in cols), None)
    ccur = cols.get("current_contract") or next((v for k,v in cols.items() if "current" in k and "next" not in k), None)
    ccar = cols.get("carry_contract") or next((v for k,v in cols.items() if "carry" in k), None)
    if not (dcol and ccur and ccar):
        raise KeyError(f"{csv_path.name}: required columns not found (DATE_TIME/current_contract/carry_contract).")
    r = df[[dcol, ccur, ccar]].rename(columns={dcol:"DATE_TIME", ccur:"current_contract", ccar:"carry_contract"})
    r["DATE_TIME"] = _parse_dt_mixed(r["DATE_TIME"])
    r["current_contract"] = r["current_contract"].map(_canon_yyyymm)
    r["carry_contract"] = r["carry_contract"].map(_canon_yyyymm)
    r = r.dropna().drop_duplicates(subset=["DATE_TIME"], keep="last").sort_values("DATE_TIME")
    if r.empty:
        return pd.DataFrame()
    idx = pd.date_range(r["DATE_TIME"].min(), r["DATE_TIME"].max(), freq="B")
    out = r.set_index("DATE_TIME").reindex(idx).ffill()
    out.index.name = "DATETIME"
    return out

def _load_price_data(csv_path: Path) -> pd.DataFrame:
    df = _read_csv_fast(csv_path)
    cols = {c.lower(): c for c in df.columns}
    needed = ("datetime","price","price_contract","forward","forward_contract")
    if not all(k in cols for k in needed):
        missing = [k for k in needed if k not in cols]
        raise KeyError(f"{csv_path.name}: missing columns {missing}")
    d = df[[cols["datetime"], cols["price"], cols["price_contract"], cols["forward"], cols["forward_contract"]]].copy()
    d.columns = ["DATETIME","PRICE","PRICE_CONTRACT","FORWARD","FORWARD_CONTRACT"]
    d["DATETIME"] = _parse_dt_mixed(d["DATETIME"])
    d["PRICE_CONTRACT"] = d["PRICE_CONTRACT"].map(_canon_yyyymm)
    d["FORWARD_CONTRACT"] = d["FORWARD_CONTRACT"].map(_canon_yyyymm)
    d = d.dropna(subset=["DATETIME","PRICE","FORWARD","PRICE_CONTRACT","FORWARD_CONTRACT"]).drop_duplicates(subset=["DATETIME","PRICE_CONTRACT","FORWARD_CONTRACT"]).sort_values("DATETIME")
    return d

def _flip_if_needed(base: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    c_price_eq_front = int((base["PRICE_CONTRACT"] == base["current_contract"]).sum())
    c_fwd_eq_front = int((base["FORWARD_CONTRACT"] == base["current_contract"]).sum())
    c_price_eq_carry = int((base["PRICE_CONTRACT"] == base["carry_contract"]).sum())
    c_fwd_eq_carry = int((base["FORWARD_CONTRACT"] == base["carry_contract"]).sum())
    flip_score = (c_fwd_eq_front + c_price_eq_carry) - (c_price_eq_front + c_fwd_eq_carry)
    thresh = max(25, 0.5 * (c_price_eq_front + c_fwd_eq_front + 1))
    flipped = False
    if flip_score > thresh:
        base = base.copy()
        base.rename(columns={"PRICE": "TMP_P", "PRICE_CONTRACT": "TMP_PC", "FORWARD": "PRICE", "FORWARD_CONTRACT": "PRICE_CONTRACT"}, inplace=True)
        base.rename(columns={"TMP_P": "FORWARD", "TMP_PC": "FORWARD_CONTRACT"}, inplace=True)
        flipped = True
    return base, flipped

def _ym_int(s: pd.Series) -> pd.Series:
    y = pd.to_numeric(s.str.slice(0, 4), errors="coerce")
    m = pd.to_numeric(s.str.slice(4, 6), errors="coerce")
    return (y * 12 + m).astype("Int64")

def _maybe_flip_sign(sym: str, series: pd.Series) -> pd.Series:
    sym_norm = str(sym).upper()
    if sym_norm in SIGN_MAP:
        return series * SIGN_MAP[sym_norm]
    if sym_norm in NON_COMMODITY_OVERRIDES:
        return series
    if CFG.CARRY_APPLY_COMMODITY_SIGN_FLIP and _looks_like_commodity_symbol(sym_norm):
        return -series
    return series

def _compute_raw_forecast(roll_daily: pd.DataFrame, price_df: pd.DataFrame, symbol: Optional[str] = None) -> Tuple[pd.Series, Dict[str, int]]:
    base = price_df.merge(roll_daily, left_on="DATETIME", right_index=True, how="inner").loc[:, ["DATETIME","PRICE","FORWARD","PRICE_CONTRACT","FORWARD_CONTRACT","current_contract","carry_contract"]].dropna(subset=["PRICE_CONTRACT","FORWARD_CONTRACT","current_contract","carry_contract"]).copy()
    base, flipped = _flip_if_needed(base)
    def _pick_best_per_date(df: pd.DataFrame, same_front_only: bool) -> pd.DataFrame:
        g = df.dropna(subset=["PRICE_CONTRACT","FORWARD_CONTRACT","current_contract","carry_contract"])
        if g.empty:
            return g
        pc_ym = _ym_int(g["PRICE_CONTRACT"]).astype("float64")
        fc_ym = _ym_int(g["FORWARD_CONTRACT"]).astype("float64")
        cur_ym = _ym_int(g["current_contract"]).astype("float64")
        car_ym = _ym_int(g["carry_contract"]).astype("float64")
        near_is_price = (pc_ym <= fc_ym)
        near_ym = np.where(near_is_price, pc_ym, fc_ym)
        far_ym = np.where(near_is_price, fc_ym, pc_ym)
        if same_front_only:
            keep = (near_ym == cur_ym.values)
            g = g.loc[keep]
            if g.empty: return g
            pc_ym, fc_ym, cur_ym, car_ym = pc_ym[keep], fc_ym[keep], cur_ym[keep], car_ym[keep]
            near_ym, far_ym = near_ym[keep], far_ym[keep]
        g = g.copy()
        g["_mgap"] = (far_ym - near_ym)
        g["_tgt"] = np.abs((car_ym - cur_ym))
        g["_pen"] = (g["_mgap"] - g["_tgt"]).abs()
        idx = g.groupby("DATETIME")["_pen"].idxmin()
        best = g.loc[idx].copy()
        best = best[best["_mgap"] > 0]
        best["_sm"] = False
        return best
    same_front_best = _pick_best_per_date(base, same_front_only=True)
    strict_days = same_front_fb_days = 0
    if not same_front_best.empty:
        pc = _ym_int(same_front_best["PRICE_CONTRACT"]).astype("float64")
        fc = _ym_int(same_front_best["FORWARD_CONTRACT"]).astype("float64")
        far_contract = np.where(pc <= fc, same_front_best["FORWARD_CONTRACT"], same_front_best["PRICE_CONTRACT"])
        is_strict = (far_contract == same_front_best["carry_contract"])
        strict_days = int(is_strict.sum())
        same_front_fb_days = int((~is_strict).sum())
    if CFG.CARRY_FALLBACK_ALLOW_ANY_FRONT:
        covered = set(same_front_best["DATETIME"].unique()) if not same_front_best.empty else set()
        any_front_best = _pick_best_per_date(base.loc[~base["DATETIME"].isin(covered)], same_front_only=False)
    else:
        any_front_best = base.iloc[0:0]
    matched = pd.concat([same_front_best, any_front_best], ignore_index=True).sort_values("DATETIME")
    if matched.empty:
        return pd.Series(dtype=float), {"days_total": 0, "days_strict": 0, "days_same_front_fb": 0, "days_any_front_fb": 0, "days_sign_mismatch_used": 0, "flipped_columns": int(flipped), "coverage": 0.0}
    any_front_fb_days = int(len(any_front_best))
    days_total = int(len(matched))
    days_sign_mismatch_used = int(matched["_sm"].sum()) if "_sm" in matched.columns else 0
    pc_ym = _ym_int(matched["PRICE_CONTRACT"]).astype("float64")
    fc_ym = _ym_int(matched["FORWARD_CONTRACT"]).astype("float64")
    near_is_price = (pc_ym <= fc_ym).to_numpy()
    price_vals = matched["PRICE"].to_numpy(dtype=float)
    fwd_vals = matched["FORWARD"].to_numpy(dtype=float)
    near_price = np.where(near_is_price, price_vals, fwd_vals)
    far_price = np.where(near_is_price, fwd_vals, price_vals)
    mgap_months = np.abs((fc_ym - pc_ym).to_numpy(dtype=float))
    years_gap = mgap_months / 12.0
    years_gap[years_gap == 0.0] = np.nan
    annualized_raw = (far_price - near_price) / years_gap
    near_series = pd.Series(near_price, index=pd.to_datetime(matched["DATETIME"])).sort_index()
    diff = near_series.diff()
    vol_short = diff.rolling(CFG.CARRY_VOL_SHORT, min_periods=max(5, int(CFG.CARRY_VOL_SHORT * 0.2))).std()
    vol_long = diff.rolling(CFG.CARRY_VOL_LONG, min_periods=max(20, int(CFG.CARRY_VOL_LONG * 0.2))).std()
    sigma = CFG.CARRY_W_SHORT * vol_short + (1.0 - CFG.CARRY_W_SHORT) * vol_long
    med = float(sigma.median(skipna=True))
    if not np.isfinite(med) or med <= 0:
        med = float(diff.abs().median(skipna=True))
    floor = max(CFG.CARRY_SIGMA_FLOOR_MIN, CFG.CARRY_SIGMA_FLOOR_PCT * med) if np.isfinite(med) and med > 0 else CFG.CARRY_SIGMA_FLOOR_MIN
    sigma = sigma.fillna(floor).clip(lower=floor)
    denom = sigma.reindex(pd.to_datetime(matched["DATETIME"])).to_numpy() * CFG.CARRY_VOL_DENOM
    with np.errstate(divide='ignore', invalid='ignore'):
        values = annualized_raw / denom
    raw_fc = pd.Series(values, index=pd.to_datetime(matched["DATETIME"])).replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    raw_fc = raw_fc[~raw_fc.index.duplicated(keep="last")]
    meta = {"days_total": days_total, "days_strict": strict_days, "days_same_front_fb": same_front_fb_days, "days_any_front_fb": any_front_fb_days, "days_sign_mismatch_used": days_sign_mismatch_used, "flipped_columns": int(flipped), "coverage": float(len(raw_fc)) / float(days_total) if days_total else 0.0}
    return raw_fc, meta

def _trimmed_mean_abs(s: pd.Series, q: float) -> float:
    if s.empty: return 0.0
    abs_s = s.abs()
    qv = abs_s.quantile(q)
    if not np.isfinite(qv):
        return float(abs_s.mean())
    return float(abs_s.clip(upper=qv).mean())

def _cap_aware_scale(smooth: pd.Series, target: float, cap: float, q: float, symbol: Optional[str]) -> float:
    if smooth is None or smooth.empty or not np.isfinite(target) or target <= 0:
        return 1.0
    nobs = int(smooth.notna().sum())
    if nobs < CFG.CARRY_MIN_OBS_HARD:
        return 1.0
    base_ma = _trimmed_mean_abs(smooth, q=q)
    if nobs < CFG.CARRY_MIN_OBS_FOR_K and base_ma < 1e-6:
        return 1.0
    def f(k: float) -> float:
        return _trimmed_mean_abs((smooth * k).clip(-cap, cap), q)
    if f(0.0) >= target:
        return 0.0
    lo, hi, val_hi, n_guard = 0.0, 1.0, f(1.0), 0
    while (val_hi < target) and (hi < 1e9) and (n_guard < 60):
        hi *= 2.0; val_hi = f(hi); n_guard += 1
    if val_hi < target:
        return min(1.0, _cap_for_symbol(symbol))
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if f(mid) < target: lo = mid
        else: hi = mid
    return min(0.5 * (lo + hi), _cap_for_symbol(symbol))

def _build_carry_series_carver(raw_forecast: pd.Series, symbol: Optional[str], spans: Tuple[int, ...], target_abs: float, cap: float, q: float) -> pd.DataFrame:
    if raw_forecast is None or len(raw_forecast) == 0:
        cols = ["Date"] + [f"Carry{s}" for s in spans]
        return pd.DataFrame(columns=cols)
    sym = symbol or ""
    rf = _maybe_flip_sign(sym, raw_forecast)
    out_idx = pd.DatetimeIndex(sorted(rf.index.unique()))
    out = pd.DataFrame(index=out_idx)
    for span in spans:
        smooth = rf.reindex(out.index).ewm(span=int(span), adjust=False).mean()
        k = _cap_aware_scale(smooth, target=target_abs, cap=cap, q=q, symbol=sym)
        series = (smooth * k).clip(-cap, cap)
        post_ma = _trimmed_mean_abs(series, q=q)
        clip_rate = float((series.abs() >= cap - 1e-12).mean())
        nobs = int(smooth.notna().sum())
        reason = " (short history)" if nobs < CFG.CARRY_MIN_OBS_HARD and k == 1.0 else ""
        log.info(f"{sym} Span {span}: k={k:.4g}{reason}, post-cap trimmed|.|={post_ma:.3f}, clip={clip_rate:.1%}")
        out[f"Carry{int(span)}"] = series
    final = out.reset_index().rename(columns={"index":"Date"})
    final["Date"] = pd.to_datetime(final["Date"]).dt.strftime("%Y-%m-%d")
    return final

def build_carry_for_symbol(symbol: str, rolls_dir: Path, prices_dir: Path, out_dir: Path, spans: Tuple[int, ...], force_rebuild: bool = False) -> bool:
    try:
        out_path = Path(out_dir) / f"{symbol}.csv"
        out_csv = Path(out_dir) / f"{symbol}_carry_signals.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_csv.exists() and (not force_rebuild):
            return True
        rpath = Path(rolls_dir) / f"{symbol}.csv"
        ppath = _price_path_for_symbol(symbol)
        if (not rpath.exists()) or (ppath is None):
            log.warning(f"{symbol}: missing inputs for carry; skip.")
            return False
        roll_daily = _load_roll_calendar(rpath)
        price_df = _load_price_data(ppath)
        raw_fc, meta = _compute_raw_forecast(roll_daily, price_df, symbol=symbol)
        if raw_fc.empty or len(raw_fc) < CFG.CARRY_MIN_ROWS_OUTPUT:
            log.warning(f"{symbol}: no carry signals (len={len(raw_fc)}). strict={meta.get('days_strict',0)}, same-front-fb={meta.get('days_same_front_fb',0)}, any-front-fb={meta.get('days_any_front_fb',0)}, flipped={meta.get('flipped_columns',0)}, coverage={meta.get('coverage',0):.1%}")
            return False
        final_df = _build_carry_series_carver(raw_forecast=raw_fc, symbol=symbol, spans=tuple(sorted(set(int(x) for x in spans))), target_abs=CFG.CARRY_TARGET_ABS, cap=CFG.CARRY_CAP, q=CFG.CARRY_TRIM_Q)
        final_df["symbol"] = symbol
        final_df.to_csv(out_csv, index=False)
        dr = f"{final_df['Date'].min()} → {final_df['Date'].max()}"
        log.info(f"Wrote {out_csv.name}: {len(final_df):,d} rows [{dr}]")
        return True
    except Exception as e:
        log.error(f"FAILED carry build for {symbol}: {e}")
        return False

def maybe_build_carry_signals_inline(symbols_to_build: List[str]) -> None:
    if not CFG.USE_CARRY or not CFG.BUILD_CARRY_INLINE:
        return
    if not symbols_to_build:
        return
    print("\n⚒️  Building Carver-style carry signals inline ...")
    ok, fail = 0, 0
    for sym in sorted(set(symbols_to_build)):
        success = build_carry_for_symbol(symbol=sym, rolls_dir=CFG.FUT_ROLL_DIR, prices_dir=CFG.FUT_PRICE_DIR, out_dir=CFG.CARRY_SIGNALS_DIR, spans=CFG.KEEP_CARRY, force_rebuild=CFG.CARRY_FORCE_REBUILD)
        ok += int(success); fail += int(not success)
    print(f"   Carry build done — success: {ok}, failed: {fail}. Output: {CFG.CARRY_SIGNALS_DIR}")

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
    if any(np.isfinite(v) and v > -np.inf for v in scores.values()):
        best = max(scores, key=lambda k: scores[k])
    else:
        best = float(lambdas[0])
        log.warning("CV produced no finite scores; falling back to smallest lambda.")
    log.info(f"CV best lambda={best:.6f} (CV corr={scores.get(best, np.nan):.3f})")
    return float(best)

def fit_quarterly_betas(R: pd.DataFrame, y: pd.Series, lam: float, fix_after: Optional[pd.Timestamp]) -> pd.DataFrame:
    q_start = R.index.min(); q_end = R.index.max()
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
    lo = med * (1 - w); hi = med * (1 + w)
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
    idx = rep.index.intersection(y.index); idx = idx[idx >= pd.Timestamp(start, tz="UTC")]
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

COLOR_PALETTE = {'model':'#d62728','benchmark':'#9467bd','normal':'#2c3e50','crisis':'#e74c3c'}
CRISIS_PERIODS = [('2007-07-01','2009-03-31','GFC'),('2011-05-01','2011-10-31','EU Debt'),('2015-08-01','2016-02-29','China/Oil'),('2020-02-15','2020-04-30','COVID-19'),('2022-02-15','2022-10-31','Ukraine/Inflation')]

def annotate_crisis_periods(ax):
    xlim = ax.get_xlim()
    for start, end, label in CRISIS_PERIODS:
        sdt = pd.to_datetime(start); edt = pd.to_datetime(end)
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
        ax.plot(_to_naive(y_c.index), y_c.values, label="NEIXCTA (excess)", color=COLOR_PALETTE['benchmark'], lw=1.5)
        ax.set_yscale("log"); ax.grid(True, alpha=0.3)
        ax.set_title(f"Cumulative Growth of $100 (Process-Based, target {CFG.FINAL_VOL_TARGET:.1%})")
        ax.set_ylabel("Portfolio Value ($)"); ax.set_xlabel("Date")
        ax.axvspan(_to_naive(rep.index[0]), _to_naive(burn_in_end), alpha=0.1, color="gray", label=f"{CFG.BURN_IN_MONTHS}-Month Burn-in")
        ax.axvline(_to_naive(oos_start), color="black", ls="--", lw=2, alpha=0.7, label="OOS Start")
        annotate_crisis_periods(ax); ax.legend(loc="upper left", framealpha=0.9)
        plt.tight_layout(); plt.savefig(VIZ_DIR / "A_cumulative_equity_curves_process_v6.png", bbox_inches="tight")
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
            ax1.plot(roll_corr.index, roll_corr, color=COLOR_PALETTE['normal'], lw=1.5)
            ax1.axhline(float(roll_corr.mean()), color="black", ls="--", alpha=0.5, label=f"Mean {roll_corr.mean():.3f}")
            ax1.set_ylabel("Corr"); ax1.legend(); ax1.grid(True, alpha=0.3)
            ax1.set_title("Rolling 1Y Statistics (Ex-Burn-in)")
            ax2.plot(roll_rs2.index, roll_rs2, color=COLOR_PALETTE['model'], lw=1.5)
            ax2.fill_between(roll_rs2.index, 0, roll_rs2, color=COLOR_PALETTE['model'], alpha=0.2)
            ax2.set_ylabel("R²"); ax2.grid(True, alpha=0.3); ax2.axhline(float(roll_rs2.mean()), color="black", ls="--", alpha=0.5)
            ax3.plot(roll_te.index, 100*roll_te, color=COLOR_PALETTE['crisis'], lw=1.5)
            ax3.set_ylabel("TE (%)"); ax3.grid(True, alpha=0.3)
            ax3.axhline(float((100*roll_te).mean()), color="black", ls="--", alpha=0.5, label=f"Mean {float((100*roll_te).mean()):.1f}%")
            ax3.legend()
            ax4.plot(roll_ir.index, roll_ir, color=COLOR_PALETTE['normal'], lw=1.5)
            ax4.axhline(0, color="black", lw=1, alpha=0.5)
            ax4.axhline(float(roll_ir.mean()), color="black", ls="--", alpha=0.5, label=f"Mean IR {float(roll_ir.mean()):.3f}")
            ax4.set_ylabel("IR"); ax4.set_xlabel("Date"); ax4.legend(); ax4.grid(True, alpha=0.3)
            for ax in (ax1, ax2, ax3, ax4):
                annotate_crisis_periods(ax)
                ax.axvline(_to_naive(oos_start), color='gray', ls='--', lw=2, alpha=0.7)
            plt.tight_layout(); plt.savefig(VIZ_DIR / "B_rolling_statistics_process_v6.png", bbox_inches="tight")
            if SHOW_FIGS: plt.show(); plt.close(fig)
            else: plt.close(fig)
        if isinstance(beta_daily.columns, pd.MultiIndex) and 'lookback' in beta_daily.columns.names:
            w_by_lb_abs = beta_daily.abs().groupby(axis=1, level='lookback').sum()
        else:
            w_by_lb_abs = weight_by_lb_daily.abs()
        denom = w_by_lb_abs.sum(axis=1).replace(0, np.nan)
        lb_share = (w_by_lb_abs.T / denom).T.fillna(0.0)
        fig, ax = plt.subplots(figsize=(14, 6))
        lbs = list(lb_share.columns)
        ax.stackplot(lb_share.index, [lb_share[c].values for c in lbs], labels=[f"{c}d" for c in lbs], alpha=0.9)
        ax.set_ylim(0,1); ax.set_ylabel('Share of |weights|'); ax.set_xlabel('Date'); ax.grid(True, alpha=0.3)
        ax.set_title('Lookback Share of Absolute Weights (normalized daily)')
        ax.legend(loc='upper left', ncol=5, bbox_to_anchor=(1.0, 1.0))
        ax.axvline(_to_naive(CFG.OOS_START), color='black', ls='--', lw=2, alpha=0.7)
        plt.tight_layout(); plt.savefig(VIZ_DIR / "E_lookback_weight_share_process_v6.png", bbox_inches="tight")
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
        ax.plot(scale.index, scale.values, lw=1.2)
        ax.axhline(CFG.FINAL_SCALE_CAP, ls="--", alpha=0.5)
        ax.axhline(CFG.FINAL_SCALE_FLOOR, ls="--", alpha=0.5)
        ax.set_title("Overlay Scale (×)"); ax.set_ylabel("×"); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(VIZ_DIR / "D_overlay_scale_process_v6.png", bbox_inches="tight")
        if SHOW_FIGS: plt.show(); plt.close(fig)
        else: plt.close(fig)
    except Exception as e:
        print(f"overlay scale plot failed: {e}")
    try:
        res = (r - b).dropna()
        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(res.values*100, bins=50, alpha=0.85)
        ax.set_title("Residuals Histogram (rep - bench, daily %)"); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(VIZ_DIR / "F_residuals_hist_process_v6.png", bbox_inches="tight")
        if SHOW_FIGS: plt.show(); plt.close(fig)
        else: plt.close(fig)
        fig, ax = plt.subplots(figsize=(14,4))
        ax.plot(res.index, (res.cumsum()*100).values, lw=1.4)
        ax.set_title("Cumulative Residual P&L (rep - bench, % points)"); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(VIZ_DIR / "F_residuals_cum_process_v6.png", bbox_inches="tight")
        if SHOW_FIGS: plt.show(); plt.close(fig)
        else: plt.close(fig)
    except Exception as e:
        print(f"residual plots failed: {e}")

def _monthly_return(s: pd.Series) -> pd.Series:
    return s.resample('M').apply(lambda x: (1.0 + x).prod() - 1.0)

def _calendar_year_return(s: pd.Series) -> pd.Series:
    return s.resample('Y').apply(lambda x: (1.0 + x).prod() - 1.0)

def export_turnover_by_lookback(W_eff_adj: pd.DataFrame):
    if not (isinstance(W_eff_adj.columns, pd.MultiIndex) and 'lookback' in W_eff_adj.columns.names):
        print("export_turnover_by_lookback: columns do not have a 'lookback' level; skipping.")
        return
    by_lb_daily = 0.5 * W_eff_adj.diff().abs().groupby(level='lookback', axis=1).sum()
    by_lb_daily.to_csv(VIZ_DIR / "turnover_by_lookback_daily_process_v6.csv", float_format="%.8f")
    mean_d = by_lb_daily.mean(axis=0); std_d = by_lb_daily.std(axis=0)
    summary = pd.DataFrame({"lookback": mean_d.index.astype(int), "mean_daily": mean_d.values, "std_daily": std_d.values, "annual": (mean_d * 252).values}).sort_values("lookback")
    summary.to_csv(VIZ_DIR / "turnover_by_lookback_process_v6.csv", index=False, float_format="%.8f")
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
    plt.savefig(VIZ_DIR / "C_turnover_by_lookback_process_v6.png", bbox_inches="tight")
    if SHOW_FIGS:
        plt.show()
        plt.close(fig)
    else:
        plt.close(fig)

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
    out.to_csv(CFG.OUT_DIR / "implementation_quality_process_v6.csv", index=False, float_format="%.8f")
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
    df.to_csv(CFG.OUT_DIR / "oos_performance_process_v6.csv", index=False, float_format="%.8f")
    print("\nIS / OOS performance\n", df.to_string(index=False))

def make_calendar_year_bars(rep: pd.Series, y: pd.Series):
    r_y = _calendar_year_return(rep)
    b_y = _calendar_year_return(y.reindex(r_y.index))
    a_y = (1.0 + (rep - y.reindex(rep.index))).resample('Y').apply(lambda x: x.prod() - 1.0)
    tbl = pd.DataFrame({"model": r_y, "benchmark": b_y, "alpha": a_y})
    tbl.index = tbl.index.year
    tbl.to_csv(VIZ_DIR / "calendar_year_table_process_v6.csv", float_format="%.8f")
    fig, ax = plt.subplots(figsize=(14, 8))
    xs = np.arange(len(tbl)); width = 0.28
    ax.bar(xs - width, 100*tbl["model"].values, width, label="Model")
    ax.bar(xs, 100*tbl["benchmark"].values, width, label="Benchmark")
    ax.bar(xs + width, 100*tbl["alpha"].values, width, label="Alpha")
    ax.set_xticks(xs); ax.set_xticklabels(tbl.index.astype(int).astype(str))
    ax.yaxis.set_major_formatter(PercentFormatter(100.0))
    ax.set_title("Calendar-Year Returns")
    ax.grid(True, axis='y', alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(VIZ_DIR / "Y_calendar_year_bars_process_v6.png", bbox_inches="tight")
    if SHOW_FIGS: plt.show(); plt.close(fig)
    else: plt.close(fig)

def make_monthly_heatmap(rep: pd.Series, y: pd.Series, annotate: bool = True, fmt: str = "{:.1f}", diverging: bool = True):
    m_model = _monthly_return(rep)
    m_bench = _monthly_return(y.reindex(m_model.index))
    m_alpha = _monthly_return(rep - y.reindex(rep.index))
    df = pd.DataFrame({"model": m_model, "benchmark": m_bench, "alpha": m_alpha})
    df.to_csv(VIZ_DIR / "monthly_returns_process_v6.csv", float_format="%.8f")
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
    plt.tight_layout(); plt.savefig(VIZ_DIR / "I_monthly_return_heatmap_process_v6.png", bbox_inches="tight")
    if SHOW_FIGS: plt.show(); plt.close(fig)
    else: plt.close(fig)

def make_monthly_scatter_regime(rep: pd.Series, y: pd.Series):
    m_model = _monthly_return(rep)
    m_bench = _monthly_return(y.reindex(m_model.index))
    rs_daily = (y.rolling(63).mean() / (y.rolling(63).std() + 1e-12)) * np.sqrt(252)
    rs_m_ends = rs_daily.reindex(m_bench.index, method="ffill")
    reg = rs_m_ends.apply(classify_regime)
    tmp = pd.DataFrame({"x": m_bench.values, "y": m_model.values, "regime": reg.values}, index=m_model.index)
    tmp.to_csv(VIZ_DIR / "monthly_scatter_data_process_v6.csv", float_format="%.8f")
    colors = {"Strong Bear":"#b2182b","Bear":"#ef8a62","Weak Bear":"#fddbc7","Neutral":"#cccccc","Weak Bull":"#d1e5f0","Bull":"#67a9cf","Strong Bull":"#2166ac"}
    fig, ax = plt.subplots(figsize=(14, 8))
    for g, sub in tmp.groupby("regime"):
        ax.scatter(100*sub["x"], 100*sub["y"], s=18, alpha=0.85, label=g, edgecolors="none", c=colors.get(g, "#333333"))
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1], 5)
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=1, alpha=0.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Benchmark monthly (%)"); ax.set_ylabel("Model monthly (%)")
    ax.set_title("Monthly Return Scatter (regime‑colored)")
    ax.grid(True, alpha=0.3); ax.legend(ncol=3, fontsize=9)
    plt.tight_layout(); plt.savefig(VIZ_DIR / "C_monthly_scatter_regime_process_v6.png", bbox_inches="tight")
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
    dd_tbl.to_csv(VIZ_DIR / "drawdown_stats_process_v6.csv", index=False, float_format="%.8f")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.plot(dd_b.index, 100*dd_b, label="Benchmark"); ax1.plot(dd_r.index, 100*dd_r, label="Model")
    ax1.set_title("Drawdowns (%)"); ax1.grid(True, alpha=0.3); ax1.legend()
    ax2.plot(diff.index, 100*diff, label="Model − Benchmark")
    ax2.set_title("Drawdown Difference (pp)"); ax2.grid(True, alpha=0.3)
    for ax in (ax1, ax2):
        ax.yaxis.set_major_formatter(PercentFormatter(100.0))
        ax.axvline(_to_naive(CFG.OOS_START), color="black", ls="--", lw=2, alpha=0.7)
        annotate_crisis_periods(ax)
    plt.tight_layout(); plt.savefig(VIZ_DIR / "H_drawdown_analysis_process_v6.png", bbox_inches="tight")
    if SHOW_FIGS: plt.show(); plt.close(fig)
    else: plt.close(fig)

def plot_asset_class_rolling_contrib(rolling_ac: pd.DataFrame, outfile: Path):
    fig, ax = plt.subplots(figsize=(14, 8))
    for col in rolling_ac.columns:
        ax.plot(rolling_ac.index, 100*rolling_ac[col], label=col, lw=1.4)
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
    contrib_daily = (contrib_daily.T * scale.reindex(contrib_daily.index)).T
    fam_daily = contrib_daily.groupby(axis=1, level='family').sum()
    fam_m = fam_daily.resample("M").sum()
    fam_m.to_csv(VIZ_DIR / "pb_signal_family_contrib_monthly_process_v6.csv", float_format="%.8f")
    if len(fam_m) > 0:
        start_bar = fam_m.index.max() - pd.DateOffset(years=10)
        fmb = fam_m.loc[start_bar:]
        fig, ax = plt.subplots(figsize=(14, 8))
        bottom = np.zeros(len(fmb))
        for col in fmb.columns:
            ax.bar(fmb.index, 100*fmb[col].values, bottom=bottom, label=col)
            bottom += 100*fmb[col].values
        ax.set_title("Family Contribution — Monthly (last 10y, %)")
        ax.yaxis.set_major_formatter(PercentFormatter(100.0))
        ax.grid(True, axis='y', alpha=0.3); ax.legend(ncol=5, fontsize=8, bbox_to_anchor=(1.0, 1.02), loc="upper right")
        ax.axvline(_to_naive(CFG.OOS_START), color="black", ls="--", lw=2, alpha=0.7)
        _save_only(fig, VIZ_DIR / "PB_signal_family_contrib_monthly_process_v6.png")
        fam_y = fam_daily.resample("Y").sum()
        fam_y.index = fam_y.index.year
        fam_y.to_csv(VIZ_DIR / "pb_signal_family_contrib_calendar_process_v6.csv", float_format="%.8f")
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
        plt.tight_layout(); plt.savefig(VIZ_DIR / "PB_signal_family_contrib_calendar_bars_process_v6.png", bbox_inches="tight")
        if SHOW_FIGS: plt.show(); plt.close(fig)
        else: plt.close(fig)
    lb_daily = contrib_daily.groupby(axis=1, level='lookback').sum()
    lb_m = lb_daily.resample("M").sum()
    lb_m.to_csv(VIZ_DIR / "pb_signal_lookback_contrib_monthly_process_v6.csv", float_format="%.8f")
    if len(lb_m) > 0:
        start_bar = lb_m.index.max() - pd.DateOffset(years=10)
        lbm = lb_m.loc[start_bar:]
        fig, ax = plt.subplots(figsize=(14, 8))
        bottom = np.zeros(len(lbm))
        for col in sorted(lbm.columns):
            ax.bar(lbm.index, 100*lbm[col].values, bottom=bottom, label=f"{int(col)}d")
            bottom += 100*lbm[col].values
        ax.set_title("Lookback Contribution — Monthly (last 10y, %)")
        ax.yaxis.set_major_formatter(PercentFormatter(100.0))
        ax.grid(True, axis='y', alpha=0.3); ax.legend(ncol=6, fontsize=8, bbox_to_anchor=(1.0, 1.02), loc="upper right")
        ax.axvline(_to_naive(CFG.OOS_START), color="black", ls="--", lw=2, alpha=0.7)
        _save_only(fig, VIZ_DIR / "PB_signal_lookback_contrib_monthly_process_v6.png")
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
        plt.tight_layout(); plt.savefig(VIZ_DIR / "PB_signal_lookback_contrib_calendar_bars_process_v6.png", bbox_inches="tight")
        if SHOW_FIGS: plt.show(); plt.close(fig)
        else: plt.close(fig)
    ctrt_daily = contrib_daily.groupby(axis=1, level='contract').sum()
    ctrt_daily.to_csv(VIZ_DIR / "pb_signal_contract_contrib_daily_process_v6.csv", float_format="%.8f")
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
        plt.tight_layout(); plt.savefig(VIZ_DIR / "PB_signal_contract_top10_contrib_bars_process_v6.png", bbox_inches="tight")
        if SHOW_FIGS: plt.show(); plt.close(fig)
        else: plt.close(fig)
        rc = ctrt_daily[top].rolling(63).sum()
        fig, ax = plt.subplots(figsize=(14, 8))
        for c in top:
            ax.plot(rc.index, 100*rc[c], label=c, lw=1.2)
        ax.set_title("Top‑10 Contracts — Rolling 63‑day Contribution (%)")
        ax.yaxis.set_major_formatter(PercentFormatter(100.0))
        ax.grid(True, alpha=0.3); ax.legend(ncol=5)
        plt.tight_layout(); plt.savefig(VIZ_DIR / "PB_signal_contract_top10_contrib_process_v6.png", bbox_inches="tight")
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
    fam_te.to_frame("te_contribution").to_csv(VIZ_DIR / "pb_signal_family_TE_contrib_process_v6.csv", float_format="%.8f")
    fig, ax = plt.subplots(figsize=(12, 6))
    xs = np.arange(len(fam_te))
    ax.bar(xs, 100*fam_te.values)
    ax.set_xticks(xs); ax.set_xticklabels(fam_te.index)
    ax.set_ylabel("Avg reduction in TE (pp)")
    ax.set_title("Family Contribution to Tracking Error (higher = more TE explained)")
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(VIZ_DIR / "PB_signal_family_TE_contrib_process_v6.png", bbox_inches="tight")
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
        plot_asset_class_rolling_contrib(rolling_ac, VIZ_DIR / "PB_asset_class_rolling_contrib_process_v6.png")

def export_library_coverage_heatmap(R_eff: pd.DataFrame):
    live = R_eff.notna().mean(axis=0)
    meta = pd.DataFrame({"family": R_eff.columns.get_level_values("family"), "contract": R_eff.columns.get_level_values("contract"), "lookback": R_eff.columns.get_level_values("lookback").astype(int), "live_frac": live.values})
    agg = meta.groupby(["family", "lookback"]).agg(n_contracts=("contract", "nunique"), avg_live=("live_frac", "mean")).reset_index()
    agg["effective_series"] = agg["n_contracts"] * agg["avg_live"]
    agg.to_csv(VIZ_DIR / "pb_library_summary_process_v6.csv", index=False, float_format="%.6f")
    pivot = agg.pivot(index="family", columns="lookback", values="effective_series")
    pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(pivot.fillna(0.0).values, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns))); ax.set_xticklabels(pivot.columns.astype(int).astype(str))
    ax.set_title("Strategy‑Library Coverage / Density (effective series)")
    cbar = plt.colorbar(im, ax=ax); cbar.ax.set_ylabel("effective series", rotation=270, labelpad=15)
    plt.tight_layout(); plt.savefig(VIZ_DIR / "PB_library_coverage_heatmap_process_v6.png", bbox_inches="tight")
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
    idx = R0.index
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
    df.to_csv(VIZ_DIR / "pb_cv_lambda_summary_process_v6.csv", index=False, float_format="%.6f")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["lambda"], df["cv_corr"], lw=1.8)
    ax.set_xscale("log"); ax.set_xlabel("λ (Ridge)"); ax.set_ylabel("Mean OOS corr (CV)")
    ax.set_title("CV λ‑Path (blocked, purged)")
    ax.grid(True, alpha=0.3)
    _save_only(fig, VIZ_DIR / "PB_cv_lambda_path_process_v6.png")

def export_cost_decomposition(turnover_w: pd.Series, turnover_s: pd.Series):
    cw = (turnover_w * (CFG.REPL_TRADE_COST_BPS / 1e4)).resample("M").sum() * 1e4
    cs = (turnover_s * (CFG.OVERLAY_TRADE_COST_BPS / 1e4)).resample("M").sum() * 1e4
    df = pd.DataFrame({"replication_bps_per_month": cw, "overlay_bps_per_month": cs})
    df.to_csv(VIZ_DIR / "pb_cost_decomposition_process_v6.csv", float_format="%.6f")
    fig, ax = plt.subplots(figsize=(14,8))
    ax.bar(df.index, df["replication_bps_per_month"], label="Replication trading (bps/mo)")
    ax.bar(df.index, df["overlay_bps_per_month"], bottom=df["replication_bps_per_month"], alpha=0.35, label="Overlay rebalancing (bps/mo)")
    ax.set_title("Cost Decomposition (monthly bps, one-way)")
    ax.grid(True, axis='y', alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(VIZ_DIR / "PB_cost_decomposition_process_v6.png", bbox_inches="tight")
    if SHOW_FIGS: plt.show(); plt.close(fig)
    else: plt.close(fig)

def export_timeavg_lookback_weights_csv(W_eff_adj: pd.DataFrame):
    w_by_lb = W_eff_adj.abs().groupby(axis=1, level="lookback").sum()
    share = (w_by_lb.T / w_by_lb.sum(axis=1).replace(0, np.nan)).T
    timeavg = share.mean().sort_index()
    out = pd.DataFrame({"lookback": timeavg.index.astype(int), "share": timeavg.values})
    out.to_csv(VIZ_DIR / "pb_timeavg_lookback_weights_process_v6.csv", index=False, float_format="%.8f")

def plot_frozen_vs_live_oos(rep_live: pd.Series, rep_frozen_scaled_net: pd.Series, bench: pd.Series):
    idx = rep_frozen_scaled_net.index.intersection(rep_live.index).intersection(bench.index)
    if len(idx) == 0:
        return
    r_live = rep_live.loc[idx]; r_froz = rep_frozen_scaled_net.loc[idx]; b = bench.loc[idx]
    fig, ax = plt.subplots(figsize=(14,8))
    for ser, label in [(r_live, "Live (net)"), (r_froz, "Frozen (overlay-scaled, net)"), (b, "Benchmark")]:
        ax.plot((1+ser).cumprod().index, 100*(1+ser).cumprod(), label=label, lw=1.6)
    ax.set_yscale("log"); ax.set_title("Frozen vs Live — OOS (log)")
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(VIZ_DIR / "PB_frozen_vs_live_oos_process_v6.png", bbox_inches="tight")
    if SHOW_FIGS: plt.show(); plt.close(fig)
    else: plt.close(fig)

def main():
    print("\n==================================================================")
    print("PROCESS-BASED SG CTA")
    print("==================================================================\n")
    CFG.OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    ensure_neixcta_pickles()
    print("Building SG excess returns and selecting NEIXCTA ...")
    cal, y = load_calendar_and_neixcta(DATA_DIR, TBILL_XLSX, SG_XLSX)
    print(f"NEIXCTA realized vol: {ann(y):.1%}\n")
    X_all = load_xlarge()
    X_all = X_all.reindex(cal).astype("float64")
    cal2, y2, X_all2 = align_to_common_span(y=y, X=X_all)
    print(f"Aligned span: {cal2[0].date()} → {cal2[-1].date()}  (y last = {y2.dropna().index.max().date() if y2.dropna().size else 'n/a'}, panel last = {X_all2.dropna(how='all').index.max().date() if not X_all2.dropna(how='all').empty else 'n/a'})")
    cal, y, X_all = cal2, y2, X_all2
    print("Selecting paper 27 via robust aliases ...")
    chosen, hits, misses = pick_alias_columns(X_all)
    sel = list(chosen.values())
    X27 = X_all[sel].copy()
    rename = {v:k for k,v in chosen.items() if v in X27.columns}
    X27.rename(columns=rename, inplace=True)
    print(f"   Found {len(hits)}/{len(PAPER_27)} contracts.")
    if misses:
        print(f" Missing: {misses}  (consider adding better proxies)")
    if CFG.USE_CARRY and CFG.BUILD_CARRY_INLINE:
        maybe_build_carry_signals_inline(symbols_to_build=list(chosen.values()))
    carry_panel = pd.DataFrame(index=cal)
    if CFG.USE_CARRY and CFG.KEEP_CARRY:
        print("Loading carry signals ...")
        carry_panel = load_carry_signals(CFG.CARRY_SIGNALS_DIR, chosen_map=chosen, keep_spans=CFG.KEEP_CARRY, cal=cal)
        if carry_panel.empty:
            print("No carry signals found — proceeding without CARRY family.")
        else:
            (CFG.OUT_DIR / "library").mkdir(parents=True, exist_ok=True)
            carry_panel.to_parquet(CFG.OUT_DIR / "library" / "carry_signals_panel.parquet", compression="snappy")
            loaded = sorted(set(carry_panel.columns.get_level_values("contract")))
            print(f"   Carry contracts loaded: {len(loaded)} ({', '.join(loaded)})")
    print("Building strategy library (multi-family) + per-strategy pre-scale (t-1) ...")
    R = build_strategy_families(returns_panel=X27, trend_lbs=CFG.LOOKBACKS, breakout_lbs=CFG.KEEP_BREAKOUT, skewabs_lbs=CFG.KEEP_SKEWABS, accel_lbs=CFG.KEEP_ACCEL, carry_panel=carry_panel)
    (CFG.OUT_DIR / "library").mkdir(parents=True, exist_ok=True)
    R.to_parquet(CFG.OUT_DIR / "library" / "strategy_library_multifamily.parquet", compression="snappy")
    print("Saved strategy library\n")
    print("Cross-validating lambda (purged blocks; maximize correlation) ...")
    lam0 = cv_lambda_on_corr(R, y, CFG.LAMBDA_GRID, CFG.CV_FOLDS, end_date=CFG.LAMBDA_FIX_DATE)
    print("\n Fitting quarterly betas (positive, L1=1, 5% cap + renorm) ...")
    B = fit_quarterly_betas(R, y, lam0, fix_after=CFG.LAMBDA_FIX_DATE)
    B.to_parquet(CFG.OUT_DIR / "weights_daily.parquet", compression="snappy")
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
    gate = live_mass_under_cap[live_mass_under_cap >= gate_thresh].index.min()
    if pd.isna(gate):
        gate = live_mass_under_cap[live_mass_under_cap > 0].index.min()
    if pd.notna(gate):
        print(f"Gating start at {gate.date()} (live L1 ≥ {int(gate_thresh*100)}%)")
        R_eff = R_eff.loc[gate:]
        W_eff_adj = W_eff_adj.loc[gate:]
        y_eff = y_eff.loc[gate:]
        if CFG.SOFT_START_DAYS > 0:
            end_ix = min(CFG.SOFT_START_DAYS, len(W_eff_adj))
            ramp = np.linspace(1/(end_ix), 1.0, end_ix)
            W_eff_adj.iloc[:end_ix,:] = (W_eff_adj.iloc[:end_ix,:].T * ramp).T
        W_eff_adj.to_parquet(CFG.OUT_DIR / "weights_daily_availability_adjusted_gated.parquet", compression="snappy")
    else:
        raise RuntimeError("No date with live mass > 0 — check strategy library build.")
    if not (isinstance(W_eff_adj.columns, pd.MultiIndex) and "lookback" in W_eff_adj.columns.names):
        if isinstance(W_eff_adj.columns, pd.MultiIndex) and W_eff_adj.columns.nlevels == 2:
            W_eff_adj.columns = W_eff_adj.columns.set_names(["contract", "lookback"])
        else:
            raise ValueError("W_eff_adj must have a MultiIndex with a 'lookback' level.")
    weight_by_lb_daily_eff = W_eff_adj.groupby(level="lookback", axis=1).sum()
    print("\n Generating replication + final overlay (t-1) ...")
    rep, scale, turnover_w, turnover_s = generate_replication(R_eff, W_eff_adj, final_target=CFG.FINAL_VOL_TARGET, cost_bps=CFG.REPL_TRADE_COST_BPS, overlay_cost_bps=CFG.OVERLAY_TRADE_COST_BPS)
    rep.to_csv(CFG.OUT_DIR / "replication_returns_net.csv", float_format="%.8f")
    scale.to_csv(CFG.OUT_DIR / "scale_factor.csv", float_format="%.6f")
    turnover_w.to_csv(CFG.OUT_DIR / "turnover_series_weights.csv", float_format="%.6f")
    turnover_s.to_csv(CFG.OUT_DIR / "turnover_series_overlay.csv", float_format="%.6f")
    print(" Saved replication returns (net), overlay scale, and turnover series\n")
    perf_start = str(R_eff.index[0].date())
    print(" Performance (from gated start) ...")
    perf = performance_table(rep, y_eff, start=perf_start)
    print(perf.to_string())
    make_rev26h_style_figures_process(rep=rep, y=y_eff, beta_daily=W_eff_adj, weight_by_lb_daily=weight_by_lb_daily_eff, turnover_portfolio=turnover_w)
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
        sr = (r.mean()/r.std()) * np.sqrt(252) if r.std() > 0 else np.nan
        vol = r.std() * np.sqrt(252)
        return corr, te, sr, vol
    is_corr, is_te, is_sr, is_vol = brief_stats(rep_is, y_is)
    oos_corr, oos_te, oos_sr, oos_vol = brief_stats(rep_oos, y_oos)
    print(f"IS  (<= {oos_start.date()-pd.Timedelta(days=1)}): corr={is_corr:.3f} | TE={is_te:.2%} | SR={is_sr:.2f} | vol={is_vol:.2%}")
    print(f"OOS (>= {oos_start.date()}): corr={oos_corr:.3f} | TE={oos_te:.2%} | SR={oos_sr:.2f} | vol={oos_vol:.2%}")
    pd.DataFrame({"segment": ["IS","OOS"], "corr": [is_corr, oos_corr], "TE": [is_te, oos_te], "SR": [is_sr, oos_sr], "vol": [is_vol, oos_vol]}).to_csv(CFG.OUT_DIR / "oos_split_process_v6.csv", index=False, float_format="%.6f")
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
    rep_oos_frozen_scaled_net.to_csv(CFG.OUT_DIR / "replication_returns_frozen_oos_net.csv", float_format="%.8f")
    bench_oos = y.reindex(rep_oos_frozen_scaled_net.index)
    fo_corr_s = float(rep_oos_frozen_scaled_net.corr(bench_oos))
    fo_te_s = float((rep_oos_frozen_scaled_net - bench_oos).std() * np.sqrt(252))
    fo_sr_s = float(rep_oos_frozen_scaled_net.mean() / (rep_oos_frozen_scaled_net.std() + 1e-12) * np.sqrt(252))
    fo_vol_s = float(rep_oos_frozen_scaled_net.std() * np.sqrt(252))
    print(f"Frozen OOS (overlay-scaled, net): corr={fo_corr_s:.3f} | TE={fo_te_s:.2%} | SR={fo_sr_s:.2f} | vol={fo_vol_s:.2%}")
    plot_frozen_vs_live_oos(rep.loc[rep_oos_frozen_scaled_net.index], rep_oos_frozen_scaled_net, bench_oos)
    cfg_dump = {
        "LOOKBACKS": list(map(int, CFG.LOOKBACKS)),
        "KEEP_BREAKOUT": list(map(int, CFG.KEEP_BREAKOUT)),
        "KEEP_SKEWABS": list(map(int, CFG.KEEP_SKEWABS)),
        "KEEP_ACCEL": list(map(int, CFG.KEEP_ACCEL)),
        "EWMA_SPAN_VOL": CFG.EWMA_SPAN_VOL,
        "Z_CAP": CFG.Z_CAP,
        "PRE_SCALE_TARGET": CFG.PRE_SCALE_TARGET,
        "PRE_SCALE_CAP": CFG.PRE_SCALE_CAP,
        "PRE_SCALE_FLOOR": CFG.PRE_SCALE_FLOOR,
        "FINAL_VOL_TARGET": CFG.FINAL_VOL_TARGET,
        "FINAL_SCALE_CAP": CFG.FINAL_SCALE_CAP,
        "FINAL_SCALE_FLOOR": CFG.FINAL_SCALE_FLOOR,
        "FINAL_SCALE_WINSOR": CFG.FINAL_SCALE_WINSOR,
        "LAMBDA_GRID": [float(x) for x in CFG.LAMBDA_GRID],
        "LAMBDA_FIX_DATE": str(CFG.LAMBDA_FIX_DATE.date()),
        "CV_FOLDS": CFG.CV_FOLDS,
        "PURGE_GAP_DAYS": CFG.PURGE_GAP_DAYS,
        "MAX_SINGLE_WEIGHT": CFG.MAX_SINGLE_WEIGHT,
        "WEIGHT_EMA_SPAN": CFG.WEIGHT_EMA_SPAN,
        "FORCE_REBUILD_XLARGE": CFG.FORCE_REBUILD_XLARGE,
        "REPL_TRADE_COST_BPS": CFG.REPL_TRADE_COST_BPS,
        "OVERLAY_TRADE_COST_BPS": CFG.OVERLAY_TRADE_COST_BPS,
        "SOFT_START_DAYS": CFG.SOFT_START_DAYS,
        "OOS_START": str(CFG.OOS_START.date()),
        "BURN_IN_DAYS": CFG.BURN_IN_DAYS,
        "USE_CARRY": CFG.USE_CARRY,
        "KEEP_CARRY": list(map(int, CFG.KEEP_CARRY)),
        "BUILD_CARRY_INLINE": CFG.BUILD_CARRY_INLINE,
        "CARRY_FORCE_REBUILD": CFG.CARRY_FORCE_REBUILD,
        "CARRY_SIGNALS_DIR": str(CFG.CARRY_SIGNALS_DIR),
        "FUT_PRICE_DIR": str(CFG.FUT_PRICE_DIR),
        "FUT_ROLL_DIR": str(CFG.FUT_ROLL_DIR),
        "FUT_CFG_DIR": str(CFG.FUT_CFG_DIR),
        "XLARGE_PATH": str(CFG.XLARGE_PATH),
        "OUT_DIR": str(CFG.OUT_DIR),
        "DATA_DIR": str(DATA_DIR),
        "FIGURES_DIR": str(FIGURES_DIR)
    }
    (CFG.OUT_DIR / "config.json").write_text(json.dumps(cfg_dump, indent=2))
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
    print(f"Breakout LBs: {CFG.KEEP_BREAKOUT} | SkewAbs LBs: {CFG.KEEP_SKEWABS} | Accel LBs: {CFG.KEEP_ACCEL}")
    print(f"Carry: use={CFG.USE_CARRY} | build_inline={CFG.BUILD_CARRY_INLINE} | spans={CFG.KEEP_CARRY}\n")
    main()

if __name__ == "__main__":
    run()
