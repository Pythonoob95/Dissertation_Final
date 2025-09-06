import os
import sys
import logging
import warnings
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 8)

def apply_plot_style(dark: bool = True):
    if dark:
        bg, fg, grid = '#0E1117', '#E6E6E6', '#30363D'
        base_style = 'dark_background'
        seaborn_style = 'darkgrid'
    else:
        bg, fg, grid = '#FFFFFF', '#111111', '#D0D7DE'
        base_style = 'default'
        seaborn_style = 'whitegrid'
    plt.style.use(base_style)
    factor_cycle = [config.FACTOR_COLORS[k] for k in config.FACTOR_COLORS]
    plt.rcParams.update({
        'figure.facecolor': bg,
        'axes.facecolor': bg,
        'savefig.facecolor': bg,
        'savefig.edgecolor': bg,
        'axes.edgecolor': fg,
        'axes.labelcolor': fg,
        'xtick.color': fg,
        'ytick.color': fg,
        'text.color': fg,
        'axes.grid': True,
        'grid.color': grid,
        'grid.alpha': 0.5,
        'grid.linestyle': '--',
        'lines.linewidth': 2.2,
        'axes.prop_cycle': cycler('color', factor_cycle),
        'legend.frameon': True,
        'legend.facecolor': bg if dark else '#F7F7F7',
        'legend.edgecolor': grid,
        'font.size': 12,
        'figure.dpi': 140,
    })
    sns.set_theme(style=seaborn_style, rc={'axes.facecolor': bg, 'figure.facecolor': bg})

apply_plot_style(dark=True)

class EnhancedModelConfig:
    TARGET_NAMES = ['NEIXCTAT Index']
    EQUITY_BASKET_RICS = ['ESc1', 'FDXc1', 'FFIc1', 'NKc1']
    BOND_BASKET_RICS = ['TYc1', 'FGBLc1', 'FLGc1']
    COMMODITY_BASKET_RICS = ['CLc1', 'GCc1', 'HGCc1', 'S c1']
    FX_BASKET_RICS = ['DXc1', '6Ec1', '6Jc1', '6Bc1']
    REVERSAL_BASKET_RICS = list(set(
        EQUITY_BASKET_RICS + BOND_BASKET_RICS + COMMODITY_BASKET_RICS[:2] + FX_BASKET_RICS
    ))
    YIELD_CURVE_RICS = ['TYc1', 'TUc1']
    VOLATILITY_RIC = 'VXc1'
    FX_CARRY_RICS = ['6Ac1', '6Cc1', '6Ec1', '6Bc1', '6Jc1', '6Sc1']
    FX_CARRY_YIELD_RICS = {
        '6Ac1': 'AU3MT=RR',
        '6Cc1': 'CA3MT=RR',
        '6Ec1': 'DE3MT=RR',
        '6Bc1': 'GB3MT=RR',
        '6Jc1': 'JP3MT=RR',
        '6Sc1': 'CH3MT=RR'
    }
    COMMODITY_CURVE_RICS_MAP = {'CLc1': 'CLc2', 'GCc1': 'GCc2', 'S c1': 'S c2'}
    FACTOR_NAMES = [
        'Trend_Equities',
        'Trend_Commodities',
        'Trend_Bonds',
        'Trend_FX',
        'CS_Mom_Equities',
        'CS_Mom_Commodities',
        'USD_Factor',
    ]
    START_DATE = '2009-01-01'
    DAILY_ROLLING_WINDOW_DAYS = 252
    TREND_LOOKBACK_WINDOWS = [8, 16, 32, 64, 128, 252]
    REVERSAL_LOOKBACK_DAYS = 5
    CARRY_REBALANCE_PERIOD = 'M'
    VOLATILITY_LOOKBACK_DAYS = 60
    CS_MOMENTUM_LOOKBACK_DAYS = 120
    POSITION_CLIP_THRESHOLD = 4.0
    NEWEY_WEST_LAGS = 21
    USE_HAC_ERRORS = True

FACTOR_CACHE_DIR = os.path.join(config.DATA_DIR, 'factor_cache')
LSEG_DIR = os.path.join(config.DATA_DIR, 'LSEG_data')
TARGETS_PRICES_CSV = os.path.join(FACTOR_CACHE_DIR, 'cta_targets_prices.csv')
TARGETS_RETURNS_CSV = os.path.join(FACTOR_CACHE_DIR, 'cta_targets_returns.csv')
LSEG_DAILY_PRICES_CSV = os.path.join(LSEG_DIR, 'all_instruments_daily_prices.csv')
LSEG_RETURNS_YIELDS_CSV = os.path.join(LSEG_DIR, 'all_instruments_returns_and_yields_daily.csv')

def _ensure_dirs():
    for p in [config.DATA_DIR, config.FIGURES_DIR, FACTOR_CACHE_DIR, LSEG_DIR]:
        os.makedirs(p, exist_ok=True)

def _require_file(path: str, help_msg: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required CSV not found: {path}\n{help_msg}")

def construct_cross_sectional_momentum(
    returns: pd.DataFrame,
    basket_rics: List[str],
    lookback: int,
    vol_window: int,
    clip_threshold: float
) -> pd.Series:
    factor_returns = pd.DataFrame(index=returns.index)
    valid_basket_rics = [ric for ric in basket_rics if ric in returns.columns]
    if len(valid_basket_rics) < 2:
        return pd.Series(0, index=returns.index)
    lookback_returns = returns[valid_basket_rics].rolling(window=lookback).sum()
    ranked_returns = lookback_returns.rank(axis=1, ascending=False, method='first')
    num_assets = len(valid_basket_rics)
    num_to_select = max(1, num_assets // 3)
    positions_series = ranked_returns.apply(
        lambda x: np.where(x <= num_to_select, 1, np.where(x > (num_assets - num_to_select), -1, 0)),
        axis=1
    )
    positions = pd.DataFrame(positions_series.to_list(), index=returns.index, columns=valid_basket_rics)
    for ric in valid_basket_rics:
        realized_vol = returns[ric].rolling(window=vol_window).std().shift(1)
        position_raw = positions[ric] / (realized_vol + 1e-10)
        position_clipped = position_raw.clip(-clip_threshold, clip_threshold).shift(1)
        factor_returns[f'{ric}_cs_mom_ret'] = position_clipped * returns[ric]
    final_returns = factor_returns.mean(axis=1)
    num_positions = (positions.abs().sum(axis=1) / 2).replace(0, 1)
    return (final_returns / num_positions).fillna(0)

def construct_trend_factor_vol_scaled(
    returns: pd.DataFrame,
    basket_rics: List[str],
    lookbacks: List[int],
    vol_window: int,
    clip_threshold: float
) -> pd.Series:
    basket_trend_returns = pd.DataFrame(index=returns.index)
    for ric in basket_rics:
        if ric not in returns.columns or returns[ric].isnull().all():
            continue
        realized_vol = returns[ric].rolling(window=vol_window).std().shift(1)
        positions = pd.DataFrame(index=returns.index)
        for window in lookbacks:
            signal = returns[ric].rolling(window=window).sum()
            positions[f'pos_{window}'] = np.sign(signal)
        final_signal = positions.mean(axis=1)
        final_position_raw = final_signal / (realized_vol + 1e-10)
        final_position_clipped = final_position_raw.clip(-clip_threshold, clip_threshold).shift(1)
        basket_trend_returns[f'{ric}_trend_ret'] = final_position_clipped * returns[ric]
    return basket_trend_returns.mean(axis=1)

def construct_reversal_factor_vol_scaled(
    returns: pd.DataFrame,
    basket_rics: List[str],
    lookback: int,
    vol_window: int,
    clip_threshold: float
) -> pd.Series:
    basket_reversal_returns = pd.DataFrame(index=returns.index)
    for ric in basket_rics:
        if ric not in returns.columns or returns[ric].isnull().all():
            continue
        realized_vol = returns[ric].rolling(window=vol_window).std().shift(1)
        signal = -returns[ric].rolling(window=lookback).sum()
        position_raw = np.sign(signal) / (realized_vol + 1e-10)
        position_clipped = position_raw.clip(-clip_threshold, clip_threshold).shift(1)
        basket_reversal_returns[f'{ric}_reversal_ret'] = position_clipped * returns[ric]
    return basket_reversal_returns.mean(axis=1)

def construct_fx_carry_factor(
    returns: pd.DataFrame,
    asset_rics: List[str],
    yield_rics_map: dict,
    rebalance_freq: str
) -> pd.Series:
    valid_yield_rics = [ric for ric in yield_rics_map.values() if ric in returns.columns]
    valid_asset_rics = [ric for ric in asset_rics if ric in returns.columns]
    valid_yield_map = {k: v for k, v in yield_rics_map.items()
                       if k in valid_asset_rics and v in valid_yield_rics}
    if not valid_yield_map or len(valid_asset_rics) < 2:
        return pd.Series(0, index=returns.index)
    yield_data = returns[list(valid_yield_map.values())]
    asset_returns = returns[valid_asset_rics]
    rebalance_dates = yield_data.resample(rebalance_freq).last().index
    positions = pd.DataFrame(0, index=returns.index, columns=valid_asset_rics)
    for i in range(len(rebalance_dates) - 1):
        start_date, end_date = rebalance_dates[i], rebalance_dates[i + 1]
        current_yields = yield_data.asof(start_date)
        if current_yields is None or current_yields.isnull().all():
            continue
        y2a = {v: k for k, v in valid_yield_map.items()}
        ranked_yield_rics = current_yields.dropna().sort_values(ascending=False).index
        ranked_assets = [y2a[y_ric] for y_ric in ranked_yield_rics if y_ric in y2a]
        if len(ranked_assets) < 2:
            continue
        num_to_select = max(1, len(ranked_assets) // 3)
        positions.loc[start_date:end_date, ranked_assets[:num_to_select]] = 1
        positions.loc[start_date:end_date, ranked_assets[-num_to_select:]] = -1
    positions = positions.ffill().shift(1).fillna(0)
    num_positions = positions.abs().sum(axis=1)
    carry_returns = (positions * asset_returns).sum(axis=1)
    return (carry_returns / num_positions.replace(0, 1)).fillna(0)

def construct_commodity_curve_factor(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    curve_rics_map: Dict[str, str]
) -> pd.Series:
    basket_curve_returns = pd.DataFrame(index=prices.index)
    for front_ric, back_ric in curve_rics_map.items():
        if (front_ric not in prices.columns) or (back_ric not in prices.columns) or (front_ric not in returns.columns):
            continue
        signal = np.sign(prices[front_ric] - prices[back_ric])
        position = signal.shift(1)
        basket_curve_returns[f'{front_ric}_curve_ret'] = position * returns[front_ric]
    return basket_curve_returns.mean(axis=1).fillna(0)

def _load_targets_from_csv() -> Tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(TARGETS_PRICES_CSV, parse_dates=['Date'], index_col='Date')
    returns = pd.read_csv(TARGETS_RETURNS_CSV, parse_dates=['Date'], index_col='Date')
    for c in returns.columns:
        returns[c] = pd.to_numeric(returns[c], errors='coerce')
    return prices.sort_index(), returns.sort_index()

def _load_lseg_from_csv() -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_prices = pd.read_csv(LSEG_DAILY_PRICES_CSV, parse_dates=['Date'], index_col='Date')
    for c in raw_prices.columns:
        raw_prices[c] = pd.to_numeric(raw_prices[c], errors='coerce')
    ry = pd.read_csv(LSEG_RETURNS_YIELDS_CSV, parse_dates=['Date'], index_col='Date')
    for c in ry.columns:
        ry[c] = pd.to_numeric(ry[c], errors='coerce')
    return raw_prices.sort_index(), ry.sort_index()

def _apply_asof_clip(df: pd.DataFrame) -> pd.DataFrame:
    asof = getattr(config, 'FACTOR_AS_OF', None)
    if asof:
        return df.loc[:pd.Timestamp(asof)]
    return df

def _construct_all_factors(
    all_returns_and_yields: pd.DataFrame,
    raw_prices: pd.DataFrame,
    model_config: EnhancedModelConfig
) -> pd.DataFrame:
    logging.info('--- Constructing enhanced factor set ---')
    factors = pd.DataFrame(index=all_returns_and_yields.index)
    vol_lookback = model_config.VOLATILITY_LOOKBACK_DAYS
    clip = model_config.POSITION_CLIP_THRESHOLD
    factors['Trend_Equities'] = construct_trend_factor_vol_scaled(
        all_returns_and_yields, model_config.EQUITY_BASKET_RICS,
        model_config.TREND_LOOKBACK_WINDOWS, vol_lookback, clip
    )
    factors['Trend_Bonds'] = construct_trend_factor_vol_scaled(
        all_returns_and_yields, model_config.BOND_BASKET_RICS,
        model_config.TREND_LOOKBACK_WINDOWS, vol_lookback, clip
    )
    factors['Trend_Commodities'] = construct_trend_factor_vol_scaled(
        all_returns_and_yields, model_config.COMMODITY_BASKET_RICS,
        model_config.TREND_LOOKBACK_WINDOWS, vol_lookback, clip
    )
    factors['Trend_FX'] = construct_trend_factor_vol_scaled(
        all_returns_and_yields, model_config.FX_BASKET_RICS,
        model_config.TREND_LOOKBACK_WINDOWS, vol_lookback, clip
    )
    factors['CS_Mom_Equities'] = construct_cross_sectional_momentum(
        all_returns_and_yields, model_config.EQUITY_BASKET_RICS,
        model_config.CS_MOMENTUM_LOOKBACK_DAYS, vol_lookback, clip
    )
    factors['CS_Mom_Commodities'] = construct_cross_sectional_momentum(
        all_returns_and_yields, model_config.COMMODITY_BASKET_RICS,
        model_config.CS_MOMENTUM_LOOKBACK_DAYS, vol_lookback, clip
    )
    if 'DXc1' in all_returns_and_yields.columns:
        factors['USD_Factor'] = all_returns_and_yields['DXc1']
    logging.info('All factors constructed.')
    return factors

def _finalize_dataset(
    target_returns: pd.DataFrame,
    factors: pd.DataFrame,
    model_config: EnhancedModelConfig
) -> Tuple[pd.DataFrame, EnhancedModelConfig]:
    logging.info('--- Merging & cleaning final dataset ---')
    if not hasattr(model_config, 'FACTOR_NAMES') or not model_config.FACTOR_NAMES:
        inferred = sorted([c for c in factors.columns if isinstance(c, str)])
        logging.warning("model_config.FACTOR_NAMES missing; inferring from factors columns: %s", inferred)
        model_config.FACTOR_NAMES = inferred
    combined = pd.concat([target_returns, factors], axis=1)
    final_df = combined[combined.index >= pd.to_datetime(model_config.START_DATE)].astype(np.float64)
    final_df.dropna(subset=model_config.TARGET_NAMES, how='any', inplace=True)
    current_factors = list(set(model_config.FACTOR_NAMES).intersection(final_df.columns))
    if not current_factors:
        raise RuntimeError("No valid factors present after merge.")
    final_df[current_factors] = final_df[current_factors].fillna(0)
    final_df.replace([np.inf, -np.inf], 0, inplace=True)
    stds = final_df[current_factors].std()
    zero_vars = stds[stds < 1e-10].index.tolist()
    if zero_vars:
        logging.warning(f"Removing zero-variance factors: {zero_vars}")
        model_config.FACTOR_NAMES = [f for f in model_config.FACTOR_NAMES if f not in zero_vars]
        current_factors = [f for f in current_factors if f not in zero_vars]
    final_df[current_factors] = (final_df[current_factors] - final_df[current_factors].mean()) / final_df[current_factors].std()
    final_df.replace([np.inf, -np.inf], 0, inplace=True)
    final_df[current_factors] = final_df[current_factors].fillna(0)
    logging.info("Active factors used: %s", sorted(current_factors))
    logging.info(f"Dataset ready: shape {final_df.shape} from {final_df.index.min().date()} to {final_df.index.max().date()}")
    return final_df, model_config

def smooth_data(df: pd.DataFrame, smoothing_window: int) -> pd.DataFrame:
    if smoothing_window > 1:
        logging.info(f'--- Applying {smoothing_window}-day rolling average to smooth data ---')
        return df.rolling(window=smoothing_window, min_periods=1).mean()
    return df

def check_multicollinearity(factor_returns_df: pd.DataFrame, model_config: EnhancedModelConfig):
    logging.info('--- Checking for Factor Multicollinearity ---')
    active = list(set(model_config.FACTOR_NAMES).intersection(factor_returns_df.columns))
    if not active:
        logging.warning('No factors available for correlation matrix.')
        return
    plt.figure(figsize=(22, 20))
    corr = factor_returns_df[active].corr()
    cmap = 'RdBu_r'
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        vmin=-1, vmax=1, center=0,
        square=True,
        linewidths=0.5,
        linecolor=plt.rcParams['grid.color'],
        cbar_kws={'shrink': 0.85, 'ticks': [-1, -0.5, 0, 0.5, 1]},
        annot_kws={'size': 11, 'color': plt.rcParams['text.color']}
    )
    plt.title('Factor Correlation Matrix (Refined Model)', fontsize=20, pad=14)
    plt.tight_layout()
    save_path = os.path.join(config.FIGURES_DIR, 'factor_correlation_matrix_refined.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    plt.show()
    high = 0.70
    hits = []
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            val = corr.iloc[i, j]
            if abs(val) > high:
                hits.append((active[i], active[j], val))
    if hits:
        print("\nHigh correlations detected (|r| > 0.70):")
        for f1, f2, r in hits:
            print(f"  {f1} <-> {f2}: {r:.3f}")

def perform_fullsample_regression_with_newey_west(
    y: pd.Series, X: pd.DataFrame, lag: int = 21
) -> sm.regression.linear_model.RegressionResultsWrapper:
    return sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lag})

def perform_rolling_analysis(
    returns_df: pd.DataFrame, target_name: str, model_config: EnhancedModelConfig
):
    logging.info(f'--- Rolling analysis for target: {target_name} ---')
    active = list(set(model_config.FACTOR_NAMES).intersection(returns_df.columns))
    if not active:
        logging.warning(f'No factors for rolling analysis on {target_name}.')
        return
    smooth = smooth_data(returns_df, smoothing_window=5)
    y = smooth[target_name].fillna(0)
    X = sm.add_constant(smooth[active].fillna(0))
    if len(y) < model_config.DAILY_ROLLING_WINDOW_DAYS:
        logging.warning(f'Insufficient data for window {model_config.DAILY_ROLLING_WINDOW_DAYS}. Len={len(y)}')
        return
    if model_config.USE_HAC_ERRORS:
        lag = model_config.NEWEY_WEST_LAGS
        print(f"\nUsing Newey-West HAC standard errors with {lag} lags")
        full_model = perform_fullsample_regression_with_newey_west(y, X, lag=lag)
    else:
        full_model = sm.OLS(y, X).fit()
    print('\n' + '=' * 80)
    print(f'FULL SAMPLE REGRESSION RESULTS FOR: {target_name}')
    print(f'Regression Type: {"Newey-West HAC" if model_config.USE_HAC_ERRORS else "Standard OLS"}')
    print('=' * 80)
    print(full_model.summary())
    print('=' * 80 + '\n')
    print(f"Durbin-Watson: {sm.stats.durbin_watson(full_model.resid):.4f}")
    print(f"R-squared: {full_model.rsquared:.4f} | Adj R-squared: {full_model.rsquared_adj:.4f}")
    window = model_config.DAILY_ROLLING_WINDOW_DAYS
    rols = RollingOLS(y, X, window=window, min_nobs=int(window * 0.9))
    if model_config.USE_HAC_ERRORS:
        rols_res = rols.fit(cov_type='HAC', cov_kwds={'maxlags': model_config.NEWEY_WEST_LAGS})
    else:
        rols_res = rols.fit()
    rolling_betas = rols_res.params.dropna()
    if 'const' in rolling_betas.columns:
        rolling_betas = rolling_betas.drop(columns=['const'])
    rolling_rsquared = rols_res.rsquared.dropna()
    if not rolling_betas.empty:
        fig, ax = plt.subplots(figsize=(18, 9))
        cols = list(rolling_betas.columns)
        colors = [config.FACTOR_COLORS.get(c, None) for c in cols]
        rolling_betas.plot(ax=ax, linewidth=2.0, color=colors)
        ax.set_title(f'Rolling Exposures: {target_name} ({window}d, '
                     f'{"HAC" if model_config.USE_HAC_ERRORS else "OLS"})', fontsize=20)
        ax.set_ylabel('Beta')
        ax.set_xlabel('Date')
        leg = ax.legend(title='Factors', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        leg.get_frame().set_alpha(0.9)
        leg.get_frame().set_edgecolor(plt.rcParams['grid.color'])
        ax.grid(True, alpha=0.35)
        plt.tight_layout()
        out = os.path.join(config.FIGURES_DIR, f'rolling_betas_{target_name}_{"hac" if model_config.USE_HAC_ERRORS else "ols"}.png')
        plt.savefig(out, dpi=300, bbox_inches='tight',
                    facecolor=plt.gcf().get_facecolor(), edgecolor='none')
        plt.show()
    if not rolling_rsquared.empty:
        avg_r2 = full_model.rsquared
        mean_r2 = rolling_rsquared.mean()
        tss_weights = y.rolling(window).var(ddof=1) * (window - 1)
        w = tss_weights.loc[rolling_rsquared.index].replace([np.inf, -np.inf], np.nan)
        valid = (~rolling_rsquared.isna()) & (~w.isna()) & (w > 0)
        if valid.any():
            w_mean_r2 = np.average(rolling_rsquared[valid], weights=w[valid])
        else:
            w_mean_r2 = np.nan
        fig, ax = plt.subplots(figsize=(18, 9))
        ax.plot(rolling_rsquared.index,
                rolling_rsquared.values,
                linewidth=2.8,
                color='#2E86FF',
                label=fr'Rolling $R^2$ (Mean: {mean_r2:.2f}'
                      + (f' | TSS-wt: {w_mean_r2:.2f}' if not np.isnan(w_mean_r2) else '') + ')')
        ax.axhline(y=avg_r2,
                   color='#FF4C4C',
                   linestyle=(0, (6, 4)),
                   linewidth=2.6,
                   label=fr'Full-sample $R^2$ ({avg_r2:.2f})')
        ax.set_title(f'Model Fit: {target_name} ({window}d, ', fontsize=20)
        ax.set_ylabel(r'$R^2$')
        ax.set_xlabel('Date')
        ax.set_ylim(0, max(1.0, rolling_rsquared.max() * 1.1))
        leg = ax.legend()
        leg.get_frame().set_alpha(0.9)
        leg.get_frame().set_edgecolor(plt.rcParams['grid.color'])
        ax.grid(True, alpha=0.35)
        plt.tight_layout()
        out = os.path.join(
            config.FIGURES_DIR,
            f'rolling_rsquared_{target_name}_{"hac" if model_config.USE_HAC_ERRORS else "ols"}.png'
        )
        plt.savefig(out, dpi=300, bbox_inches='tight',
                    facecolor=plt.gcf().get_facecolor(), edgecolor='none')
        plt.show()
    if model_config.USE_HAC_ERRORS:
        print("\nComparing standard errors (HAC vs. Standard OLS):")
        std_model = sm.OLS(y, X).fit()
        cmp_df = pd.DataFrame({
            'Coefficient': full_model.params,
            'Std Err (HAC)': full_model.bse,
            'Std Err (OLS)': std_model.bse,
            't-stat (HAC)': full_model.tvalues,
            't-stat (OLS)': std_model.tvalues,
            'p-value (HAC)': full_model.pvalues,
            'p-value (OLS)': std_model.pvalues
        })
        cmp_df['SE Ratio'] = cmp_df['Std Err (HAC)'] / cmp_df['Std Err (OLS)']
        print(cmp_df.round(4))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        x_pos = np.arange(len(cmp_df.index)); width = 0.35
        ax1.bar(x_pos - width/2, cmp_df['Std Err (OLS)'], width, label='OLS',
                edgecolor=plt.rcParams['grid.color'])
        ax1.bar(x_pos + width/2, cmp_df['Std Err (HAC)'], width, label='HAC',
                edgecolor=plt.rcParams['grid.color'])
        ax1.set_title('Standard Errors: OLS vs. HAC')
        ax1.legend()
        ax1.grid(True, alpha=0.35)
        ax1.set_xticks(x_pos); ax1.set_xticklabels(cmp_df.index, rotation=45, ha='right')
        ax2.bar(x_pos - width/2, np.abs(cmp_df['t-stat (OLS)']), width, label='OLS',
                edgecolor=plt.rcParams['grid.color'])
        ax2.bar(x_pos + width/2, np.abs(cmp_df['t-stat (HAC)']), width, label='HAC',
                edgecolor=plt.rcParams['grid.color'])
        ax2.axhline(y=1.96, color='#FF5252', linestyle='--', linewidth=1.5, label='5% significance')
        ax2.set_title('|t|-stats: OLS vs. HAC')
        ax2.legend()
        ax2.grid(True, alpha=0.35)
        ax2.set_xticks(x_pos); ax2.set_xticklabels(cmp_df.index, rotation=45, ha='right')
        plt.tight_layout()
        out = os.path.join(config.FIGURES_DIR, f'hac_comparison_{target_name}.png')
        plt.savefig(out, dpi=300, bbox_inches='tight',
                    facecolor=plt.gcf().get_facecolor(), edgecolor='none')
        plt.show()

def analyze_residuals(model, target_name: str):
    residuals = model.resid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    ax1.plot(residuals.index, residuals, linewidth=1.2)
    ax1.axhline(0, color='#FF5252', linestyle='--', linewidth=1.2)
    ax1.set_title('Residuals Over Time')
    ax1.grid(True, alpha=0.35)
    ax2.hist(residuals, bins=50, edgecolor=plt.rcParams['axes.edgecolor'], linewidth=0.6)
    ax2.set_title('Distribution of Residuals')
    ax2.grid(True, alpha=0.35)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot')
    ax3.grid(True, alpha=0.35)
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals.dropna(), lags=40, ax=ax4, alpha=0.05)
    ax4.set_title('Residual ACF')
    ax4.grid(True, alpha=0.35)
    plt.suptitle(f'Residual Analysis for {target_name}', fontsize=16)
    plt.tight_layout()
    out = os.path.join(config.FIGURES_DIR, f'residual_analysis_{target_name}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight',
                facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    plt.show()
    print(f"\nResidual diagnostics for {target_name}:")
    print(f"Jarque-Bera: stat={stats.jarque_bera(residuals)[0]:.4f}, p={stats.jarque_bera(residuals)[1]:.4f}")
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = acorr_ljungbox(residuals.dropna(), lags=10, return_df=True)
    print("Ljung-Box (10 lags):")
    print(lb[['lb_stat', 'lb_pvalue']].tail(1))

def compare_hac_vs_standard(daily_returns_df, model_config):
    print("\n" + "=" * 80)
    print("COMPARING RESULTS: STANDARD OLS vs NEWEY-WEST HAC")
    print("=" * 80)
    for target in model_config.TARGET_NAMES:
        if target not in daily_returns_df:
            continue
        print(f"\nTarget: {target}")
        print("-" * 40)
        smooth = smooth_data(daily_returns_df, smoothing_window=5)
        y = smooth[target].fillna(0)
        active = list(set(model_config.FACTOR_NAMES).intersection(smooth.columns))
        X = sm.add_constant(smooth[active].fillna(0))
        std_model = sm.OLS(y, X).fit()
        hac_model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': model_config.NEWEY_WEST_LAGS})
        rows = []
        for var in X.columns:
            rows.append({
                'Variable': var,
                'Coefficient': std_model.params[var],
                'SE (Standard)': std_model.bse[var],
                'SE (HAC)': hac_model.bse[var],
                'SE Inflation': hac_model.bse[var] / std_model.bse[var],
                't-stat (Standard)': std_model.tvalues[var],
                't-stat (HAC)': hac_model.tvalues[var],
                'p-val (Standard)': std_model.pvalues[var],
                'p-val (HAC)': hac_model.pvalues[var],
                'Sig (Standard)': '***' if std_model.pvalues[var] < 0.01 else '**' if std_model.pvalues[var] < 0.05 else '*' if std_model.pvalues[var] < 0.10 else '',
                'Sig (HAC)': '***' if hac_model.pvalues[var] < 0.01 else '**' if hac_model.pvalues[var] < 0.05 else '*' if hac_model.pvalues[var] < 0.10 else ''
            })
        cmp = pd.DataFrame(rows)
        print(cmp.to_string(index=False))
        changed = cmp[cmp['Sig (Standard)'] != cmp['Sig (HAC)']]['Variable'].tolist()
        if changed:
            print(f"\nSignificance changes after HAC correction: {', '.join(changed)}")

def save_results_to_excel(daily_returns_df, model_config, filename=None):
    if filename is None:
        filename = os.path.join(config.DATA_DIR, 'cta_Factor_analysis_results.xlsx')
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        active = list(set(model_config.FACTOR_NAMES).intersection(daily_returns_df.columns))
        data = daily_returns_df[model_config.TARGET_NAMES + active]
        data.to_excel(writer, sheet_name='Factor_Returns')
        data.corr().to_excel(writer, sheet_name='Correlations')
        data.describe().to_excel(writer, sheet_name='Summary_Stats')
    print(f"\nResults saved to {filename}")

def run():
    _ensure_dirs()
    model_config = EnhancedModelConfig()
    print("=" * 80)
    print("CTA FACTOR ANALYSIS WITH NEWEY-WEST STANDARD ERRORS (V3: CSV-only)")
    print("=" * 80)
    print(f"Using Newey-West HAC errors: {model_config.USE_HAC_ERRORS}  |  Lags: {model_config.NEWEY_WEST_LAGS}")
    print("=" * 80)
    help_msg = "Run the V2.1 pipeline to generate this file first."
    _require_file(TARGETS_PRICES_CSV, help_msg)
    _require_file(TARGETS_RETURNS_CSV, help_msg)
    _require_file(LSEG_DAILY_PRICES_CSV, help_msg)
    _require_file(LSEG_RETURNS_YIELDS_CSV, help_msg)
    raw_prices, all_ry = _load_lseg_from_csv()
    target_prices, target_returns = _load_targets_from_csv()
    raw_prices = _apply_asof_clip(raw_prices)
    all_ry = _apply_asof_clip(all_ry)
    target_returns = _apply_asof_clip(target_returns)
    factors = _construct_all_factors(all_ry, raw_prices, model_config)
    final_df, model_config = _finalize_dataset(target_returns, factors, model_config)
    logging.info('--- Checking for Factor Multicollinearity ---')
    check_multicollinearity(final_df, model_config)
    for tgt in model_config.TARGET_NAMES:
        if tgt in final_df:
            perform_rolling_analysis(final_df, tgt, model_config)
            print("\nPerforming residual analysis...")
            smooth = smooth_data(final_df, smoothing_window=5)
            y = smooth[tgt].fillna(0)
            active = list(set(model_config.FACTOR_NAMES).intersection(smooth.columns))
            X = sm.add_constant(smooth[active].fillna(0))
            model = (perform_fullsample_regression_with_newey_west(y, X, lag=model_config.NEWEY_WEST_LAGS)
                     if model_config.USE_HAC_ERRORS else sm.OLS(y, X).fit())
            analyze_residuals(model, tgt)
    logging.info('--- Analysis Complete ---')
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    active = list(set(model_config.FACTOR_NAMES).intersection(final_df.columns))
    print(final_df[model_config.TARGET_NAMES + active].describe().round(4))
    print("\n" + "=" * 80)
    print("FACTOR CORRELATIONS WITH TARGETS")
    print("=" * 80)
    for tgt in model_config.TARGET_NAMES:
        if tgt in final_df:
            corr = final_df[active].corrwith(final_df[tgt])
            print(f"\nCorrelations with {tgt}:")
            print(corr.sort_values(ascending=False).round(3))
    compare_hac_vs_standard(final_df, model_config)
    save_results_to_excel(final_df, model_config)
    print("\n" + "=" * 80)
    print("FACTOR DECOMPOSITION ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved under: {config.DATA_DIR}")
    print(f"Figures saved under: {config.FIGURES_DIR}")

if __name__ == '__main__':
    run()
