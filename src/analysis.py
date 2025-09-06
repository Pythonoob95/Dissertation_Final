from __future__ import annotations

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import seaborn as sns

from src import config

try:
    import lseg.data as rd
except Exception:
    rd = None


def _save_plot(figure, filename):
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    path = os.path.join(config.FIGURES_DIR, filename)
    figure.savefig(path, format='png', dpi=300, bbox_inches='tight')
    logging.info(f"Plot saved to {path}")


def _calculate_and_display_metrics(returns_series: pd.Series, name="Portfolio"):
    returns_series = returns_series.dropna()
    n_months = len(returns_series)
    if n_months == 0:
        logging.warning(f"No data available to calculate metrics for {name}.")
        return

    cumulative_return = (1 + returns_series).prod() - 1
    annualized_return = (1 + cumulative_return) ** (12 / n_months) - 1
    annualized_volatility = returns_series.std(ddof=0) * np.sqrt(12)
    sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility != 0 else 0.0

    wealth_index = (1 + returns_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    max_drawdown = drawdowns.min()
    calmar_ratio = (annualized_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    downside = returns_series[returns_series < 0]
    downside_deviation = downside.std(ddof=0) * np.sqrt(12)
    sortino_ratio = (annualized_return / downside_deviation) if downside_deviation != 0 else 0.0

    best_month = returns_series.max()
    worst_month = returns_series.min()
    pct_positive_months = (returns_series > 0).mean()

    print("\n" + "=" * 55)
    print(f" Performance Metrics for: {name}")
    print(f" Analysis Period: {returns_series.index.min():%b %Y} to {returns_series.index.max():%b %Y}")
    print("=" * 55)
    print(f"{'Cumulative Return:':<35} {cumulative_return:>15.2%}")
    print(f"{'Annualized Return (Geometric):':<35} {annualized_return:>15.2%}")
    print(f"{'Annualized Volatility (Std. Dev.):':<35} {annualized_volatility:>15.2%}")
    print("-" * 55)
    print(f"{'Sharpe Ratio (Rf=0%):':<35} {sharpe_ratio:>15.2f}")
    print(f"{'Sortino Ratio:':<35} {sortino_ratio:>15.2f}")
    print(f"{'Calmar Ratio:':<35} {calmar_ratio:>15.2f}")
    print(f"{'Maximum Drawdown:':<35} {max_drawdown:>15.2%}")
    print("-" * 55)
    print(f"{'Best Month:':<35} {best_month:>15.2%}")
    print(f"{'Worst Month:':<35} {worst_month:>15.2%}")
    print(f"{'Percentage of Positive Months:':<35} {pct_positive_months:>15.2%}")
    print("=" * 55 + "\n")


def _simulate_rebalanced_portfolio(monthly_returns: pd.DataFrame, weights: dict, name: str) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    missing = [c for c in w.index if c not in monthly_returns.columns]
    if missing:
        raise KeyError(f"Missing return columns for: {missing}")
    r = monthly_returns[w.index].mul(w, axis=1).sum(axis=1)
    r.name = name
    return r


def _plot_btop50_standalone(btop_returns: pd.Series):
    logging.info("Generating Chart 1: BTOP50 Standalone Long-Term Performance...")
    wealth_index = (1 + btop_returns).cumprod()
    start_date = wealth_index.index.min() - pd.DateOffset(months=1)
    wealth_index = pd.concat([pd.Series([1.0], index=[start_date]), wealth_index])

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.plot(wealth_index.index, wealth_index, color=config.BTOP50_COLOR, lw=2.5, label='BTOP50 Index Growth')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('$%.2f'))

    colors = plt.cm.get_cmap('Paired', len(config.CRISIS_PERIODS))
    legend_patches = []
    for i, (name, dates) in enumerate(config.CRISIS_PERIODS.items()):
        ax.axvspan(pd.to_datetime(dates[0]), pd.to_datetime(dates[1]), color=colors(i), alpha=0.30)
        legend_patches.append(mpatches.Patch(color=colors(i), alpha=0.40, label=name))

    ax.set_title('BTOP50 Index Performance Through Financial Crises', fontsize=20, pad=20)
    ax.set_ylabel('Growth of $1 (Log Scale)', fontsize=14)
    ax.set_xlabel('Year', fontsize=14)
    line_legend = ax.legend(loc='upper left', fontsize=12)
    ax.add_artist(line_legend)
    ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=True, ncol=len(config.CRISIS_PERIODS))

    _save_plot(fig, '01_btop50_long_term_performance.png')
    plt.show()


def _plot_portfolio_comparison(df_returns: pd.DataFrame):
    logging.info("Generating Chart 2: Cumulative Portfolio Comparison ...")
    fig, ax = plt.subplots(figsize=(18, 9))
    sns.set_style("whitegrid")

    enhanced = df_returns[config.ENHANCED_PORTFOLIO_NAME].dropna()
    dd = pd.DataFrame({'Wealth': (1 + enhanced).cumprod()})
    dd['Peak'] = dd['Wealth'].cummax()
    dd['Drawdown_pct'] = (dd['Wealth'] - dd['Peak']) / dd['Peak']
    mask = dd['Drawdown_pct'] < -0.10
    starts = dd.index[mask & ~mask.shift(1, fill_value=False)]
    ends = dd.index[~mask & mask.shift(1, fill_value=False)]
    if len(ends) < len(starts):
        ends = ends.append(pd.Index([dd.index[-1]]))
    for s, e in zip(starts, ends):
        ax.axvspan(s, e, color='grey', alpha=0.30)

    for col in df_returns.columns:
        color = (config.TRADITIONAL_PORTFOLIO_COLOR
                 if col == config.TRADITIONAL_PORTFOLIO_NAME
                 else config.ENHANCED_PORTFOLIO_COLOR)
        ((1 + df_returns[col]).cumprod()).plot(ax=ax, label=col, color=color, lw=2.5)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('$%.2f'))
    ax.set_title(f'Portfolio Comparison with {config.ENHANCED_PORTFOLIO_NAME}', fontsize=20, pad=20)
    ax.set_ylabel('Growth of $1 (Log Scale)', fontsize=14)
    ax.set_xlabel('Year', fontsize=14)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    _save_plot(fig, '02_cumulative_portfolio_comparison.png')
    plt.show()


def _plot_crisis_bar_chart(df_combined: pd.DataFrame):
    logging.info("Generating Chart 3: Crisis Period Bar Chart Comparison...")
    rows = []
    for name, dates in config.CRISIS_PERIODS.items():
        sub = df_combined.loc[pd.to_datetime(dates[0]):pd.to_datetime(dates[1])]
        if sub.empty:
            continue
        total = (1 + sub).prod() - 1
        start_str = pd.to_datetime(dates[0]).strftime('%b %Y')
        end_str = pd.to_datetime(dates[1]).strftime('%b %Y')
        total['Crisis'] = f"{name}\n({start_str} - {end_str})"
        rows.append(total)

    results_df = pd.DataFrame(rows).set_index('Crisis')
    fig, ax = plt.subplots(figsize=(20, 10))
    results_df.plot(kind='bar', ax=ax, color=[config.BTOP50_COLOR, config.TRADITIONAL_PORTFOLIO_COLOR], width=0.8)

    ax.set_title(f'Total Return During Crisis Periods: BTOP50 vs. {config.TRADITIONAL_PORTFOLIO_NAME}', fontsize=22, pad=20)
    ax.set_ylabel('Total Return', fontsize=14)
    ax.set_xlabel('')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.axhline(0, color='black', linewidth=0.8)
    ax.legend(title='Asset', fontsize=12)

    plt.setp(ax.get_xticklabels(), rotation=35, ha='right', fontsize=9)
    fig.subplots_adjust(bottom=0.22)

    _save_plot(fig, '03_crisis_period_bar_chart.png')
    plt.show()


def _load_btop50_monthly(path: str) -> pd.Series:
    logging.info("--- Step 1: Processing BTOP50 data ---")
    df_btop = pd.read_excel(path, sheet_name='Sheet1', header=1, index_col=0, nrows=39)
    df_btop = df_btop.iloc[:, :12]
    df_btop.index = df_btop.index.astype(int)
    df_btop.index.name = 'Year'

    btop_long = df_btop.reset_index().melt(id_vars='Year', var_name='Month', value_name='ROR')
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    btop_long['Month_Num'] = btop_long['Month'].map(month_map)
    btop_long['Date'] = pd.to_datetime(
        btop_long['Year'].astype(str) + '-' + btop_long['Month_Num'].astype(str)
    ) + pd.offsets.MonthEnd(0)

    btop_returns = btop_long.set_index('Date')['ROR'].dropna().sort_index()
    btop_returns.name = 'BTOP50'
    logging.info("BTOP50 data processed successfully.")
    return btop_returns


def _get_lseg_prices(force_refresh: bool | None = None, offline_only: bool | None = None) -> pd.DataFrame:
    cache_path = getattr(
        config,
        "LSEG_EQ_BOND_CSV",
        os.path.join(getattr(config, "LSEG_AUDIT_DIR", os.path.join(config.DATA_DIR, "LSEG_data")),
                     "equity_bond_daily_prices.csv")
    )
    force_refresh = getattr(config, "LSEG_FORCE_REFRESH", False) if force_refresh is None else force_refresh
    offline_only = getattr(config, "LSEG_OFFLINE_ONLY", False) if offline_only is None else offline_only

    if os.path.exists(cache_path) and not force_refresh:
        logging.info(f"Loading cached LSEG prices from {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=['Date']).set_index('Date').sort_index()
        df.index.name = 'Date'
        return df

    if offline_only:
        raise FileNotFoundError(
            f"Offline-only is enabled but cache not found at:\n  {cache_path}\n"
            f"Run this once on a machine with LSEG to create the cache."
        )

    if rd is None:
        raise ImportError(
            "lseg.data is not available, so I cannot refresh the dataset.\n"
            "Either install/configure LSEG Data Library and rerun,\n"
            "or set LSEG_OFFLINE_ONLY=1 and make sure the cache CSV exists at:\n"
            f"  {cache_path}"
        )

    logging.info("--- Step 3: Fetching data from LSEG Workspace ---")
    rd.open_session()
    try:
        raw = rd.get_history(
            [config.EQUITY_RIC, config.BOND_RIC],
            start=config.LSEG_FETCH_START_DATE,
            end=pd.Timestamp.today().normalize(),
            interval='daily',
            fields=[config.LSEG_DATA_FIELD]
        )
    finally:
        rd.close_session()
        logging.info("LSEG session closed.")

    price_df = raw.rename(columns={config.EQUITY_RIC: 'Equities', config.BOND_RIC: 'Bonds'}).sort_index()
    price_df.index.name = 'Date'

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    price_df.to_csv(cache_path)
    logging.info(f"LSEG data saved to: {cache_path}")

    return price_df


def run():
    try:
        btop_returns = _load_btop50_monthly(config.BTOP50_INPUT_PATH)
        _calculate_and_display_metrics(btop_returns, name="BTOP50 Index")

        price_df = _get_lseg_prices()
        lseg_monthly_returns = price_df.resample('M').last().pct_change().dropna(how='all')

        trad_rets = _simulate_rebalanced_portfolio(
            lseg_monthly_returns, config.TRADITIONAL_PORTFOLIO_ALLOCATION, name=config.TRADITIONAL_PORTFOLIO_NAME
        ).dropna()

        enhanced_universe = pd.concat(
            [lseg_monthly_returns[['Equities', 'Bonds']], btop_returns.rename('BTOP50')], axis=1
        ).dropna()
        enhanced_rets = _simulate_rebalanced_portfolio(
            enhanced_universe, config.ENHANCED_PORTFOLIO_ALLOCATION, name=config.ENHANCED_PORTFOLIO_NAME
        ).dropna()

        _calculate_and_display_metrics(trad_rets, name=config.TRADITIONAL_PORTFOLIO_NAME)
        _calculate_and_display_metrics(enhanced_rets, name=config.ENHANCED_PORTFOLIO_NAME)

        _plot_btop50_standalone(btop_returns)

        df_for_cum = pd.concat([trad_rets, enhanced_rets], axis=1).dropna()
        _plot_portfolio_comparison(df_for_cum)

        df_for_crises = pd.concat([btop_returns, trad_rets], axis=1).dropna()
        _plot_crisis_bar_chart(df_for_crises)

    except Exception as e:
        logging.error(f"An error occurred during the analysis run: {e}", exc_info=True)
