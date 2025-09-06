import os
import numpy as np
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from src import config


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _plot_heatmap(corr_matrix: pd.DataFrame, title: str, filename: str) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
        vmin=-1, vmax=1, cbar=True, linewidths=0.5,
        square=True, annot_kws={"fontsize": 10}, ax=ax
    )
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    plt.xticks(fontsize=11, rotation=0, ha='center')
    plt.yticks(fontsize=11, rotation=0)
    ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    ax.set_title(title, fontsize=18, pad=40)
    plt.tight_layout()
    _ensure_dir(config.FIGURES_DIR)
    output_path = os.path.join(config.FIGURES_DIR, filename)
    plt.savefig(output_path, format='png', dpi=300)
    logging.info(f"Heatmap saved to: {output_path}")
    plt.show()


def _read_series_csv(path: str, preferred_names: list[str]) -> pd.Series:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    for col in preferred_names:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce')
            return s.dropna().sort_index()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 1:
        s = df[num_cols[0]]
        return s.dropna().sort_index()
    raise ValueError(
        f"Could not identify a single returns column in {path}. "
        f"Tried {preferred_names}; numeric columns found: {num_cols}"
    )


def _read_monthly_returns_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(how='all', axis=1).sort_index()
    if df.empty:
        raise ValueError(f"No usable numeric columns found in {path}.")
    return df


def run():
    logging.info("--- Starting Comprehensive Correlation Analysis (All Assets) [V3: CSV-only] ---")
    data_dir = config.DATA_DIR
    figures_dir = config.FIGURES_DIR
    lseg_monthly_csv_path = os.path.join(data_dir, 'LSEG_data', 'multi_asset_monthly_returns.csv')
    btop_csv_path = os.path.join(data_dir, 'btop50_monthly_returns.csv')
    usbonds_csv_path = os.path.join(data_dir, 'us_bonds_monthly_returns.csv')
    missing = [p for p in [lseg_monthly_csv_path, btop_csv_path, usbonds_csv_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required CSV(s):\n  - " + "\n  - ".join(missing) +
            "\nRun your V2.1 script once (with LSEG access) to generate these files."
        )
    cta_returns = _read_series_csv(btop_csv_path, preferred_names=['CTA (BTOP50)', 'BTOP50'])
    us_bond_returns = _read_series_csv(usbonds_csv_path, preferred_names=['US Bonds'])
    lseg_monthly = _read_monthly_returns_table(lseg_monthly_csv_path)
    all_returns = pd.concat([cta_returns.rename('CTA (BTOP50)'), us_bond_returns.rename('US Bonds'), lseg_monthly], axis=1)
    final_df = all_returns.dropna().sort_index()
    as_of = getattr(config, 'CORR_AS_OF', None)
    if as_of:
        as_of_ts = pd.Timestamp(as_of)
        final_df = final_df.loc[:as_of_ts]
    if final_df.empty:
        logging.warning("CRITICAL: No overlapping data found for the complete set of assets. Cannot generate heatmaps.")
        return
    desired_order = [
        'CTA (BTOP50)', 'Global Equities', 'US Equities',
        'Global Bonds', 'US Bonds', 'Commodities', 'US Dollar'
    ]
    final_df = final_df[[c for c in desired_order if c in final_df.columns]]
    analysis_period = f"{final_df.index.min().strftime('%b %Y')} to {final_df.index.max().strftime('%b %Y')}"
    logging.info(f"Analysis period for all assets: {analysis_period}")
    static_corr_matrix = final_df.corr()
    static_title = f'Static Full-Period Correlation (All Assets)\n({analysis_period})'
    _plot_heatmap(static_corr_matrix, static_title, 'static_corr_heatmap_all_assets.png')
    window = int(getattr(config, 'CORR_WINDOW_MONTHS', 12))
    asset_names = final_df.columns
    avg_rolling_corr_matrix = pd.DataFrame(index=asset_names, columns=asset_names, dtype=float)
    for a1, a2 in combinations(asset_names, 2):
        rolling_corr = final_df[a1].rolling(window=window).corr(final_df[a2])
        mean_corr = float(rolling_corr.mean())
        avg_rolling_corr_matrix.loc[a1, a2] = mean_corr
        avg_rolling_corr_matrix.loc[a2, a1] = mean_corr
    for a in asset_names:
        avg_rolling_corr_matrix.loc[a, a] = 1.0
    rolling_title = f'Average of {window}-Month Rolling Correlation (All Assets)\n({analysis_period})'
    _plot_heatmap(avg_rolling_corr_matrix, rolling_title, 'avg_rolling_corr_heatmap_all_assets.png')
    try:
        out_dir = os.path.join(data_dir, "outputs")
        _ensure_dir(out_dir)
        static_corr_matrix.to_csv(os.path.join(out_dir, "corr_static_full_period.csv"))
        avg_rolling_corr_matrix.to_csv(os.path.join(out_dir, f"corr_avg_rolling_{window}m.csv"))
        logging.info(f"Correlation matrices saved to: {out_dir}")
    except Exception as e:
        logging.warning(f"Could not save correlation matrices CSVs (non-fatal): {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run()
