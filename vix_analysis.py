import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

def run_vix_analysis(start_date="2010-01-01", end_date="2025-10-10"):
    # Step 1: Download forex pairs
    pairs = [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X', 'NZDUSD=X',
        'EURGBP=X', 'EURJPY=X', 'EURAUD=X', 'EURCAD=X', 'EURCHF=X', 'EURNZD=X',
        'GBPJPY=X', 'GBPAUD=X', 'GBPCAD=X', 'GBPCHF=X', 'GBPNZD=X',
        'AUDJPY=X', 'AUDNZD=X', 'AUDCAD=X', 'AUDCHF=X',
        'CADJPY=X', 'CHFJPY=X', 'NZDJPY=X', 'NZDCAD=X', 'NZDCHF=X'
    ]

    data = yf.download(pairs, start=start_date, end=end_date)['Close']
    returns = data.pct_change().dropna()

    vix_data = yf.download("^VIX", start=start_date, end=end_date)['Close']
    vix_returns = vix_data.pct_change().dropna()

    common_index = returns.index.intersection(vix_returns.index)
    forex_returns_aligned = returns.loc[common_index]
    vix_returns_aligned = vix_returns.loc[common_index]

    vix_sens = []
    for pair in forex_returns_aligned.columns:
        y = forex_returns_aligned[pair]
        X = sm.add_constant(vix_returns_aligned)
        model = sm.OLS(y, X, missing='drop').fit()

        ci_lower, ci_upper = model.conf_int().loc['^VIX']
        vix_sens.append({
            "Pair": pair,
            "VIX_Beta": model.params['^VIX'],
            "p_value": model.pvalues['^VIX'],
            "Lower_CI": ci_lower,
            "Upper_CI": ci_upper
        })

    vix_sens_df = pd.DataFrame(vix_sens).sort_values("VIX_Beta")

    # Step 5: Plot
    plt.figure(figsize=(14,6))
    bars = plt.bar(vix_sens_df['Pair'], vix_sens_df['VIX_Beta'],
                   color='skyblue', edgecolor='black', yerr=[
                       vix_sens_df['VIX_Beta'] - vix_sens_df['Lower_CI'],
                       vix_sens_df['Upper_CI'] - vix_sens_df['VIX_Beta']
                   ],
                   capsize=3)

    for i, row in enumerate(vix_sens_df.itertuples()):
        if row.p_value < 0.05:
            bars[i].set_color('orange')

    plt.axhline(0, color='black', linewidth=1)
    plt.xticks(rotation=90)
    plt.ylabel("Beta to VIX")
    plt.title("Forex Pairs Sensitivity to VIX (Orange = Statistically Significant)")
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", "vix_sensitivity.png")
    plt.savefig(plot_path)
    plt.close()

    return vix_sens_df.to_dict(orient="records"), plot_path
