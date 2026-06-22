import numpy as np
import pandas as pd
import tensorflow as tf

from data_loader_mib import build_dataset_mib
from preprocessing_mib import (compute_returns, drop_artifact_stocks,
                               value_weighted_index, rolling_windows,
                               live_columns, standardize_split)
from autoencoder_mib import (build_autoencoder, train_autoencoder,
                             reconstruction_error_per_stock)
from selector_mib import select_stocks
from optimizer_mib import optimize_weights

N = 20
MAX_WEIGHT = 0.30
SEED = 42


def oos_reduced_returns(R, index_ret, window, sel_idx):
    """Pesi ottimizzati sul train, rendimenti del portafoglio ridotto sul test."""
    R_tr = R.loc[window["train"]].values[:, sel_idx]
    i_tr = index_ret.loc[window["train"]].values
    w = optimize_weights(R_tr, i_tr, max_weight=MAX_WEIGHT)["weights"]
    R_te = R.loc[window["test"]].values[:, sel_idx]
    return R_te @ w


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    ds = build_dataset_mib("data_mib")
    prices = drop_artifact_stocks(ds["prices"])
    shares = ds["shares"][prices.columns]
    returns = compute_returns(prices)
    index_ret = value_weighted_index(prices, shares)
    windows = rolling_windows(returns.index)

    dates = []
    idx_oos, ae_oos, mv_oos = [], [], []

    for k, window in enumerate(windows):
        cols = live_columns(returns, window)
        R = returns[cols]
        Xtr, Xva, Xte, _ = standardize_split(R, window)

        model = build_autoencoder(Xtr.shape[1])
        train_autoencoder(model, Xtr, Xva)
        err = reconstruction_error_per_stock(model, Xtr)

        cap = shares[cols].values * prices.loc[window["train"], cols].mean(axis=0).values
        order_mv = np.argsort(cap)[::-1]

        ae_idx = select_stocks(err, n_select=N)["selected_idx"]
        mv_idx = np.sort(order_mv[:N])

        dates.extend(window["test"])
        idx_oos.extend(index_ret.loc[window["test"]].values)
        ae_oos.extend(oos_reduced_returns(R, index_ret, window, ae_idx))
        mv_oos.extend(oos_reduced_returns(R, index_ret, window, mv_idx))
        print(f"  finestra #{k:>2} fatta")

    df = pd.DataFrame({"indice": idx_oos, "ae": ae_oos, "mv": mv_oos},
                      index=pd.DatetimeIndex(dates)).sort_index()

    # ricchezza cumulata (base 100)
    wealth = (1 + df).cumprod() * 100
    # deriva cumulata (replica - indice), in punti percentuali
    diff_ae = (df["ae"] - df["indice"]).cumsum() * 100
    diff_mv = (df["mv"] - df["indice"]).cumsum() * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(wealth.index, wealth["indice"], color="black", lw=1.5, label="Index (target)")
    ax1.plot(wealth.index, wealth["ae"], color="seagreen", lw=1, label="Reduced - Autoencoder")
    ax1.plot(wealth.index, wealth["mv"], color="crimson", lw=1, label="Reduced - Market-value")
    ax1.set_ylabel("wealth (base 100)")
    ax1.legend()
    ax1.set_title(f"Out-of-sample tracking, n={N} stocks (14 windows stitched)")

    ax2.axhline(0, color="grey", lw=0.8)
    ax2.plot(diff_ae.index, diff_ae, color="seagreen", lw=1, label="Autoencoder")
    ax2.plot(diff_mv.index, diff_mv, color="crimson", lw=1, label="Market-value")
    ax2.set_ylabel("cumulative drift (%)")
    ax2.legend()
    ax2.set_xlabel("test period (OOS)")

    fig.tight_layout()
    fig.savefig("tracking_mib.png", dpi=120)
    print("\nGrafico salvato in: tracking_mib.png")
