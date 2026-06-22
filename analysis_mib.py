import numpy as np
import tensorflow as tf

from data_loader_mib import build_dataset_mib
from preprocessing_mib import (compute_returns, drop_artifact_stocks,
                               value_weighted_index, rolling_windows,
                               live_columns, standardize_split)
from autoencoder_mib import (build_autoencoder, train_autoencoder,
                             reconstruction_error_per_stock)
from selector_mib import select_stocks
from optimizer_mib import optimize_weights

N_LIST = [5, 10, 15, 20, 25, 30]      # i valori di n testati: confronto a parita' di n
MAX_WEIGHT = 0.30
SEED = 42


# ---------------------------------------------------------------------------
# Titoli ordinati per capitalizzazione media sul train
# ---------------------------------------------------------------------------
def market_value_rank(prices, shares, cols, train_idx):
    cap = shares[cols].values * prices.loc[train_idx, cols].mean(axis=0).values  # cap = azioni x prezzo medio sul train (array 1D)
    return np.argsort(cap)[::-1]      # indici dal piu' grande al piu' piccolo  ([::-1] inverte)


# ---------------------------------------------------------------------------
# Tracking error out-of-sample per una selezione
# ---------------------------------------------------------------------------
def ate_oos(R, index_ret, window, sel_idx):
    """Ottimizza i pesi sul train, misura il tracking error RMS sul test."""
    R_tr = R.loc[window["train"]].values[:, sel_idx]   # rendimenti titoli scelti, righe TRAIN -> matrice (T_tr x n)
    R_te = R.loc[window["test"]].values[:, sel_idx]    # stessi titoli, righe TEST -> matrice (T_te x n)
    i_tr = index_ret.loc[window["train"]].values       # indice sul train, vettore (T_tr,)
    i_te = index_ret.loc[window["test"]].values        # indice sul test,  vettore (T_te,)
    opt = optimize_weights(R_tr, i_tr, max_weight=MAX_WEIGHT)  # pesi ottimizzati solo sul train
    track = i_te - R_te @ opt["weights"]               # errore di tracking sul test coi pesi congelati
    return float(np.sqrt(np.mean(track ** 2)))         # RMSE out-of-sample


# ---------------------------------------------------------------------------
# Confronto su tutte le finestre e tutti gli n
# ---------------------------------------------------------------------------
def run_comparison(prices, shares):
    tf.random.set_seed(SEED)          # semi fissi -> esecuzione riproducibile
    np.random.seed(SEED)

    returns = compute_returns(prices)
    index_ret = value_weighted_index(prices, shares)   # Series: indice target
    windows = rolling_windows(returns.index)           # lista di finestre (ognuna un dict train/val/test)
    print(f"Finestre: {len(windows)}\n")

    ate_ae = {n: [] for n in N_LIST}  # dict: per ogni n, lista degli errori AE finestra per finestra
    ate_mv = {n: [] for n in N_LIST}  # idem per il market-value
    corr_cap = []                     # correlazione errore<->dimensione, una per finestra

    for k, window in enumerate(windows):               # enumerate dà anche il contatore k (numero finestra)
        cols = live_columns(returns, window)           # ticker vivi tutta questa finestra
        R = returns[cols]
        Xtr, Xva, Xte, _ = standardize_split(R, window)  # z-score con statistiche del solo train

        model = build_autoencoder(Xtr.shape[1])          # un autoencoder allenato UNA volta per finestra
        train_autoencoder(model, Xtr, Xva)
        err = reconstruction_error_per_stock(model, Xtr) # array 1D: errore per titolo

        order_mv = market_value_rank(prices, shares, cols, window["train"])   # classifica per capitalizzazione
        cap = shares[cols].values * prices.loc[window["train"], cols].mean(axis=0).values  # cap di ogni titolo (per la correlazione)
        corr_cap.append(float(np.corrcoef(err, np.log(cap))[0, 1]))  # corr tra errore AE e log-cap; [0,1] = casella fuori diagonale della matrice 2x2

        for n in N_LIST:                                 # a parita' di finestra, confronto le due selezioni a ogni n
            ae_idx = select_stocks(err, n_select=n)["selected_idx"]  # selezione autoencoder 80/20
            mv_idx = np.sort(order_mv[:n])               # i primi n per cap; sort solo per ordinarne gli indici
            ate_ae[n].append(ate_oos(R, index_ret, window, ae_idx))  # errore OOS della selezione AE
            ate_mv[n].append(ate_oos(R, index_ret, window, mv_idx))  # errore OOS del baseline

        print(f"  finestra #{k:>2}: {len(cols)} titoli vivi | "
              f"corr(err,log-cap)={corr_cap[-1]:+.2f}")  # corr_cap[-1] = l'ultima aggiunta, cioe' questa finestra

    return ate_ae, ate_mv, corr_cap


# ---------------------------------------------------------------------------
# Esecuzione diretta: tabella + Figura 10
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")             # backend senza finestra grafica: salva solo su file
    import matplotlib.pyplot as plt

    ds = build_dataset_mib("data_mib")
    prices = drop_artifact_stocks(ds["prices"])
    shares = ds["shares"][prices.columns]              # riallineo le azioni ai titoli rimasti
    print(f"{prices.shape[1]} titoli, {prices.shape[0]} giorni\n")  # shape = (righe=giorni, colonne=titoli)

    ate_ae, ate_mv, corr_cap = run_comparison(prices, shares)

    print("\n=== ATE out-of-sample medio (bps/giorno): AE vs market-value ===")
    for n in N_LIST:
        ae = np.mean(ate_ae[n]) * 1e4  # media sulle finestre, x1e4 = da frazione a basis points
        mv = np.mean(ate_mv[n]) * 1e4
        print(f"  n={n:>2}: AE {ae:5.1f}  |  market-value {mv:5.1f}  |  AE/MV {ae / mv:.2f}")  # AE/MV>1 = il baseline vince
    print(f"\ncorr(errore, log-cap) media sulle finestre: "
          f"{np.mean(corr_cap):+.3f}  (atteso ~0 se l'AE non vede la cap)")

    # --- Figura 10: due curve dell'ATE OOS in funzione di n ---
    ae_curve = [np.mean(ate_ae[n]) * 1e4 for n in N_LIST]  # list comprehension: errore medio AE per ogni n
    mv_curve = [np.mean(ate_mv[n]) * 1e4 for n in N_LIST]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(N_LIST, ae_curve, "o-", color="seagreen", label="Autoencoder + 80/20")   
    ax.plot(N_LIST, mv_curve, "s--", color="crimson", label="Market-value ranking")
    ax.set_xlabel("n stocks kept")
    ax.set_ylabel("Out-of-sample tracking error (bps/day)")
    ax.set_title("FTSE MIB daily: OOS replica, lower = better")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figura10_mib.png", dpi=120)
    print("\nGrafico salvato in: figura10_mib.png")