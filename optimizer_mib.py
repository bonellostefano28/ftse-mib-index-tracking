import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Ottimizzazione dei pesi
# ---------------------------------------------------------------------------
def optimize_weights(returns_selected, index_returns, max_weight=0.30):
    """Pesi a minimi quadrati vincolati per replicare l'indice.

    returns_selected : matrice Numpy (T x n) dei rendimenti dei titoli selezionati
    index_returns    : vettore Numpy (T,) dei rendimenti dell'indice da replicare
    max_weight       : peso massimo per singolo titolo (0.30 = 30%)
    """
    n = returns_selected.shape[1] # numero di colonne

    def objective(w): #funzione da minimizzare
        tracking = index_returns - returns_selected @ w # vettore T: err tracking day by day
        return np.mean(tracking ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}] # vincolo di uguaglianza -> np.sum(w) = 1
    bounds = [(0.0, max_weight)] * n  # vincoli per ogni titolo
    w0 = np.full(n, 1.0 / n)          # riempie un vettore di 1/n; partenza -> pesi uguali

    result = minimize(objective, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 500, "ftol": 1e-12})

    w = result.x
    w[w < 1e-6] = 0.0                             # azzera pesi numericamente nulli
    if w.sum() > 0:
        w = w / w.sum()                           # ri-normalizza dopo l'azzeramento

    tracking = index_returns - returns_selected @ w
    ate = float(np.sqrt(np.mean(tracking ** 2)))  # tracking error in-sample
    return {"weights": w, "ate_in_sample": ate, "success": result.success} #Dizionario


# ---------------------------------------------------------------------------
# Rendimenti dell'indice ridotto dato il set di pesi
# ---------------------------------------------------------------------------
def reduced_index_returns(returns_selected, weights):
    
    return returns_selected @ weights # (Txn) x n -> T; 


# ---------------------------------------------------------------------------
# Esecuzione diretta: pipeline completa sulla prima finestra
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tensorflow as tf
    from data_loader_mib import build_dataset_mib
    from preprocessing_mib import (compute_returns, drop_artifact_stocks,
                                   value_weighted_index, rolling_windows,
                                   live_columns, standardize_split)
    from autoencoder_mib import (build_autoencoder, train_autoencoder,
                                 reconstruction_error_per_stock)
    from selector_mib import select_stocks

    ds = build_dataset_mib("data_mib")                  # dizionario: prices, shares, names, index
    prices = drop_artifact_stocks(ds["prices"])         # tolgo i titoli con salti da corporate action
    shares = ds["shares"][prices.columns]               # riallineo le azioni ai soli titoli rimasti
    names = ds["names"]
    returns = compute_returns(prices)
    index_ret = value_weighted_index(prices, shares)    # Series: indice target

    w0 = rolling_windows(returns.index)[0]              # [0] = prima finestra; w0 è un dict train/val/test
    cols = live_columns(returns, w0)                    # lista di ticker vivi tutta la finestra
    Rlive = returns[cols]
    Xtr, Xva, Xte, _ = standardize_split(Rlive, w0)     # z-score con statistiche del solo train; _ scarta lo scaler

    # autoencoder + selezione 80/20
    model = build_autoencoder(Xtr.shape[1])             # shape[1] = n. titoli = n. neuroni input/output
    train_autoencoder(model, Xtr, Xva)
    errors = reconstruction_error_per_stock(model, Xtr) # array 1D: un errore per titolo
    sel = select_stocks(errors, n_select=20, low_frac=0.8)
    sel_idx = sel["selected_idx"]                       # array di indici dei 20 titoli scelti

    # rendimenti reali dei titoli selezionati e dell'indice, sul train
    R_sel = Rlive.loc[w0["train"]].values[:, sel_idx]   # solo righe train -> NumPy -> solo colonne selezionate -> matrice (T x 20)
    R_idx = index_ret.loc[w0["train"]].values           # indice sul train, come vettore (T,)

    res = optimize_weights(R_sel, R_idx, max_weight=0.30)
    w = res["weights"]                                  # array 1D dei 20 pesi ottimi
    print(f"Ottimizzazione riuscita: {res['success']}")
    print(f"Titoli con peso > 0: {int((w > 0).sum())} su {len(w)}")  # (w>0) è una maschera True/False; .sum() conta i True
    print(f"Tracking error in-sample (ATE per giorno): {res['ate_in_sample']:.5f}\n")

    def nome(j):                                        # j = indice tra i selezionati
        t = cols[sel_idx[j]]                            # doppia traduzione: j -> posizione tra i vivi -> ticker
        return str(names[t]) if names is not None and t in names.index else t  # nome esteso se c'e', altrimenti il ticker

    print("Pesi dell'indice ridotto (ordinati):")
    for j in np.argsort(w)[::-1]:                       # argsort = crescente; [::-1] lo inverte -> dal peso piu' grande
        if w[j] > 0:                                    # salto i titoli azzerati dall'ottimizzatore
            print(f"  {cols[sel_idx[j]]:>8}  {w[j]:6.2%}  {nome(j)}")  # :>8 allinea a destra, :6.2% mostra come percentuale

    # confronto con equal-weight sui 20 selezionati
    w_eq = np.full(len(sel_idx), 1 / len(sel_idx))      # pesi tutti uguali: 1/20 = 5% ciascuno
    ate_eq = np.sqrt(np.mean((R_idx - R_sel @ w_eq) ** 2))  # RMSE del portafoglio equipesato (@ = prodotto matrice-vettore)
    print(f"\nATE in-sample: ottimizzato {res['ate_in_sample']:.5f} "
          f"vs equal-weight {ate_eq:.5f} "
          f"(guadagno {(1 - res['ate_in_sample'] / ate_eq) * 100:.0f}%)")  # riduzione % dell'errore rispetto all'equipesato