import numpy as np


def select_stocks(errors, n_select=20, low_frac=0.8):
    """Regola 80/20: low_frac dei titoli a errore piu' basso (core) + il resto a errore piu' alto (satellite)"""

    n_total = len(errors)
    if n_select > n_total:
        raise ValueError(f"n_select ({n_select}) > titoli disponibili ({n_total})") #Check Robustezza

    ordine = np.argsort(errors)             # indici in ordine crescente degli errori
    n_low = int(round(n_select * low_frac)) 
    n_high = n_select - n_low

    core = ordine[:n_low]                                  # errore piu' basso
    if n_high > 0:
        satellite = ordine[n_total - n_high:]             # errore piu' alto
    else:
        satellite = np.array([], dtype=int)

    selected = np.concatenate([core, satellite]).astype(int)
    return {"selected_idx": selected, "core_idx": core, "satellite_idx": satellite} #Dizionario


# ---------------------------------------------------------------------------
# Esecuzione diretta: selezione sulla prima finestra
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tensorflow as tf
    from data_loader_mib import build_dataset_mib
    from preprocessing_mib import (compute_returns, drop_artifact_stocks,
                                   rolling_windows, live_columns, standardize_split)
    from autoencoder_mib import (build_autoencoder, train_autoencoder,
                                 reconstruction_error_per_stock)

    tf.random.set_seed(42) 
    np.random.seed(42)

    ds = build_dataset_mib("data_mib") 
    prices = drop_artifact_stocks(ds["prices"]) # righe = giorni (datetime), colonne = ticker
    names = ds["names"]
    returns = compute_returns(prices) # righe = giorni (datetime), colonne = ticker

    w0 = rolling_windows(returns.index)[0] # [0] -> Prima finestra
    cols = live_columns(returns, w0) 
    Xtr, Xva, Xte, _ = standardize_split(returns[cols], w0) # 3 array giorni x titoli

    model = build_autoencoder(Xtr.shape[1])     # Xtr.shape = Input neurons = Output neurons
    train_autoencoder(model, Xtr, Xva)  
    errors = reconstruction_error_per_stock(model, Xtr) #MSE tra input (Xtr) e ricostruzione per ogni titolo.


    sel = select_stocks(errors, n_select=20, low_frac=0.8)

    def nome(i):
        t = cols[i]
        return str(names[t]) if names is not None and t in names.index else t

    print(f"Selezionati {len(sel['selected_idx'])} su {len(errors)}: "
          f"{len(sel['core_idx'])} core + {len(sel['satellite_idx'])} satellite\n")
    print("CORE (errore basso, sistematici):")
    for i in sel["core_idx"]:
        print(f"  {cols[i]:>8}  err={errors[i]:.3f}  {nome(i)}")
    print("\nSATELLITE (errore alto, idiosincratici):")
    for i in sel["satellite_idx"]:
        print(f"  {cols[i]:>8}  err={errors[i]:.3f}  {nome(i)}")
