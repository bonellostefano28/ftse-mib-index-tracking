import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Rendimenti giornalieri
# ---------------------------------------------------------------------------
def compute_returns(prices):
    """(P_t / P_{t-1}) - 1."""
    returns = prices.pct_change()
    return returns.iloc[1:] # Scarta la prima riga di NaN

# ---------------------------------------------------------------------------
# Scarto dei titoli con artefatti da corporate action
# ---------------------------------------------------------------------------
def drop_artifact_stocks(prices, threshold=0.40):
    """Toglie i titoli con un rendimento giornaliero oltre +/-threshold."""
    returns = compute_returns(prices) #Rows = Days; Col = Stocks
    buoni = []
    for c in prices.columns:
        serie = returns[c].dropna()
        salto = max(abs(serie.min()), abs(serie.max())) if len(serie) else 0.0
        if salto <= threshold:
            buoni.append(c)
    return prices[buoni] #Rows = Days; Col = Stocks(Buoni)


# ---------------------------------------------------------------------------
# Indice target value-weighted ricostruito
# ---------------------------------------------------------------------------
def value_weighted_index(prices, shares):
    """Rendimento giornaliero dell'indice pesato per capitalizzazione.

    Peso del titolo i nel giorno t = cap_{i,t-1} / somma delle cap, con
    cap = azioni * prezzo. Usiamo solo i titoli con un rendimento valido quel
    giorno (vivi sia a t-1 sia a t): cosi' nascite e uscite non creano salti
    finti nel livello dell'indice. E' l'indice che proveremo a replicare.
    """
    returns = compute_returns(prices)
    cap_prev = prices.shift(1).mul(shares, axis=1)     # DataFrame x Series
    cap_prev = cap_prev.reindex(returns.index)         # Stesso indice dei rendimenti
    cap_prev = cap_prev.where(returns.notna())         # Robustezza dei dati
    weights = cap_prev.div(cap_prev.sum(axis=1), axis=0) 
    index_ret = (weights * returns).sum(axis=1)
    return index_ret


# ---------------------------------------------------------------------------
# Validazione contro il FTSE MIB ufficiale
# ---------------------------------------------------------------------------
def validate_index(index_ret, ftse_mib):
    """Correlazione tra i rendimenti dell'indice ricostruito e quelli del FTSE MIB."""
    mib_ret = ftse_mib.pct_change().dropna()
    common = index_ret.index.intersection(mib_ret.index) # Confronto su date combacianti
    return float(np.corrcoef(index_ret.loc[common], mib_ret.loc[common])[0, 1])


# ---------------------------------------------------------------------------
# Divisione Dataset per l'Autoencoder
# ---------------------------------------------------------------------------
def rolling_windows(index, train=1008, val=63, test=126, step=126):
    """Genera finestre temporali scorrevoli (train/val/test) su un indice di date, 
       avanzando di step giorni ad ogni iterazione."""

    days = list(index)
    size = train + val + test
    out = []
    start = 0
    while start + size <= len(days):
        tr = days[start: start + train] 
        va = days[start + train: start + train + val] 
        te = days[start + train + val: start + size] 
        out.append({"train": pd.DatetimeIndex(tr),
                    "val": pd.DatetimeIndex(va),
                    "test": pd.DatetimeIndex(te)})
        start += step # Finestra mobile (6 mesi)
    return out


# ---------------------------------------------------------------------------
# Titoli stabili nella finestra (not a number)
# ---------------------------------------------------------------------------
def live_columns(returns, window):
    """Restituisce solo le colonne con nessun valore mancante per l'intera finestra di train/val/test. """
    idx = window["train"].union(window["val"]).union(window["test"])
    block = returns.loc[idx] 
    cols = []
    for c in block.columns:
        if block[c].notna().all():
            cols.append(c)
    return cols


# ---------------------------------------------------------------------------
# Standardizzazione (z-score)
# ---------------------------------------------------------------------------
def standardize_split(returns, window):
    """Z-score di ogni titolo con media/dev.std stimate solo sul train."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(returns.loc[window["train"]].values)
    Xva = scaler.transform(returns.loc[window["val"]].values)
    Xte = scaler.transform(returns.loc[window["test"]].values)
    return Xtr, Xva, Xte, scaler


# ---------------------------------------------------------------------------
# Script di validazione: indice ricostruito vs FTSE MIB ufficiale
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib 
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from data_loader_mib import build_dataset_mib

    ds = build_dataset_mib("data_mib")
    prices, shares, mib = ds["prices"], ds["shares"], ds["index"]
    prices = drop_artifact_stocks(prices)
    shares = shares[prices.columns]              # riallinea le azioni ai titoli buoni

    index_ret = value_weighted_index(prices, shares)
    corr = validate_index(index_ret, mib)       

    # --- grafico di validazione: nostro indice vs FTSE MIB (base 100) ---
    our_level = (1 + index_ret).cumprod()
    our_level = our_level / our_level.iloc[0] * 100
    mib_level = mib.reindex(our_level.index).ffill()
    mib_level = mib_level / mib_level.iloc[0] * 100

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(our_level.index, our_level, lw=1,
            label="Value-weighted index (total return)")
    ax.plot(mib_level.index, mib_level, lw=1, alpha=0.8,
            label="Official FTSE MIB (price index)")
    ax.set_title(f"Index validation - daily return correlation {corr:.3f}")
    ax.set_ylabel("level (base 100)"); ax.legend()
    fig.tight_layout()
    fig.savefig("validation_index_mib.png", dpi=120)