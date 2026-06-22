import os
import pandas as pd

# ---------------------------------------------------------------------------
# Risoluzione della cartella dati
# ---------------------------------------------------------------------------
def _resolve_folder(folder):
    """Trova dove stanno davvero i CSV.

    Prova prima la cartella richiesta (default "data_mib/"), poi ripiega sulla
    cartella corrente. Cosi' il progetto gira sia con i dati in "data_mib/" sia
    con i dati nella root del repo, senza dover toccare i moduli a valle.
    """
    for cand in (folder, "."):
        if os.path.exists(os.path.join(cand, "prices.csv")):
            return cand
    raise FileNotFoundError(
        f"prices.csv non trovato ne' in '{folder}/' ne' nella cartella corrente. "
        f"Metti prices.csv, shares.csv, names.csv, index.csv in una di queste posizioni."
    )


# ---------------------------------------------------------------------------
# Lettura dei files
# ---------------------------------------------------------------------------
def load_prices(folder):
    """Legge i prezzi"""
    path = os.path.join(folder, "prices.csv")
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    return prices.sort_index()


def load_shares(folder):
    """Legge il numero di azioni in circolazione"""
    path = os.path.join(folder, "shares.csv")
    shares = pd.read_csv(path, index_col=0)["shares"]
    return shares.dropna()                       # elimino i titoli senza azioni


def load_names(folder):
    """Legge i nomi estesi se il file c'e'; altrimenti torna None."""
    path = os.path.join(folder, "names.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0).iloc[:, 0]
    return None


def load_index(folder):
    """Legge il FTSE MIB ufficiale per la validazione. """
    path = os.path.join(folder, "index.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    return None


# ---------------------------------------------------------------------------
def build_dataset_mib(folder="data_mib"):
    """Unisce prezzi, azioni e nomi su un solo elenco di titoli."""
    folder = _resolve_folder(folder)          # trova i dati in data_mib/ o nella root
    prices = load_prices(folder)
    shares = load_shares(folder)
    names = load_names(folder)
    index = load_index(folder)            

    # teniamo solo i titoli presenti sia nei prezzi sia nelle azioni
    cols = []
    for c in prices.columns:
        if c in shares.index:
            cols.append(c)

    prices = prices[cols]
    shares = shares[cols]
    if names is not None:
        names = names.reindex(cols) # titolo presente ma senza nome -> NaN

    return {"prices": prices, "shares": shares, "names": names, "index": index} #dizionario
