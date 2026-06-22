import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Tracking the FTSE MIB with few stocks",
                   layout="wide")



# Diagramma della pipeline come SVG (Scalable Vector Graphics)
def diagramma_pipeline():
    svg = """
    <style>
      :root{
        --page:#ffffff;
        --node-fill:#EEF1F5; --node-stroke:#C5CCD6; --node-num:#3A4049;
        --ae-fill:#FBF1D9;   --ae-stroke:#C9A24B;   --ae-num:#8A6D1F;
        --conn:#9AA1AA; --title:#1F2430; --sub:#6B727C;
      }
      @media (prefers-color-scheme: dark){
        :root{
          --page:#0E1117;
          --node-fill:#222831; --node-stroke:#39414B; --node-num:#C9CDD3;
          --ae-fill:#2B2614;   --ae-stroke:#C9A24B;   --ae-num:#E6C977;
          --conn:#5C636C; --title:#ECE9E2; --sub:#9098A2;
        }
      }
      html,body{margin:0; background:var(--page);}
      .nd{fill:var(--node-fill); stroke:var(--node-stroke); stroke-width:0.8;}
      .ae{fill:var(--ae-fill); stroke:var(--ae-stroke); stroke-width:1.4;}
      .num{fill:var(--node-num); font-size:14px; font-weight:500;}
      .aenum{fill:var(--ae-num); font-size:14px; font-weight:500;}
      .ttl{fill:var(--title); font-size:14px; font-weight:500;}
      .sub{fill:var(--sub); font-size:12px;}
      .conn{stroke:var(--conn); stroke-width:1.5; fill:none;}
    </style>
    <div style="max-width:980px; margin:0 auto; padding:8px 0;">
    <svg width="100%" viewBox="0 0 680 190" role="img"
         font-family="'Segoe UI', system-ui, Arial, sans-serif">
      <title>Index-tracking pipeline</title>
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
                markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
                stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </marker>
      </defs>

      <!-- connettori tra un nodo e il successivo (direzione del flusso) -->
      <line class="conn" x1="102" y1="110" x2="162" y2="110" marker-end="url(#arrow)"/>
      <line class="conn" x1="206" y1="110" x2="266" y2="110" marker-end="url(#arrow)"/>
      <line class="conn" x1="310" y1="110" x2="370" y2="110" marker-end="url(#arrow)"/>
      <line class="conn" x1="414" y1="110" x2="474" y2="110" marker-end="url(#arrow)"/>
      <line class="conn" x1="518" y1="110" x2="578" y2="110" marker-end="url(#arrow)"/>

      <!-- 1. Data loader (neutro), etichetta SOPRA -->
      <circle class="nd" cx="80" cy="110" r="18"/>
      <text class="num" x="80" y="110" text-anchor="middle" dominant-baseline="central">1</text>
      <text class="ttl" x="80" y="58"  text-anchor="middle">Data loader</text>
      <text class="sub" x="80" y="76"  text-anchor="middle">reads &amp; aligns</text>

      <!-- 2. Preprocessing (neutro), etichetta SOTTO -->
      <circle class="nd" cx="184" cy="110" r="18"/>
      <text class="num" x="184" y="110" text-anchor="middle" dominant-baseline="central">2</text>
      <text class="ttl" x="184" y="150" text-anchor="middle">Preprocessing</text>
      <text class="sub" x="184" y="168" text-anchor="middle">returns, windows</text>

      <!-- 3. Autoencoder (ACCENTO ORO: cuore del metodo), etichetta SOPRA -->
      <circle class="ae" cx="288" cy="110" r="18"/>
      <text class="aenum" x="288" y="110" text-anchor="middle" dominant-baseline="central">3</text>
      <text class="ttl" x="288" y="58"  text-anchor="middle">Autoencoder</text>
      <text class="sub" x="288" y="76"  text-anchor="middle">reconstruction error</text>

      <!-- 4. Stock selector (neutro), etichetta SOTTO -->
      <circle class="nd" cx="392" cy="110" r="18"/>
      <text class="num" x="392" y="110" text-anchor="middle" dominant-baseline="central">4</text>
      <text class="ttl" x="392" y="150" text-anchor="middle">Stock selector</text>
      <text class="sub" x="392" y="168" text-anchor="middle">80/20 rule</text>

      <!-- 5. Weight optimizer (neutro), etichetta SOPRA -->
      <circle class="nd" cx="496" cy="110" r="18"/>
      <text class="num" x="496" y="110" text-anchor="middle" dominant-baseline="central">5</text>
      <text class="ttl" x="496" y="58"  text-anchor="middle">Weight optimizer</text>
      <text class="sub" x="496" y="76"  text-anchor="middle">constrained LSQ</text>

      <!-- 6. Evaluation (neutro), etichetta SOTTO -->
      <circle class="nd" cx="600" cy="110" r="18"/>
      <text class="num" x="600" y="110" text-anchor="middle" dominant-baseline="central">6</text>
      <text class="ttl" x="600" y="150" text-anchor="middle">Evaluation</text>
      <text class="sub" x="600" y="168" text-anchor="middle">OOS error</text>
    </svg>
    </div>
    """
    # altezza dell'iframe = altezza max del disegno (980/680*190 ~ 274) + margine
    components.html(svg, height=300, scrolling=False)



# Utility: mostra un'immagine
#"""For loop in caso di espansione del progetto e voglia cercare in altre cartelle"""
def mostra_immagine(nome_file, didascalia=""):
    for p in [nome_file, os.path.join("assets", nome_file)]:
        if os.path.exists(p):
            st.image(p, caption=didascalia, use_container_width=True)
            return
    st.info(f"(image not found: {nome_file} - put it next to the app)")



# Utility: mostra il codice di un modulo

def mostra_codice(nome_file, aperto=True):
    if not os.path.exists(nome_file):
        st.info(f"(module not found: {nome_file})")
        return
    with st.expander(f"Code: {nome_file}", expanded=aperto):
        with open(nome_file, "r", encoding="utf-8") as f:
            st.code(f.read(), language="python")



# Sezione 1 - Goal

def sez_goal():
    st.title("Tracking the FTSE MIB with few stocks ")
    st.subheader("Can a handful of stocks stand in for the whole index?")
    st.write(
        "The FTSE MIB has ~40 stocks. Can just a few of them replicate it? "
        "This project builds a **reduced portfolio**: an **autoencoder** chooses "
        "which stocks to keep, a **constrained optimizer** sets their weights, "
        "and the replica is judged **out-of-sample** by its tracking error."
    )
    st.write(
        "But tracking is the easy part. The real test: does the autoencoder's "
        "selection beat a **trivial baseline** - just taking the largest stocks "
        "by market cap - out-of-sample?"
    )
    st.caption("Machine Learning in Finance - Introduzione a Python project. FTSE MIB, daily, ~11 years.")
    mostra_immagine("Upo.png")


# Sezione 2 - Pipeline overview

def sez_pipeline():
    st.header("From data to a tracked index")
    st.write("From raw prices to a tracked index in six steps, each one a self-contained Python module.")
    diagramma_pipeline()
    st.markdown(
        "- **Data loader** -> reads and aligns prices and shares.\n"
        "- **Preprocessing** -> returns, value-weighted index, rolling windows.\n"
        "- **Autoencoder** -> compresses returns, gives a reconstruction error "
        "per stock.\n"
        "- **Stock Selector** -> ranks stocks by error and keeps an 80/20 core-satellite mix.\n"
        "- **Weight Optimizer** -> constrained least squares for the weights.\n"
        "- **Evaluation** -> freezes everything and measures the error on unseen data."
    )



# Sezione 3 - Data loader

def sez_data_loader():
    st.header("Phase 1 - Data loader")
    st.write(
        "The data loader reads the cached daily data (prices, shares, names, "
        "official index) and hand a clean, aligned dataset to the next phases. "
        "Prices are Adjusted Close from yfinance. "
        "The output is a single dictionary, so every downstream module works on the same universe."
    )
    mostra_codice("data_loader_mib.py")



# Sezione 4 - Preprocessing (+ validazione)

def sez_preprocessing():
    st.header("Phase 2 - Preprocessing")
    st.markdown(
        "- **Returns**: simple daily returns.\n"
        "- **Value-weighted index** built in-house (the target to replicate) and "
        "**validated** against the official FTSE MIB.\n"
        "- **+/-40% rule**: drop stocks with corporate-action artifacts.\n"
        "- **Rolling windows** (train 1008 / val 63 / test 126, step 126).\n"
        "- **live_columns**: only stocks quoted for the whole window (point-in-time).\n"
        "- **Standardize** with train statistics only (no lookahead)."
    )
    st.write("**Validation**: the return correlation between our index and the "
             "official FTSE MIB is **0.99**.")
    mostra_immagine("validation_index_mib.png",
                    "Our value-weighted index vs official FTSE MIB")
    mostra_codice("preprocessing_mib.py", aperto=False)



# Sezione 5 - The Autoencoder (architettura)

def sez_autoencoder():
    st.header("Phase 3 - The Autoencoder")
    st.write(
        "An **undercomplete** autoencoder with one bottleneck layer. It takes the "
        "standardized returns of one day and reconstructs them through a few "
        "latent factors. Stocks well reconstructed are 'core' (systematic); badly "
        "reconstructed ones are 'satellite' (idiosyncratic). The per-stock "
        "reconstruction error is the signal for the Stock Selector."
    )

    spec = pd.DataFrame({
        "property": [
            "Input neurons", "Hidden layers", "Neurons per hidden layer",
            "Output neurons", "Hidden activation", "Output activation",
            "Loss function", "Optimizer", "Regularization", "Weight init",
            "Training",
        ],
        "value (FTSE MIB)": [
            "= n live stocks in the window (~32-39)",
            "1 (the bottleneck)",
            "4 (latent_dim, chosen by sweep)",
            "= input neurons (reconstruction)",
            "SeLU",
            "Linear (none)",
            "MSE (mean squared error)",
            "Adam (lr 1e-3)",
            "L1 on latent activations (1e-4) -> sparse AE",
            "LeCun normal (pairs with SeLU)",
            "Early stopping (patience 40, restore best), batch 32",
        ],
    })
    st.table(spec)

    st.markdown(
        "**Design choices**: \n"
        "- **SeLU, not ReLU**: standardized returns are positive *and* negative; "
        "ReLU would zero out the negatives, SeLU keeps them and is "
        "self-normalizing (with LeCun-normal init).\n"
        "- **Linear output**: it must reconstruct real, unbounded numbers; linear "
        "+ MSE is the standard for a regression autoencoder.\n"
        "- **Sparse (L1 on activations)**: a sparse, cleaner latent code -> more "
        "interpretable common factors.\n"
        "- **Undercomplete (4 << n)**: forces the network to *learn* the factors "
        "instead of copying the input."
    )
    mostra_codice("autoencoder_mib.py", aperto=False)



# Sezione 6 - Stock Selector

def sez_selector():
    st.header("Phase 4 - Stock Selector (80/20 rule)")
    st.write(
        "From the per-stock reconstruction error, sort stocks from low error "
        "(core) to high error (satellite), then keep:"
    )
    st.markdown(
        "- **80%** from the low-error group (**core**): well explained by the "
        "common factors, the heart of the index;\n"
        "- **20%** from the high-error group (**satellite**): idiosyncratic, they "
        "add the diversification the factors alone miss."
    )
    st.latex(r"\mathcal{L}_j = \sum_{i=1}^{N} \left\| x_j^{(i)} - x'^{(i)}_j \right\|_2")
    st.caption("Reconstruction loss of stock j = the selection signal.")
    mostra_codice("selector_mib.py")



# Sezione 7 - Weight Optimizer

def sez_optimizer():
    st.header("Phase 5 - Weight Optimizer")
    st.write("Find the weights of the selected stocks that best replicate the "
             "index returns: constrained least squares.")
    st.latex(r"w^* = \arg\min_w \; \left\| R_{index} - R_{stocks}\, w \right\|_2^2")
    st.markdown(
        "Constraints:\n"
        "- $w_i \\ge 0$ (no short selling);\n"
        "- $w_i \\le 0.30$ (no stock above 30%);\n"
        "- $\\sum_i w_i = 1$ (fully invested)."
    )
    st.caption("Real (not standardized) returns: the weights must replicate the "
               "actual index, the standardization was only for the autoencoder.")
    mostra_codice("optimizer_mib.py")



# Sezione 8 - The research question

def sez_domanda():
    st.header("The research question")
    st.write("The project separates two goals that are easy to confuse:")
    st.markdown(
        "- **Tracking the index** - the easy goal: does the reduced portfolio "
        "follow the benchmark? The optimizer manages this almost always, so on "
        "its own it proves little about the *selection*.\n"
        "- **Beating the baseline out-of-sample** - the real goal: does the "
        "autoencoder selection replicate the index *better* than a trivial rule "
        "- just take the largest stocks by market cap - when stocks and weights "
        "are **frozen** on the train and measured on **unseen** test windows?"
    )



# Sezione 9 - Result on FTSE MIB

def sez_risultato():
    st.header("Result - FTSE MIB")
    st.write(
        "Out-of-sample tracking error (lower = better replica), autoencoder "
        "selection vs market-value ranking, across all windows:"
    )
    mostra_immagine("figura10_mib.png", "OOS tracking error: autoencoder vs market-value")
    st.write(
        "The market-value baseline **wins** by ~1.5x at every n. A cap-weighted "
        "index is almost entirely its largest stocks, so 'take the largest' is "
        "already near-optimal; the 80/20 spends 20% on idiosyncratic satellites, "
        "the worst trackers of a cap index."
    )
    st.divider()
    st.subheader("Tracking over time: levels and cumulative drift")
    mostra_immagine("tracking_mib.png",
                    "Top: wealth (base 100), index vs the two reduced portfolios. "
                    "Bottom: cumulative tracking drift (replica - index) over the OOS period.")
    st.write(
        "The 14 test windows are contiguous, so they stitch into one long "
        "out-of-sample backtest, re-selected and re-weighted every 126 days. The "
        "top panel shows the levels; the bottom panel shows how much each reduced "
        "portfolio drifts away from the index as the differences compound. "
    )



# Sezione 10 - Conclusions

def sez_conclusioni():
    st.header("Conclusions")
    st.markdown(
        "- **The replica tracks the FTSE MIB.** A reduced portfolio of ~20 stocks "
        "follows the index closely (return correlation 0.99). The easy bar is "
        "cleared.\n"
        "- **But** Out-of-sample, the "
        "autoencoder selection does **not** beat the trivial *largest by market "
        "cap* baseline: it is worse at every n (around 1.5x), and worst exactly "
        "where selection matters most - when you keep few stocks.\n"
        "- **Why:** It is not a training failure, "
        "the network generalizes and the selection is stable out-of-sample. The limit is "
        "the **criterion**: the reconstruction error is partly aligned with size "
        "(corr -0.26), and a cap-weighted index *is* its largest names - so 'take "
        "the biggest' is already near-optimal, while the 80/20 spends 20% on "
        "small, idiosyncratic satellites, the worst trackers of a cap index.\n"
        "- **Open question:** does a **de-concentrated** "
        "target (equal-weight) or a **larger** universe change the verdict? On the "
        "FTSE MIB an equal-weight target narrows the gap but does not flip it - "
        "pointing to universe size, explored separately on the STOXX Europe 600."
    )



# Navigazione

SEZIONI = {
    "1. Goal & question": sez_goal,
    "2. Pipeline overview": sez_pipeline,
    "3. Data loader": sez_data_loader,
    "4. Preprocessing": sez_preprocessing,
    "5. The Autoencoder": sez_autoencoder,
    "6. Stock Selector": sez_selector,
    "7. Weight Optimizer": sez_optimizer,
    "8. Research question": sez_domanda,
    "9. Result: FTSE MIB": sez_risultato,
    "10. Conclusions": sez_conclusioni,
}

st.sidebar.title("Presentation")
st.sidebar.caption("Stefano Bonello - Anthony Pio Fornaro")
scelta = st.sidebar.radio("Sections", list(SEZIONI.keys()))
SEZIONI[scelta]()
