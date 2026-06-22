
import numpy as np
import tensorflow as tf
from tensorflow import keras # type: ignore
from tensorflow.keras import layers, regularizers, initializers # type: ignore


# ---------------------------------------------------------------------------
# Costruzione della rete
# ---------------------------------------------------------------------------
def build_autoencoder(n_inputs, latent_dim=4, l1_coeff=1e-4, learning_rate=1e-3):
    
    init = initializers.lecun_normal()                 # inizializzatore adatto a SeLU
    activity_reg = regularizers.l1(l1_coeff) if l1_coeff else None

    inputs = keras.Input(shape=(n_inputs,))
    latent = layers.Dense(latent_dim, activation="selu",
                          kernel_initializer=init,
                          activity_regularizer=activity_reg,
                          name="latent")(inputs)
    outputs = layers.Dense(n_inputs, kernel_initializer=init,
                           name="reconstruction")(latent)

    model = keras.Model(inputs, outputs, name="autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.MeanSquaredError())
    return model


# ---------------------------------------------------------------------------
# Addestramento 
# ---------------------------------------------------------------------------
def train_autoencoder(model, X_train, X_val=None, epochs=400, batch_size=32,
                      patience=40, verbose=0):

    callbacks = []
    validation_data = None
    if X_val is not None and len(X_val) > 0:
        validation_data = (X_val, X_val)
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True))
    model.fit(X_train, X_train, validation_data=validation_data,
              epochs=epochs, batch_size=batch_size,
              callbacks=callbacks, verbose=verbose)
    return model


# ---------------------------------------------------------------------------
# Errore di ricostruzione per titolo 
# ---------------------------------------------------------------------------
def reconstruction_error_per_stock(model, X):
    
    X_hat = model.predict(X, verbose=0)
    return np.mean((X - X_hat) ** 2, axis=0) # MSE


# ---------------------------------------------------------------------------
# Sweep della dimensione latente sulla prima finestra
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader_mib import build_dataset_mib
    from preprocessing_mib import (compute_returns, drop_artifact_stocks,
                                   rolling_windows, live_columns, standardize_split)

    ds = build_dataset_mib("data_mib")
    prices = drop_artifact_stocks(ds["prices"])
    returns = compute_returns(prices)

    w0 = rolling_windows(returns.index)[0]
    cols = live_columns(returns, w0)
    Xtr, Xva, Xte, _ = standardize_split(returns[cols], w0)
    print(f"Prima finestra: {len(cols)} titoli, X_train {Xtr.shape}\n")

    # --- sweep della dimensione latente (metodo del prof) ---
    print("Sweep latente (spread = quanto sono diversi gli errori tra titoli):")
    for k in [4, 8, 12, 16, 20]:
        m = build_autoencoder(Xtr.shape[1], latent_dim=k)
        train_autoencoder(m, Xtr, Xva, epochs=300, patience=30)
        err = reconstruction_error_per_stock(m, Xtr)
        print(f"  latent={k:>2}: err medio={err.mean():.3f}  "
              f"spread(std)={err.std():.3f}  max/min={err.max()/err.min():.1f}")

    # --- modello scelto: titoli meglio e peggio ricostruiti ---
    LATENT = 4
    print(f"\nModello scelto: latent_dim={LATENT}")
    model = build_autoencoder(Xtr.shape[1], latent_dim=LATENT)
    train_autoencoder(model, Xtr, Xva)
    err = reconstruction_error_per_stock(model, Xtr)