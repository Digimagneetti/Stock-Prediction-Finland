
import pandas as pd
import numpy as np

def calculate_rsi(series, periods=4):
    """
    RSI-indikaattori. Laskee Relative Strength Indexin annetulle aikasarjalle.

    Args:
        series (pd.Series): Hintasarja (yleensä 'Close').
        periods (int): Laskentajaksojen määrä.

    Returns:
        pd.Series: RSI-arvot.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()

    # Nollalla jakamisen esto:
    # Jos loss on 0 ja gain > 0, RSI on 100 (vain nousuja).
    # Jos molemmat ovat 0, RSI on 0 (vakaa hinta).
    rs = np.where(loss == 0, np.where(gain > 0, np.inf, 0), gain / loss)

    return 100 - (100 / (1 + rs))
