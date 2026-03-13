
import pandas as pd
import numpy as np
import unittest
from indicators import calculate_rsi

class TestRSI(unittest.TestCase):
    def test_calculate_rsi_constant_prices(self):
        prices = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        rsi = calculate_rsi(prices, periods=4)
        # Both gain and loss are 0 -> rs=0 -> rsi=0
        np.testing.assert_array_equal(rsi.iloc[-2:], [0.0, 0.0])

    def test_calculate_rsi_only_gains(self):
        prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        rsi = calculate_rsi(prices, periods=4)
        # loss is 0, gain > 0 -> rs=inf -> rsi=100
        np.testing.assert_array_equal(rsi.iloc[-2:], [100.0, 100.0])

    def test_calculate_rsi_only_losses(self):
        prices = pd.Series([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
        rsi = calculate_rsi(prices, periods=4)
        # gain is 0, loss > 0 -> rs=0 -> rsi=0
        np.testing.assert_array_equal(rsi.iloc[-2:], [0.0, 0.0])

    def test_calculate_rsi_mixed(self):
        prices = pd.Series([10.0, 12.0, 10.0, 12.0, 10.0, 12.0])
        rsi = calculate_rsi(prices, periods=4)
        # avg_gain = 1.0, avg_loss = 1.0 -> rs=1.0 -> rsi=50.0
        np.testing.assert_array_equal(rsi.iloc[-2:], [50.0, 50.0])

if __name__ == "__main__":
    # We won't run it if dependencies are missing, but this is the proper test.
    try:
        unittest.main()
    except Exception as e:
        print(f"Could not run tests due to missing dependencies: {e}")
