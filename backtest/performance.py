import numpy as np
import pandas as pd

def sharpe(returns: pd.Series, periods_per_year: int = 252):
    return np.sqrt(periods_per_year) * returns.mean() / returns.std()
