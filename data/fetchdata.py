# data/fetchdata.py
from pathlib import Path
from datetime import datetime
import pandas as pd
import yfinance as yf
from data.indicators import add_indicators

# ✅ Set your custom download folder
DATA_DIR = Path(__file__).resolve().parent / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

IT_STOCKS = ["AAPL", "MSFT", "NVDA", "AMD", "ORCL"]
INDUSTRIAL_STOCKS = ["CAT", "GE", "HON", "DE", "MMM"]
ALL_STOCKS = IT_STOCKS + INDUSTRIAL_STOCKS


def get_stock_data(symbol: str, start="2015-01-01", end=None) -> pd.DataFrame:
    """Fetch daily stock data using yfinance."""
    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    df = yf.download(symbol, start=start, end=end, progress=False)

    # Fix multi-index columns (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
    }).reset_index()

    # Ensure 1D 'close'
    if isinstance(df["close"], pd.DataFrame):
        df["close"] = df["close"].squeeze()
    return df


def build_universe(start="2015-01-01", end=None) -> pd.DataFrame:
    """Fetch, enrich, and combine multiple sector datasets."""
    all_data = []

    for sym in ALL_STOCKS:
        print(f"→ fetching {sym} …", flush=True)
        df = get_stock_data(sym, start, end)
        if df.empty:
            print(f"   ! no data for {sym} — skipping")
            continue
        df["symbol"] = sym
        df = add_indicators(df)
        all_data.append(df)

        # Save individual stock CSV to the raw data folder
        out_file = DATA_DIR / f"{sym}.csv"
        df.to_csv(out_file, index=False)
        print(f"   saved {out_file.name}  rows={len(df)}")

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined_file = DATA_DIR / "combined_data.csv"
    combined.to_csv(combined_file, index=False)
    print(f"✅ Combined dataset saved: {combined_file}  shape={combined.shape}")
    return combined


if __name__ == "__main__":
    print("=== Building IT + Industrials universe ===")
    df = build_universe()
    if df.empty:
        print("⚠️ No data downloaded. Check tickers or internet connection.")
    else:
        print(df.head().to_string(index=False))
        print("🎉 Done.")
