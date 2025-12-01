# data/fetchdata.py
from pathlib import Path
import pandas as pd
import yfinance as yf
from data.indicators import add_indicators

# folder where CSVs will be saved
DATA_DIR = Path(__file__).resolve().parent / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

IT_STOCKS = ["AAPL", "MSFT", "NVDA", "AMD", "ORCL"]
INDUSTRIAL_STOCKS = ["CAT", "GE", "HON", "DE", "MMM"]
ALL_STOCKS = IT_STOCKS + INDUSTRIAL_STOCKS


def get_stock_data(symbol: str) -> pd.DataFrame:
    """Fetch daily stock data using yfinance (period-based)."""
    print(f"  → yf.download({symbol}, period='5y')")
    df = yf.download(
        symbol,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    print(f"    raw shape for {symbol}: {df.shape}")

    if df.empty:
        print(f"    ❌ no data for {symbol}")
        return df

    # handle multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = (
        df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        })
        .reset_index()
        .rename(columns={"Date": "date"})
    )

    # make sure 'close' is a 1D Series
    if isinstance(df["close"], pd.DataFrame):
        df["close"] = df["close"].squeeze()

    return df


def build_universe() -> pd.DataFrame:
    """Fetch, enrich, and combine multiple sector datasets."""
    all_data = []

    for sym in ALL_STOCKS:
        print(f"\n=== Fetching {sym} ===")
        df = get_stock_data(sym)

        if df.empty:
            print(f"   ⚠️ Skipping {sym}, still empty after download.")
            continue

        print(f"   first rows for {sym}:")
        print(df.head())

        df["symbol"] = sym

        # add indicators
        before_rows = len(df)
        df = add_indicators(df)
        after_rows = len(df)
        print(f"   rows before indicators for {sym}: {before_rows}")
        print(f"   rows after  indicators for {sym}: {after_rows}")

        if df.empty:
            print(f"   ⚠️ All rows dropped after indicators for {sym}")
            continue

        out_file = DATA_DIR / f"{sym}.csv"
        df.to_csv(out_file, index=False)
        print(f"   💾 saved {out_file}  rows={len(df)}")

        all_data.append(df)

    if not all_data:
        print("\n❌ No data for ANY symbol. combined_data.csv would be empty.")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined_file = DATA_DIR / "combined_data.csv"
    combined.to_csv(combined_file, index=False)
    print(f"\n✅ Combined dataset saved: {combined_file}  shape={combined.shape}")
    return combined


if __name__ == "__main__":
    print("=== Building IT + Industrials universe ===")
    df = build_universe()
    print("\nSummary:")
    print(f"  Total rows: {len(df)}")
    if "symbol" in df.columns:
        print("  Symbols:", df["symbol"].unique())
