import numpy as np
import pandas as pd
import yfinance as yf


FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Return_1D",
    "Return_7D",
    "Return_30D",
    "SMA_20",
    "EMA_20",
    "EMA_50",
    "Price_vs_EMA50",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "ATR_14",
    "BB_Width",
    "Volatility_20",
    "Volume_Change",
    "Volume_Z_20",
    "Momentum_7",
    "Momentum_30",
]


def download_crypto(
    ticker="BTC-USD",
    start="2017-01-01"
):
    """
    Download historical cryptocurrency data
    from Yahoo Finance.
    """

    data = yf.download(
        ticker,
        start=start,
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if data.empty:
        raise ValueError(
            f"No data returned for {ticker}."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = (
            data.columns
            .get_level_values(0)
        )

    data = data.reset_index()

    required = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    data = data[required].copy()

    data["Date"] = (
        pd.to_datetime(
            data["Date"],
            utc=True
        )
        .dt.tz_localize(None)
    )

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = (
        data
        .dropna()
        .drop_duplicates(
            subset="Date"
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return data


def calculate_rsi(
    series,
    period=14
):
    """
    Calculate Relative Strength Index.
    """

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    average_gain = (
        gain
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        )
        .mean()
    )

    average_loss = (
        loss
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        )
        .mean()
    )

    rs = (
        average_gain /
        (average_loss + 1e-10)
    )

    return (
        100 -
        (100 / (1 + rs))
    )


def add_technical_features(data):
    """
    Add time-series and technical indicators.
    """

    df = data.copy()

    # Returns
    df["Return_1D"] = (
        df["Close"].pct_change()
    )

    df["Return_7D"] = (
        df["Close"].pct_change(7)
    )

    df["Return_30D"] = (
        df["Close"].pct_change(30)
    )

    # Moving averages
    df["SMA_20"] = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    df["EMA_20"] = (
        df["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["EMA_50"] = (
        df["Close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    df["Price_vs_EMA50"] = (
        df["Close"] /
        df["EMA_50"] - 1
    )

    # RSI
    df["RSI_14"] = calculate_rsi(
        df["Close"],
        14
    )

    # MACD
    ema12 = (
        df["Close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        df["Close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    df["MACD"] = (
        ema12 - ema26
    )

    df["MACD_Signal"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["MACD_Hist"] = (
        df["MACD"] -
        df["MACD_Signal"]
    )

    # ATR
    previous_close = (
        df["Close"].shift(1)
    )

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (
                df["High"] -
                previous_close
            ).abs(),
            (
                df["Low"] -
                previous_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR_14"] = (
        true_range
        .rolling(14)
        .mean()
    )

    # Bollinger Band width
    rolling_mean = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    rolling_std = (
        df["Close"]
        .rolling(20)
        .std()
    )

    upper = (
        rolling_mean +
        2 * rolling_std
    )

    lower = (
        rolling_mean -
        2 * rolling_std
    )

    df["BB_Width"] = (
        (upper - lower) /
        rolling_mean
    )

    # Volatility
    df["Volatility_20"] = (
        df["Return_1D"]
        .rolling(20)
        .std()
    )

    # Volume
    df["Volume_Change"] = (
        df["Volume"].pct_change()
    )

    volume_mean = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    volume_std = (
        df["Volume"]
        .rolling(20)
        .std()
    )

    df["Volume_Z_20"] = (
        (df["Volume"] - volume_mean) /
        (volume_std + 1e-10)
    )

    # Momentum
    df["Momentum_7"] = (
        df["Close"] -
        df["Close"].shift(7)
    )

    df["Momentum_30"] = (
        df["Close"] -
        df["Close"].shift(30)
    )

    # Next-day target
    df["Target_Close"] = (
        df["Close"].shift(-1)
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df