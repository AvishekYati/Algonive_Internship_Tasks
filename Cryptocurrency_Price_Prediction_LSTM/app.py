import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
import torch
import torch.nn as nn

from src.features import (
    download_crypto,
    add_technical_features,
    FEATURE_COLUMNS
)


# ==================================================
# MODEL
# ==================================================

class CryptoLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size_1=128,
        hidden_size_2=64,
        dropout=0.25
    ):

        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            batch_first=True
        )

        self.dropout1 = nn.Dropout(
            dropout
        )

        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            batch_first=True
        )

        self.dropout2 = nn.Dropout(
            0.20
        )

        self.fc1 = nn.Linear(
            hidden_size_2,
            32
        )

        self.relu = nn.ReLU()

        self.output = nn.Linear(
            32,
            1
        )

    def forward(self, x):

        x, _ = self.lstm1(x)

        x = self.dropout1(x)

        x, _ = self.lstm2(x)

        x = x[:, -1, :]

        x = self.dropout2(x)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.output(x)

        return x.squeeze(1)


# ==================================================
# CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Crypto LSTM Forecast",
    page_icon="₿",
    layout="wide"
)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

ARTIFACT_DIR = (
    PROJECT_ROOT /
    "artifacts"
)


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():

    with open(
        ARTIFACT_DIR /
        "model_config.json",
        "r"
    ) as file:

        config = json.load(file)

    with open(
        ARTIFACT_DIR /
        "metadata.json",
        "r"
    ) as file:

        metadata = json.load(file)

    feature_scaler = joblib.load(
        ARTIFACT_DIR /
        "feature_scaler.pkl"
    )

    target_scaler = joblib.load(
        ARTIFACT_DIR /
        "target_scaler.pkl"
    )

    device = torch.device("cpu")

    model = CryptoLSTM(
        input_size=len(
            config["feature_columns"]
        ),
        hidden_size_1=config[
            "hidden_size_1"
        ],
        hidden_size_2=config[
            "hidden_size_2"
        ],
        dropout=config[
            "dropout_1"
        ]
    )

    state_dict = torch.load(
        ARTIFACT_DIR /
        "best_lstm_model.pth",
        map_location=device
    )

    model.load_state_dict(
        state_dict
    )

    model.to(device)

    model.eval()

    return (
        model,
        feature_scaler,
        target_scaler,
        config,
        metadata,
        device
    )


# ==================================================
# DATA
# ==================================================

@st.cache_data(ttl=300)
def load_live_data():

    return download_crypto(
        ticker="BTC-USD",
        start="2024-01-01"
    )


# ==================================================
# MC DROPOUT
# ==================================================

def mc_predict(
    model,
    X,
    target_scaler,
    device,
    n_samples=60
):

    model.train()

    predictions = []

    X = X.to(device)

    with torch.no_grad():

        for _ in range(
            n_samples
        ):

            output = model(X)

            predictions.append(
                output.cpu()
                .numpy()
            )

    predictions = np.asarray(
        predictions
    )

    prices = (
        target_scaler
        .inverse_transform(
            predictions.reshape(
                -1,
                1
            )
        )
        .reshape(
            predictions.shape
        )
    )

    mean_price = (
        prices.mean(axis=0)[0]
    )

    lower = (
        np.percentile(
            prices,
            5,
            axis=0
        )[0]
    )

    upper = (
        np.percentile(
            prices,
            95,
            axis=0
        )[0]
    )

    model.eval()

    return (
        mean_price,
        lower,
        upper
    )


# ==================================================
# LOAD
# ==================================================

(
    model,
    feature_scaler,
    target_scaler,
    config,
    metadata,
    device
) = load_model()


# ==================================================
# HEADER
# ==================================================

st.title(
    "₿ Bitcoin Price Prediction using LSTM"
)

st.subheader(
    "Algonive Internship Project"
)

st.write(
    """
    End-to-end cryptocurrency time-series
    forecasting using a PyTorch LSTM model,
    technical indicators and a live data
    pipeline.
    """
)


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header(
    "Model Configuration"
)

st.sidebar.write(
    f"Cryptocurrency: {config['ticker']}"
)

st.sidebar.write(
    f"Lookback: {config['lookback']} days"
)

st.sidebar.write(
    f"Features: {len(config['feature_columns'])}"
)

if st.sidebar.button(
    "Refresh Market Data"
):

    st.cache_data.clear()

    st.rerun()


# ==================================================
# PREPARE DATA
# ==================================================

data = load_live_data()

features = (
    add_technical_features(
        data
    )
    .dropna()
    .reset_index(drop=True)
)


latest_sequence = (
    features[
        FEATURE_COLUMNS
    ]
    .tail(
        config["lookback"]
    )
)

latest_scaled = (
    feature_scaler.transform(
        latest_sequence
    )
)

X_live = torch.tensor(
    latest_scaled,
    dtype=torch.float32
).unsqueeze(0)


# ==================================================
# FORECAST
# ==================================================

forecast_price, lower, upper = (
    mc_predict(
        model,
        X_live,
        target_scaler,
        device
    )
)

current_price = (
    features[
        "Close"
    ].iloc[-1]
)

expected_change = (
    (
        forecast_price -
        current_price
    )
    /
    current_price
) * 100


# ==================================================
# KPI CARDS
# ==================================================

col1, col2, col3, col4 = (
    st.columns(4)
)

with col1:

    st.metric(
        "Latest BTC Price",
        f"${current_price:,.2f}"
    )

with col2:

    st.metric(
        "Next-Day Forecast",
        f"${forecast_price:,.2f}"
    )

with col3:

    st.metric(
        "Estimated Difference",
        f"{expected_change:+.2f}%"
    )

with col4:

    st.metric(
        "90% Model Interval",
        (
            f"${lower:,.0f}"
            f" - "
            f"${upper:,.0f}"
        )
    )


# ==================================================
# MARKET CHART
# ==================================================

st.subheader(
    "Recent Bitcoin Market Trend"
)

recent = (
    features[
        ["Date", "Close"]
    ]
    .tail(180)
    .set_index("Date")
)

st.line_chart(
    recent
)


# ==================================================
# FORECAST CHART
# ==================================================

st.subheader(
    "Next-Day Forecast"
)

fig, ax = plt.subplots(
    figsize=(12, 5)
)

recent_plot = (
    features.tail(120)
)

ax.plot(
    recent_plot["Date"],
    recent_plot["Close"],
    label="Historical Close"
)

next_date = (
    features[
        "Date"
    ].iloc[-1]
    +
    pd.Timedelta(days=1)
)

ax.scatter(
    [next_date],
    [forecast_price],
    s=100,
    label="LSTM Forecast"
)

ax.vlines(
    next_date,
    lower,
    upper,
    linewidth=4,
    label="90% Model Interval"
)

ax.set_title(
    "Bitcoin Next-Day Forecast"
)

ax.set_xlabel("Date")

ax.set_ylabel(
    "Price (USD)"
)

ax.legend()

fig.tight_layout()

st.pyplot(fig)


# ==================================================
# MODEL PERFORMANCE
# ==================================================

st.subheader(
    "Out-of-Sample Performance"
)

c1, c2, c3, c4 = (
    st.columns(4)
)

with c1:

    st.metric(
        "RMSE",
        f"${metadata['rmse']:,.2f}"
    )

with c2:

    st.metric(
        "MAE",
        f"${metadata['mae']:,.2f}"
    )

with c3:

    st.metric(
        "R²",
        f"{metadata['r2']:.4f}"
    )

with c4:

    st.metric(
        "Direction Accuracy",
        (
            f"{metadata['directional_accuracy_percent']:.2f}%"
        )
    )


# ==================================================
# BASELINE
# ==================================================

st.subheader(
    "Baseline Comparison"
)

baseline_df = pd.DataFrame({
    "Model": [
        "PyTorch LSTM",
        "Naive Previous Close"
    ],

    "MAE": [
        metadata["mae"],
        metadata["naive_mae"]
    ],

    "RMSE": [
        metadata["rmse"],
        metadata["naive_rmse"]
    ]
})

st.dataframe(
    baseline_df,
    use_container_width=True
)


# ==================================================
# DATA FRESHNESS
# ==================================================

st.subheader(
    "Data Information"
)

d1, d2 = (
    st.columns(2)
)

with d1:

    st.write(
        "Latest data date:",
        features[
            "Date"
        ].iloc[-1]
    )

with d2:

    st.write(
        "Records loaded:",
        len(data)
    )


# ==================================================
# RESPONSIBLE USE
# ==================================================

st.warning(
    """
    This is a machine-learning research and
    internship demonstration. Cryptocurrency
    prices are volatile and model predictions
    are uncertain. Forecasts should not be
    treated as guaranteed outcomes or financial
    advice.
    """
)