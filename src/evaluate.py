import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.model import Autoencoder
from src.preprocess import FEATURE_COLS, load_and_preprocess


def evaluate():
    X_train, X_test, df, scaler = load_and_preprocess()

    model = Autoencoder()
    model.load_state_dict(torch.load("models/autoencoder.pth", weights_only=True))
    model.eval()

    X_all = scaler.transform(df[FEATURE_COLS])
    X_tensor = torch.tensor(X_all, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(X_tensor)
        reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()

    df = df.copy()
    df["reconstruction_error"] = reconstruction_errors

    train_mask = df.index < "2020-01-01"
    threshold = np.percentile(df.loc[train_mask, "reconstruction_error"], 99)
    df["is_anomaly"] = df["reconstruction_error"] > threshold

    print(f"Threshold (99th percentile on training data): {threshold:.6f}")
    print(f"Total anomalies detected: {df['is_anomaly'].sum()}")

    print("\nTop 10 anomalous days:")
    top_10 = df.nlargest(10, "reconstruction_error")
    for date, row in top_10.iterrows():
        print(f"  {date.date()}: {row['reconstruction_error']:.6f}")

    os.makedirs("models", exist_ok=True)

    fig, (ax_price, ax_error) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax_price.plot(df.index, df["Close"], label="SPY Close", color="steelblue")
    anomaly_df = df[df["is_anomaly"]]
    ax_price.scatter(
        anomaly_df.index,
        anomaly_df["Close"],
        color="red",
        s=20,
        label="Anomaly",
        zorder=5,
    )
    ax_price.set_title("SPY Closing Price with Detected Anomalies")
    ax_price.set_ylabel("Close Price ($)")
    ax_price.legend()
    ax_price.grid(True, alpha=0.3)

    ax_error.plot(
        df.index,
        df["reconstruction_error"],
        label="Reconstruction Error",
        color="steelblue",
    )
    ax_error.axhline(
        threshold,
        color="red",
        linestyle="--",
        label=f"Threshold ({threshold:.4f})",
    )
    ax_error.set_title("Reconstruction Error Over Time")
    ax_error.set_xlabel("Date")
    ax_error.set_ylabel("MSE")
    ax_error.legend()
    ax_error.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("models/anomaly_plot.png")
    plt.show()


if __name__ == "__main__":
    evaluate()
