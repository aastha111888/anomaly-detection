import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import Autoencoder
from src.preprocess import load_and_preprocess


def train():
    X_train, X_test, df, scaler = load_and_preprocess()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    dataset = TensorDataset(X_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    epoch_losses = []

    for epoch in range(100):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for (batch,) in dataloader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.6f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/autoencoder.pth")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 101), epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig("models/loss_curve.png")
    plt.close()

    print("Training complete. Model saved to models/autoencoder.pth")


if __name__ == "__main__":
    train()
