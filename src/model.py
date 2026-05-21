import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def encode(self, x):
        return self.encoder(x)


if __name__ == "__main__":
    model = Autoencoder()
    print(model)

    sample = torch.randn(1, 4)
    output = model(sample)
    print(f"\nOutput shape: {output.shape}")
