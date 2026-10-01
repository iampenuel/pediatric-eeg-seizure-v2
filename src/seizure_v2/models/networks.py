from torch import nn


class V1CNN(nn.Module):
    """Conceptual V1 topology; tensor convention is batch, channel, time."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(18, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, 64),
            nn.ReLU(), nn.Dropout(.3), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(incoming, outgoing, 7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(outgoing), nn.ReLU(),
            nn.Conv1d(outgoing, outgoing, 7, padding=3, bias=False), nn.BatchNorm1d(outgoing),
        )
        self.shortcut = nn.Identity() if incoming == outgoing and stride == 1 else nn.Sequential(
            nn.Conv1d(incoming, outgoing, 1, stride=stride, bias=False), nn.BatchNorm1d(outgoing))
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.main(x) + self.shortcut(x))


class ResidualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(18, 32, 7, padding=3, bias=False), nn.BatchNorm1d(32), nn.ReLU(),
            ResidualBlock(32, 32, 1), ResidualBlock(32, 64, 2), ResidualBlock(64, 128, 2),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(.3), nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)
