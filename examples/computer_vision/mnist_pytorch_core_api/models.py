import torch
import torch.nn as nn


class MNISTModel(nn.Module):
    def __init__(
        self,
        n_filters1: int = 32,
        n_filters2: int = 64,
        dropout1: float = 0.25,
        dropout2: float = 0.5,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, n_filters1, 3, 1),
            nn.ReLU(),
            nn.Conv2d(
                n_filters1,
                n_filters2,
                3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout1),
            nn.Flatten(),
            nn.Linear(144 * n_filters2, 128),
            nn.ReLU(),
            nn.Dropout2d(dropout2),
            nn.Linear(128, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
