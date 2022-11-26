import torch
from torch import nn
from torch.nn import functional as F


class MLPSurvivalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.reg = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass

        Args:
          x: tabular tensor of shape (B, n_feat)

        Returns:
          Predicted log-risk (B, 1)
        """

        x = self.reg(x)
        x = F.logsigmoid(x)
        return x
