import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)


class ImageFeaturizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

    def forward(self, x):
        return self.backbone(x)


class DeepSurvivalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_featurizer = ImageFeaturizer()
        self.img_proj = nn.Linear(2048, 64)

        self.mlp = nn.Sequential(
            nn.Linear(66, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, img: Tensor, tabular: Tensor) -> Tensor:
        """Compute the forward pass

        Args:
          img: image tensor of shape (B, C, W, H)

        Returns:
          Predicted log-risk (B, 1)
        """

        img_feat = self.img_featurizer(img)
        img_feat = img_feat.squeeze()
        img_proj = self.img_proj(img_feat)
        x = torch.cat([img_proj, tabular[:, -2:]], dim=1)
        x = self.mlp(x)

        pred_log_risk = F.logsigmoid(x)

        return pred_log_risk
