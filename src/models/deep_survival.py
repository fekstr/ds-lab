import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)


class DeepSurvivalModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = resnet50(ResNet50_Weights.DEFAULT)
        fc_in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(fc_in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass

        Args:
          x: image tensor of shape (B, C, W, H)

        Returns:
          Predicted log-risk (B, 1)
        """

        x = self.backbone(x)
        x = F.logsigmoid(x)
        return x


class SurvivalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt_indicator, gt_time):
        B = pred.shape[0]
        N = gt_indicator.sum()
        event_preds = pred[gt_indicator]
        risk_sets = gt_time.unsqueeze(0).repeat(B, 1) > gt_time.unsqueeze(0).T
        risk_sets = risk_sets[gt_indicator]
        risk_preds = torch.zeros((N, B))
        risk_preds[risk_sets] = torch.exp(pred).T.repeat(N, 1)[risk_sets]
        logsum = torch.log(risk_preds.sum(dim=1))
        log_partial_lhood = (event_preds.squeeze() - logsum).mean()

        return -log_partial_lhood
