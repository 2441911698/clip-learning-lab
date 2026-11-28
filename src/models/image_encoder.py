from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):

    def __init__(self, embed_dim: int = 512, pretrained: bool = True, freeze_backbone: bool = True) -> None:
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights = weights)

        backbone_out_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()


        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projection = nn.Linear(backbone_out_dim, embed_dim, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # args images形状为 [B, 3, 224, 224] 的图像张量
        features = self.backbone(images)
        embeddings = self.projection(features)
        return embeddings
        # returns 形状为 [B, embed_dim] 的图像嵌入特征