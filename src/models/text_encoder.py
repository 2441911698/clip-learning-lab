from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

class TextEncoder(nn.Module):

    def __init__(
            self,
            model_name: str = "distilbert-base-uncased",
            embed_dim: int =512,
            pretrained: bool = True,
            freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(config)

        backbone_out_dim = self.backbone.config.hidden_size

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projection = nn.Linear(backbone_out_dim, embed_dim, bias = False)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        outputs = self.backbone(input_ids = input_ids, attention_mask = attention_mask)
        cls_feature = outputs.last_hidden_state[:,0,:]
        embeddings = self.projection(cls_feature)
        return embeddings