from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.contrastive_loss import CLIPContrastiveLoss
from src.models.image_encoder import ImageEncoder
from src.models.text_encoder import TextEncoder

class CLIPModel(nn.Module):

    def __init__(
        self,
        embed_dim: int = 512,
        text_model_name: str = "distilbert-base-uncased",
        pretrained: bool = True,
        freeze_backbones: bool = False,
        init_temperature: float = 0.07,
        image_encoder: nn.Module | None = None,
        text_encoder: nn.Module | None = None,
    )->None:

        super().__init__()

        self.image_encoder = image_encoder or ImageEncoder(
            embed_dim = embed_dim,
            pretrained = pretrained,
            freeze_backbone = freeze_backbones,
        )

        self.text_encoder = text_encoder or TextEncoder(
            model_name=text_model_name,
            embed_dim = embed_dim,
            pretrained = pretrained,
            freeze_backbone = freeze_backbones,
        )

        self.criterion = CLIPContrastiveLoss(init_temperature=init_temperature)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        normalize: bool = True,
    )->dict[str, torch.Tensor]:

        image_emb = self.image_encoder(images)
        text_emb = self.text_encoder(input_ids = input_ids, attention_mask = attention_mask)

        loss, logits_per_image = self.criterion(image_emb, text_emb)

        return {
            "loss": loss,
            "logits_per_image": logits_per_image,
            "image_embeddings": F.normalize(image_emb, p = 2, dim = -1),
            "text_embeddings": F.normalize(text_emb, p = 2, dim = -1),
        }
