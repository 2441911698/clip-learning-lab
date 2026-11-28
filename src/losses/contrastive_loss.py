from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPContrastiveLoss(nn.Module):
    def __init__(self, init_temperature: float = 0.07, max_logit_scale: float = 100.0) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))
        # 余弦相似度的值域被限制在[-1, 1],直接送入 Softmax 会导致概率分布过于平缓,引入一个可学习的参数logit_scale
        self.max_logit_scale = max_logit_scale

    def forward(self,image_embeddings: torch.Tensor,text_embeddings: torch.Tensor, ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_embeddings.shape != text_embeddings.shape:
            raise ValueError(f"Shape mismatch: image {image_embeddings.shape} vs text {text_embeddings.shape}")

        image_emb = F.normalize(image_embeddings, p = 2,dim = -1)
        text_emb = F.normalize(text_embeddings, p = 2, dim = -1)

        logit_scale = self.logit_scale.exp().clamp(max = self.max_logit_scale)

        logits_per_image = logit_scale * (image_emb @ text_emb.T)
        logits_per_text = logits_per_image.T

        batch_size = image_emb.size(0)
        labels = torch.arange(batch_size, device=image_emb.device)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i2t + loss_t2i)/2.0

        return loss, logits_per_image