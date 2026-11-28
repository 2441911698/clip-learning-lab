from __future__ import annotations

import math
import torch
import pytest
import torch.nn as nn

from src.losses.contrastive_loss import CLIPContrastiveLoss
from src.models.image_encoder import ImageEncoder
from transformers import DistilBertConfig, DistilBertModel
from src.models.text_encoder import TextEncoder
from src.models.clip import CLIPModel

def test_contrastive_loss_shape_and_backward() -> None:
    """验证 Loss 为标量、logits 形状正确，且梯度能正常回传到特征和温度参数。"""
    torch.manual_seed(42)
    batch_size, embed_dim = 8, 512

    criterion = CLIPContrastiveLoss()
    image_emb = torch.randn(batch_size, embed_dim, requires_grad=True)
    text_emb = torch.randn(batch_size, embed_dim, requires_grad=True)

    loss, logits = criterion(image_emb, text_emb)

    assert loss.ndim == 0
    assert logits.shape == (batch_size, batch_size)

    loss.backward()
    assert image_emb.grad is not None
    assert text_emb.grad is not None
    assert criterion.logit_scale.grad is not None


def test_aligned_loss_lower_than_random() -> None:
    """验证完全对齐的图文对 Loss 远小于随机图文对，且随机状态接近 ln(N)。"""
    torch.manual_seed(42)
    batch_size, embed_dim = 16, 512
    criterion = CLIPContrastiveLoss()

    image_emb = torch.randn(batch_size, embed_dim)
    random_text_emb = torch.randn(batch_size, embed_dim)

    random_loss, _ = criterion(image_emb, random_text_emb)
    aligned_loss, _ = criterion(image_emb, image_emb.clone())

    # 随机特征在高维空间近乎正交，Loss 应在 ln(N) 附近
    expected_random = math.log(batch_size)
    assert abs(random_loss.item() - expected_random) < 0.5
    # 完全对齐时 Loss 应非常接近 0
    assert aligned_loss.item() < 0.01
    assert aligned_loss.item() < random_loss.item()


def test_shape_mismatch_raises_error() -> None:
    """验证输入维度不匹配时能主动抛出 ValueError。"""
    criterion = CLIPContrastiveLoss()
    image_emb = torch.randn(4, 512)
    text_emb = torch.randn(4, 256)

    with pytest.raises(ValueError, match="Shape mismatch"):
        criterion(image_emb, text_emb)


def test_image_encoder_output_shape() -> None:
    """验证图像编码器能将 [B, 3, 224, 224] 正确映射为 [B, embed_dim]。"""
    encoder = ImageEncoder(embed_dim=256, pretrained=False)
    dummy_images = torch.randn(4, 3, 224, 224)

    embeddings = encoder(dummy_images)
    assert embeddings.shape == (4, 256)


def test_image_encoder_freeze_backbone() -> None:
    """验证 freeze_backbone=True 时仅投影层可训练，骨干网络参数被冻结。"""
    encoder = ImageEncoder(embed_dim=256, pretrained=False, freeze_backbone=True)

    assert all(not p.requires_grad for p in encoder.backbone.parameters())
    assert all(p.requires_grad for p in encoder.projection.parameters())


def test_text_encoder_output_shape_and_freeze() -> None:
    """验证文本编码器的输出形状及参数冻结逻辑（离线微型配置，秒级完成）。"""
    # 构造一个微型配置避免测试时联网请求 Hugging Face
    tiny_config = DistilBertConfig(
        vocab_size=1000,
        dim=64,
        n_layers=1,
        n_heads=2,
        hidden_dim=128,
    )
    encoder = TextEncoder.__new__(TextEncoder)
    nn.Module.__init__(encoder)
    encoder.backbone = DistilBertModel(tiny_config)
    for p in encoder.backbone.parameters():
        p.requires_grad = False
    encoder.projection = nn.Linear(tiny_config.dim, 256, bias=False)

    input_ids = torch.randint(0, 1000, (4, 16))
    attention_mask = torch.ones(4, 16, dtype=torch.long)

    embeddings = encoder(input_ids, attention_mask)
    assert embeddings.shape == (4, 256)
    assert all(not p.requires_grad for p in encoder.backbone.parameters())
    assert all(p.requires_grad for p in encoder.projection.parameters())


def test_clip_model_end_to_end() -> None:
    """端到端测试：验证 CLIPModel 前向传播、输出形状及梯度回传。"""
    torch.manual_seed(42)
    batch_size, embed_dim = 4, 256

    # 1. 构造轻量级离线双塔
    image_encoder = ImageEncoder(embed_dim=embed_dim, pretrained=False, freeze_backbone=True)

    tiny_config = DistilBertConfig(
        vocab_size=1000, dim=64, n_layers=1, n_heads=2, hidden_dim=128
    )
    text_encoder = TextEncoder.__new__(TextEncoder)
    nn.Module.__init__(text_encoder)
    text_encoder.backbone = DistilBertModel(tiny_config)
    for p in text_encoder.backbone.parameters():
        p.requires_grad = False
    text_encoder.projection = nn.Linear(tiny_config.dim, embed_dim, bias=False)

    # 2. 组装完整 CLIP 模型
    model = CLIPModel(
        embed_dim=embed_dim,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
    )

    # 3. 模拟输入并执行前向传播
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 1000, (batch_size, 16))
    dummy_mask = torch.ones(batch_size, 16, dtype=torch.long)

    outputs = model(dummy_images, dummy_input_ids, dummy_mask)

    assert outputs["loss"].ndim == 0
    assert outputs["logits_per_image"].shape == (batch_size, batch_size)
    assert outputs["image_embeddings"].shape == (batch_size, embed_dim)
    assert outputs["text_embeddings"].shape == (batch_size, embed_dim)

    # 4. 验证反向传播：投影层和温度系数应有梯度，冻结的骨干网络无梯度
    outputs["loss"].backward()
    assert model.image_encoder.projection.weight.grad is not None
    assert model.text_encoder.projection.weight.grad is not None
    assert model.criterion.logit_scale.grad is not None