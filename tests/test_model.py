import pytest
import torch

from model import SelfAttention


def test_self_attention_shape():
    attn = SelfAttention(hidden_dim=50, aspect_num=5)

    x = torch.randn(4, 10, 50)

    out = attn(x)

    assert out.shape == (4, 5, 50)


def test_model_forward_loss_scalar(model, sample_batch):
    sentence, negative = sample_batch

    loss = model(sentence, negative)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.requires_grad is True


def test_orthogonality_penalty(model, sample_batch):
    sentence, negative = sample_batch

    model.aspects_embeddings.data = torch.ones_like(model.aspects_embeddings)

    loss = model(sentence, negative)

    assert loss.item() > 0


def test_device_movement(model):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    model.to("cuda")

    assert next(model.parameters()).is_cuda
