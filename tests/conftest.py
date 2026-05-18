import pytest
import torch


@pytest.fixture
def args():
    class Args:
        aspect_num = 5
        embedding_learning = True
        dropout = 0.1
        orthogonal_penalty = 0.1
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        embedding_dim = 50

    return Args()


@pytest.fixture
def word_embeddings():
    torch.manual_seed(0)
    return torch.randn(100, 50)


@pytest.fixture
def model(args, word_embeddings):
    from model import ABAE

    return ABAE(args, word_embeddings)


@pytest.fixture
def sample_batch():
    sentence = torch.randint(0, 100, (4, 10))
    negative = torch.randint(0, 100, (4, 10))

    return sentence, negative
