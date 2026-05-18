import torch
import torch.optim as optim

from model import ABAE


def test_training_loop_runs(args, word_embeddings):
    model = ABAE(args, word_embeddings)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    torch.manual_seed(42)

    losses = []

    for _ in range(5):
        sentence = torch.randint(
            0,
            args.vocab_size,
            (args.batch_size, args.seq_len),
        )

        negative = torch.randint(
            0,
            args.vocab_size,
            (args.batch_size, args.seq_len),
        )

        optimizer.zero_grad()

        loss = model(sentence, negative)

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses)))


def test_model_save_and_reload(
    args,
    word_embeddings,
    tmp_path,
):
    model = ABAE(args, word_embeddings)

    path = tmp_path / "model.pth"

    torch.save(model.state_dict(), path)

    new_model = ABAE(args, word_embeddings)

    new_model.load_state_dict(torch.load(path))

    for p1, p2 in zip(
        model.parameters(),
        new_model.parameters(),
    ):
        assert torch.allclose(p1, p2)


def test_attention_forward(model, sample_batch):
    sentence, _ = sample_batch

    embeddings = model.word_embeddings(sentence)

    output = model.self_attention(embeddings)

    assert output.shape == (
        sentence.size(0),
        model.aspect_num,
        model.embedding_dim,
    )
