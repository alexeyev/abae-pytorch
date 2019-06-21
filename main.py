import torch
from model import ABAE
from reader import get_centroids, get_w2v, read_data_tensors

# todo: args conf


if __name__ == "__main__":

    w2v_path = "word_vectors/reviews_Electronics_5.json.txt.w2v"
    text_path = "reviews_Electronics_5.json.txt"

    BATCH = 50
    ASPECTS = 30
    wv_dim = 200
    maxlen = 201
    neg_number = 5
    scans = 2

    y = torch.zeros(BATCH, 1)

    model = ABAE(wv_dim=wv_dim,
                 asp_count=ASPECTS,
                 init_aspects_matrix=get_centroids(get_w2v(w2v_path), aspects_count=ASPECTS))
    print(model)

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters())

    for t in range(scans):

        for item_number, (x, texts) in enumerate(read_data_tensors(text_path, w2v_path, batch_size=BATCH, maxlen=maxlen)):

            x = torch.from_numpy(x)
            negative_samples = torch.stack(tuple([x[torch.randperm(x.shape[0])[:neg_number]] for _ in range(BATCH)]))

            y_pred = model(x, negative_samples)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if item_number % 100 == 0:

                print(item_number, "batches")

                for i, aspect in enumerate(model.get_aspect_words(get_w2v(w2v_path))):
                    print(i, " ".join(aspect))

                print("Loss:", loss.item())
                print()
