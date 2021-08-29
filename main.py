# -*- coding: utf-8 -*-
import numpy as np
import torch
import hydra
from model import ABAE
from reader import get_centroids, get_w2v, read_data_tensors


@hydra.main("configs", "config")
def main(cfg):

    w2v_model = get_w2v(cfg.embeddings.path)
    print(cfg)
    print(w2v_model)
    # wv_dim = w2v_model.vector_size
    # y = torch.zeros(args.batch_size, 1)
    #
    # model = ABAE(wv_dim=wv_dim,
    #              asp_count=args.aspects_number,
    #              init_aspects_matrix=get_centroids(w2v_model, aspects_count=args.aspects_number))
    # print(model)
    #
    # criterion = torch.nn.MSELoss(reduction="sum")
    #
    # optimizer = None
    # scheduler = None
    #
    # if args.optimizer == "adam":
    #     optimizer = torch.optim.Adam(model.parameters())
    # elif args.optimizer == "sgd":
    #     optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    # elif args.optimizer == "adagrad":
    #     optimizer = torch.optim.Adagrad(model.parameters())
    # elif args.optimizer == "asgd":
    #     optimizer = torch.optim.ASGD(model.parameters(), lr=0.05)
    # else:
    #     raise Exception("Optimizer '%s' is not supported" % args.optimizer)
    #
    # for t in range(args.epochs):
    #
    #     print("Epoch %d/%d" % (t + 1, args.epochs))
    #
    #     data_iterator = read_data_tensors(args.dataset_path, args.wv_path,
    #                                       batch_size=args.batch_size, maxlen=args.maxlen)
    #
    #     for item_number, (x, texts) in enumerate(data_iterator):
    #         if x.shape[0] < args.batch_size:  # pad with 0 if smaller than batch size
    #             x = np.pad(x, ((0, args.batch_size - x.shape[0]), (0, 0), (0, 0)))
    #
    #         x = torch.from_numpy(x)
    #
    #         # extracting bad samples from the very same batch; not sure if this is OK, so todo
    #         negative_samples = torch.stack(
    #             tuple([x[torch.randperm(x.shape[0])[:args.neg_samples]] for _ in range(args.batch_size)]))
    #
    #         # prediction
    #         y_pred = model(x, negative_samples)
    #
    #         # error computation
    #         loss = criterion(y_pred, y)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         if item_number % 1000 == 0:
    #
    #             print(item_number, "batches, and LR:", optimizer.param_groups[0]['lr'])
    #
    #             for i, aspect in enumerate(model.get_aspect_words(w2v_model)):
    #                 print(i + 1, " ".join([a for a in aspect]))
    #
    #             print("Loss:", loss.item())
    #             print()


if __name__ == "__main__":
    main()