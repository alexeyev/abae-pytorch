# -*- coding: utf-8 -*-
import logging

import hydra
import numpy as np
import torch
import os

from model import ABAE
from reader import get_centroids, get_w2v, read_data_tensors

logger = logging.getLogger(__name__)


@hydra.main("configs", "config")
def main(cfg):
    w2v_model = get_w2v(os.path.join(hydra.utils.get_original_cwd(), cfg.embeddings.path))
    wv_dim = w2v_model.vector_size
    y = torch.zeros((cfg.model.batch_size, 1))

    model = ABAE(wv_dim=wv_dim,
                 asp_count=cfg.model.aspects_number,
                 init_aspects_matrix=get_centroids(w2v_model, aspects_count=cfg.model.aspects_number))
    logger.debug(str(model))

    criterion = torch.nn.MSELoss(reduction="sum")

    if cfg.optimizer.name == "adam":
        optimizer = torch.optim.Adam(model.parameters())
    elif cfg.optimizer.name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.optimizer.learning_rate)
    elif cfg.optimizer.name == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters())
    elif cfg.optimizer.name == "asgd":
        optimizer = torch.optim.ASGD(model.parameters(), lr=cfg.optimizer.learning_rate)
    else:
        raise Exception("Optimizer '%s' is not supported" % cfg.optimizer.name)

    for t in range(cfg.model.epochs):

        logger.debug("Epoch %d/%d" % (t + 1, cfg.model.epochs))

        data_iterator = read_data_tensors(os.path.join(hydra.utils.get_original_cwd(), cfg.data.path),
                                          os.path.join(hydra.utils.get_original_cwd(), cfg.embeddings.path),
                                          batch_size=cfg.model.batch_size, maxlen=cfg.model.max_len)

        for item_number, (x, texts) in enumerate(data_iterator):
            if x.shape[0] < cfg.model.batch_size:  # pad with 0 if smaller than batch size
                x = np.pad(x, ((0, cfg.model.batch_size - x.shape[0]), (0, 0), (0, 0)))

            x = torch.from_numpy(x)

            # extracting bad samples from the very same batch; not sure if this is OK, so todo
            negative_samples = torch.stack(
                tuple([x[torch.randperm(x.shape[0])[:cfg.model.negative_samples]]
                       for _ in range(cfg.model.batch_size)]))

            # prediction
            y_pred = model(x, negative_samples)

            # error computation
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if item_number % cfg.model.log_progress_steps == 0:

                logger.info("%d batches, and LR: %.5f" % (item_number, optimizer.param_groups[0]['lr']))

                for i, aspect in enumerate(model.get_aspect_words(w2v_model, logger)):
                    logger.info("[%d] %s" % (i + 1, " ".join([a for a in aspect])))

                logger.info("Loss: %.4f" % loss.item())


if __name__ == "__main__":
    main()
