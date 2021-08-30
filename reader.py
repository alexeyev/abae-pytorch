# -*- coding: utf-8 -*-
import gensim
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def read_data_batches(path: str, batch_size: int=50, minlength: int=5):
    """
        Reading batched texts of given min. length
    :param path: path to the text file ``one line -- one normalized sentence''
    :return: batches iterator
    """
    batch = []

    for line in open(path, encoding="utf-8"):
        line = line.strip().split()

        # lines with less than `minlength` words are omitted
        if len(line) >= minlength:
            batch.append(line)
            if len(batch) >= batch_size:
                yield batch
                batch = []

    if len(batch) > 0:
        yield batch


def text2vectors(text: list, w2v_model, maxlen: int, vocabulary):
    """
        Token sequence -- to a list of word vectors;
        if token not in vocabulary, it is skipped; the rest of
        the slots up to `maxlen` are replaced with zeroes
    :param text: list of tokens
    :param w2v_model: gensim w2v model
    :param maxlen: max. length of the sentence; the rest is just cut away
    :return:
    """

    acc_vecs = []

    for word in text:
        if word in w2v_model.wv and (vocabulary is None or word in vocabulary):
            acc_vecs.append(w2v_model.wv[word])

    # padding for consistent length with ZERO vectors
    if len(acc_vecs) < maxlen:
        acc_vecs.extend([np.zeros(w2v_model.vector_size)] * (maxlen - len(acc_vecs)))

    return acc_vecs


def get_w2v(path):
    """
        Reading word2vec model given the path
    """
    return gensim.models.Word2Vec.load(path)


def read_data_tensors(path, word_vectors_path=None,
                      batch_size=50, vocabulary=None,
                      maxlen=100, pad_value=0, min_sent_length=5):
    """
        Data for training the NN -- from text file to word vectors sequences batches
    :param path:
    :param word_vectors_path:
    :param batch_size:
    :param vocabulary:
    :param maxlen:
    :param pad_value:
    :param minsentlength:
    :return:
    """
    w2v_model = get_w2v(word_vectors_path)

    for batch in read_data_batches(path, batch_size, min_sent_length):
        batch_vecs = []
        batch_texts = []

        for text in batch:
            vectors_as_list = text2vectors(text, w2v_model, maxlen, vocabulary)
            batch_vecs.append(np.asarray(vectors_as_list[:maxlen], dtype=np.float32))
            batch_texts.append(text)

        yield np.stack(batch_vecs, axis=0), batch_texts


def get_centroids(w2v_model, aspects_count):
    """
        Clustering all word vectors with K-means and returning L2-normalizes
        cluster centroids; used for ABAE aspects matrix initialization
    """

    km = MiniBatchKMeans(n_clusters=aspects_count, verbose=0, n_init=100)
    m = []

    for k in w2v_model.wv.key_to_index:
        m.append(w2v_model.wv[k])

    m = np.matrix(m)
    km.fit(m)
    clusters = km.cluster_centers_

    # L2 normalization
    norm_aspect_matrix = clusters / np.linalg.norm(clusters, axis=-1, keepdims=True)

    return norm_aspect_matrix


if __name__ == "__main__":

    for b in read_data_tensors("reviews_Electronics_5.json.txt", "word_vectors/reviews_Electronics_5.json.txt.w2v", batch_size=3):
        print(b[0].shape, b[1][:2])
