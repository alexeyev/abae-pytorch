import gensim
import numpy as np
from sklearn.cluster.k_means_ import MiniBatchKMeans


def read_data_batches(path, batch_size=50, minlength=5):
    batch = []

    for line in open(path):
        line = line.strip().split()
        if len(line) >= minlength:
            batch.append(line)
            if len(batch) >= batch_size:
                yield batch
                batch = []

    if len(batch) > 0:
        yield batch


def text2vectors(text, w2v_model, maxlen, vocabulary):

    acc_vecs = []

    for word in text:
        if word in w2v_model and (vocabulary is None or word in vocabulary):
            acc_vecs.append(w2v_model.wv[word])

    # padding for consistent length
    if len(acc_vecs) < maxlen:
        acc_vecs.extend([np.zeros(w2v_model.vector_size)] * (maxlen - len(acc_vecs)))

    return acc_vecs


def get_w2v(path):
    return gensim.models.Word2Vec.load(path)


def read_data_tensors(path, word_vectors_path=None,
                      batch_size=50, vocabulary=None,
                      maxlen=100, pad_value=0, minsentlength=5):
    w2v_model = get_w2v(word_vectors_path)

    for batch in read_data_batches(path, batch_size, minsentlength):
        batch_vecs = []
        batch_texts = []

        for text in batch:
            vectors_as_list = text2vectors(text, w2v_model, maxlen, vocabulary)
            batch_vecs.append(np.asarray(vectors_as_list[:maxlen], dtype=np.float32))
            batch_texts.append(text)

        yield np.stack(batch_vecs, axis=0), batch_texts


def get_centroids(w2v_model, aspects_count):

    km = MiniBatchKMeans(n_clusters=aspects_count, verbose=0, n_init=100)
    m = []

    for k in w2v_model.wv.vocab:
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
