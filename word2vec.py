# -*- coding: utf-8 -*-

import codecs
import sys

import gensim
from tqdm import tqdm


class Sentences(object):
    def __init__(self, filename: str):
        self.filename = filename

    def __iter__(self):
        for line in tqdm(codecs.open(self.filename, "r", encoding="utf-8"), self.filename):
            yield line.strip().split()


def main(path):
    sentences = Sentences(path)
    model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=5, workers=7, sg=1,
                                   negative=5, iter=1, max_vocab_size=20000)
    model.save("word_vectors/" + path + ".w2v")
    # model.wv.save_word2vec_format("word_vectors/" + domain + ".txt", binary=False)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "reviews_Electronics_5.json.txt"

    try:
        import os

        os.mkdir("word_vectors/")
    except:
        pass

    print("Training w2v on dataset", path)

    main(path)

    print("Training done.")

    model = gensim.models.Word2Vec.load("word_vectors/" + path + ".w2v")

    for word in ["he", "love", "looks", "buy", "laptop"]:
        if word in model.wv.vocab:
            print(word, [w for w, c in model.wv.similar_by_word(word=word)])
        else:
            print(word, "not in vocab")
