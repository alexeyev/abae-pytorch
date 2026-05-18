import logging
import multiprocessing
from pathlib import Path

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def train_word2vec(input_file, output_path="word_vectors/word2vec.model"):
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.INFO,
    )

    sentences = LineSentence(input_file)

    model = Word2Vec(
        sentences,
        vector_size=300,
        window=10,
        min_count=5,
        workers=multiprocessing.cpu_count(),
    )

    model.save(output_path)

    logging.info(f"Word2Vec model saved to {output_path}")


if __name__ == "__main__":
    import sys

    train_word2vec(sys.argv[1])
