# -*- coding: utf-8 -*-
import json
from functools import lru_cache

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

tokenizer = TweetTokenizer(preserve_case=False)
lemmatizer = WordNetLemmatizer()
stops = set(stopwords.words("english"))


@lru_cache(1000000000)
def lemmatize(w: str):
    # caching the word-based lemmatizer to speed the process up
    return lemmatizer.lemmatize(w)


def read_amazon_format(path: str, sentence=True):
    """
        Reading Amazon dataset-like JSON files, splitting the reviews into sentences (or not),
        tokenizing, lemmatizing, filtering and saving into a text file

    :param path: a path to a filename
    :param sentence: whether to split the reviews into sentences
    """
    with open(path + ("" if sentence else "-full_text") + ".txt", "w+", encoding="utf-8") as wf:

        for line in tqdm(open(path, "r", encoding="utf-8")):
            # reading the text
            text = json.loads(line.strip())["reviewText"].replace("\n", " ")
            # splitting into sentences
            sentences = sent_tokenize(text)
            tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]

            # removing stopwords and non-alphanumeric tokens
            lemmatized_sentences = [[lemmatize(word) for word in s if not word in stops and str.isalpha(word)]
                                    for s in tokenized_sentences]

            for sentence in lemmatized_sentences:
                wf.write(" ".join(sentence) + "\n" if sentence else " ")

            if not sentence:
                wf.write("\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "reviews_Cell_Phones_and_Accessories_5.json"

    read_amazon_format(path, sentence=True)
