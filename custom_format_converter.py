import json
from functools import lru_cache

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from tqdm import tqdm

tokenizer = TweetTokenizer(preserve_case=False)
lemmatizer = WordNetLemmatizer()
stops = set(stopwords.words("english"))


@lru_cache(1000000000)
def lemmatize(w):
    return lemmatizer.lemmatize(w)


def read_amazon_format(path):

    with open(path + ".txt", "w+") as wf:

        for line in tqdm(open(path)):
            text = json.loads(line.strip())["reviewText"].replace("\n", " ")
            sentences = sent_tokenize(text)
            tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
            lemmatized_sentences = [[lemmatize(word) for word in s if not word in stops and str.isalpha(word)]
                                    for s in tokenized_sentences]

            for sentence in lemmatized_sentences:
                wf.write(" ".join(sentence) + "\n")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "reviews_Electronics_5.json"

    read_amazon_format(sys.argv[1])