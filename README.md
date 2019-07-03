# ABAE-PyTorch

Yet another PyTorch implementation of the model described in the paper [**An Unsupervised Neural Attention Model for Aspect Extraction**](https://aclweb.org/anthology/papers/P/P17/P17-1036/) by He, Ruidan and  Lee, Wee Sun  and  Ng, Hwee Tou  and  Dahlmeier, Daniel, **ACL2017**.

## Example

`
    wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
    gunzip reviews_Electronics_5.json.gz
    python3 custom_format_converter.py reviews_Electronics_5.json
`
    Then we need to train the word vectors:
`

python3 word2vec.py reviews_Electronics_5.json.txt
`
And run

`
    python3 main.py
`

I acknowledge the implementation is raw, code modification requests and issues are welcome.

## Please also see

* Reference [Keras implementation](https://github.com/ruidan/Unsupervised-Aspect-Extraction)
* Updated [Keras version](https://github.com/madrugado/Attention-Based-Aspect-Extraction), compliant with Keras 2.x and Python 3.x.
