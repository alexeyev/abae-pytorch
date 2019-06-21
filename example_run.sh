#!/usr/bin/env bash

### this may take a while

# wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
# gunzip reviews_Electronics_5.json.gz
# python3 custom_format_converter.py reviews_Electronics_5.json

### data prepared, training word vectors

#python3 word2vec.py reviews_Electronics_5.json.txt
python3 main.py