#!/usr/bin/env bash

DATA_NAME=reviews_Cell_Phones_and_Accessories_5

if [ ! -f ./$DATA_NAME.json.txt ]; then
    echo "File not found! Downloading..."

    ### this may take a while
    wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/$DATA_NAME.json.gz
    gunzip $DATA_NAME.json.gz
    python custom_format_converter.py $DATA_NAME.json
    rm $DATA_NAME.json.gz $DATA_NAME.json
    mkdir word_vectors
fi

if [ ! -f ./word_vectors/$DATA_NAME.json.txt.w2v ]; then
    echo "Training custom word vectors..."
    python word2vec.py $DATA_NAME.json.txt
fi

echo "Training ABAE..."
echo "A working example is in progress... Please see 'main.py' code."
#python main.py -as 30 -d $DATA_NAME.json.txt -wv word_vectors/$DATA_NAME.json.txt.w2v