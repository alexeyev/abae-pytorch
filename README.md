# ABAE-PyTorch

Yet another PyTorch implementation of the model described in the paper [**An Unsupervised Neural Attention Model for Aspect Extraction**](https://aclweb.org/anthology/papers/P/P17/P17-1036/) by He, Ruidan and  Lee, Wee Sun  and  Ng, Hwee Tou  and  Dahlmeier, Daniel, **ACL2017**.

**NOTA BENE**: now `gensim>=4.0.0` and `hydra` are required.

## Example

**For a working example of a whole pipeline please refer to `example_run.sh`**

Let's get some data:

```
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz
gunzip reviews_Cell_Phones_and_Accessories_5.json.gz    
python3 custom_format_converter.py reviews_Cell_Phones_and_Accessories_5.json
```

Then we need to train the word vectors:
    
```
python3 word2vec.py reviews_Cell_Phones_and_Accessories_5.json.txt
```
And run 

**TODO**: running with hydra params example is in progress
```
usage: main.py ...

```

For a working example of a whole pipeline please refer to `example_run.sh` 

I acknowledge the implementation is raw, code modification requests and issues are welcome.

## Thanks for contributions
[@alexdevmotion](https://github.com/alexdevmotion)

## TODOs

* Evaluation: PMI, NPMI, LCP, L1/L2/coord/cosine (Nikolenko SIGIR'16), ...
* Aspects prediction on text + visualization
* Saving the model, aspects, etc.

## How to cite

Not neccessary, but greatly appreciated, if you use this work.

```latex
@misc{abaepytorch2019alekseev,
  title     = {{alexeyev/abae-pytorch: ABAE, PyTorch implementation}},
  year      = {2019},
  url       = {https://github.com/alexeyev/abae-pytorch},
  language  = {english}
}
```

## Please also see

* Reference [Keras implementation](https://github.com/ruidan/Unsupervised-Aspect-Extraction)
* Updated [Keras version](https://github.com/madrugado/Attention-Based-Aspect-Extraction), compliant with Keras 2.x and Python 3.x.
