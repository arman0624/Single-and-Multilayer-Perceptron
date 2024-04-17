## Single and Multilayer Perceptron

This project is an implementation of a simple linear perceptron model from scratch. Instead
of having to compute the derivative over the entire training set, the perceptron simply picks examples
in sequence, and tries to classify them given the current weight vector. If it gets them right, it simply
moves on to the next examples, otherwise it updates the weight vector with the difference of the feature
counts in the correct labels and in the prediction

There is also an implementation of a multilayer perceptron (MLP). Only pytorch is used for this model.

The repository contains training and development data from the Stanford Sentiment Treebank and
20 Newsgroups datasets, in the data/ directory. You will also find the test data for these two datasets
without the labels.

Sentiment classification is the task of determining the sentiment - often positive,
negative or neutral - expressed in a given text. In general, this requires using a variety of cues such as
the presence of emotionally charged words such as “vile” or “amazing,” while taking into account the full
context of word use or phenomena like negation or sarcasm. I've written classifiers
for the Stanford Sentiment Treebank dataset, which contains snippets taken from Rotten Tomatoes movie
reviews, where the sentiment is aligned directly with the review score. I trained classifiers on a
filtered version of the dataset containing only full sentences and with neutral reviews removed, reducing
the task to binary classification of positive and negative sentiment.

Newsgroup Classification uses 20 Newsgroups dataset, which contains 18,846 newsgroup documents written
on a variety of topics. I used the text of each document to predict its newsgroup. Unlike the binary
sentiment analysis task, each document could belong to one of twenty newsgroups. Additionally, many
of these newsgroups share similar themes, such as computing, science, or politics. However, the distributions 
of words across each of these newsgroups are also fairly distinctive. For example, a document that uses the names 
of weapons will likely be in talk.politics.guns, a document mentioning “computer” will probably be in a comp.* group, 
and if a document uses the word “ice,” it was likely written for rec.sport.hockey rather than talk.politics.mideast.

### Environment

It's highly recommended to use a virtual environment (e.g. conda, venv).

Example of virtual environment creation using conda:
```
conda create -n env_name python=3.10
conda activate env_name
python -m pip install -r requirements.txt
```

### Train and predict commands

```
python3 perceptron.py -d newsgroups -f full -m slp
python3 perceptron.py -d newsgroups -f bow -m slp
python3 perceptron.py -d newsgroups -f bi-gram -m slp
python3 perceptron.py -d newsgroups -f tfidf -m slp

python3 perceptron.py -d sst2 -f full -m slp
python3 perceptron.py -d sst2 -f bow -m slp
python3 perceptron.py -d sst2 -f bi-gram -m slp
python3 perceptron.py -d sst2 -f tfidf -m slp

python3 multilayer_perceptron.py -d newsgroups -f bow -m mlp
python3 multilayer_perceptron.py -d sst2 -f bow -m mlp
```

### Commands to run unittests

```
pytest tests/test_perceptron.py
pytest tests/test_multilayer_perceptron.py
```
