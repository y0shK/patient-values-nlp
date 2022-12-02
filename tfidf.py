"""
tfidf.py

Conduct TF-IDF algorithm on review contents to find most frequent words, making sure to penalize words that show up frequently that are not specific or informative 
"""

from itertools import islice

import json
import nltk
import random
import sklearn
import os
import string
import ssl
import collections
import nltk.metrics
import numpy as np

from nltk.metrics.scores import (precision, recall)
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import precision_score, recall_score, f1_score

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# perform preprocessing
# tokenize corpus & remove stopwords

nltk.download('punkt')
from nltk import word_tokenize
nltk.download("stopwords")
from nltk.corpus import stopwords

# create customized data structure for logistic regression
# e.g., {businessID1: [name, review, rating], ...}
# data is dictionary - key is business id, value is list [name, review, rating]

reviews = {}

# insert json file here
path = 'my_health_json_reviews.json'
count = 0 # counts every line

# put every different element of the txt file into separate list

# set filepath and load data

# https://www.programiz.com/python-programming/json
with open(path, encoding="utf8") as f:
    data = json.load(f)

# begin TF-IDF analysis with sklearn library

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def get_tfidf_matrix(corpus):

    # fit and transform the review contents into a tf-idf vector
    v = TfidfVectorizer()
    tfidf_matrix = v.fit_transform(corpus) # fit_transform is fitting and transforming on corpus

    # transform review contents to get most informative feature names
    response = v.transform(corpus)

    # https://stackoverflow.com/questions/34232190/scikit-learn-tfidfvectorizer-how-to-get-top-n-terms-with-highest-tf-idf-score
    feature_array = np.array(v.get_feature_names_out())
    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]

    n = 5
    top_n = feature_array[tfidf_sorting][:n]

    print("Top N words:")
    print(top_n)

    return tfidf_matrix

# go through each of the reviews by looping through dictionary
# get review from second position in tuple, index 1
# input data structure: ['reviewContent1', 'reviewContent2', 'reviewContent3', ...]

review_contents = []
for k, v in data.items():
    review_contents.append(data[k][1])

# matrix output of tf-idf:
# https://stackoverflow.com/questions/46959801/understanding-the-matrix-output-of-tfidfvectorizer-in-sklearn

print(get_tfidf_matrix(review_contents))
