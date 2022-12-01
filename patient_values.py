"""
patient_values.py

Authors: Yash Karandikar, Courtney Saqueton, and Adrian Daues
Poster presenters: Yash Karandikar, Courtney Saqueton
Used in: "What Do Patients Value? Using Natural Language Processing to Define a Good Doctor's Visit
"""

# import NLP libraries like nltk

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

from nltk.metrics.scores import (precision, recall)
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import precision_score, recall_score, f1_score

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

"""
1. preprocessing
"""

# perform preprocessing
# tokenize corpus, remove stopwords, lemmatize

nltk.download('punkt')
from nltk import word_tokenize
nltk.download("stopwords")
from nltk.corpus import stopwords

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# create customized data structure for logistic regression
# e.g., {businessID1: [name, review, rating], ...}
# data is dictionary - key is business id, value is list [name, review, rating]

reviews = {}

# insert json file here
path = 'new_train_set_1billion.json'
count = 0 # counts every line

# put every different element of the txt file into separate list

# set filepath and load data

# https://www.programiz.com/python-programming/json
with open(path, encoding="utf8") as f:
    data = json.load(f)

# get positive and negative reviews
# i.e., only reviews with 4/5 stars (positive) or 1/2 stars (negative)
# also remove capitalization, puncutation, digits, \n

pos_reviews = []
neg_reviews = []

pos_unedited = []
neg_unedited = []

for key,value in data.items():
  if value[2] == 4.0 or value[2] == 5.0:
    review = value[1]
    pos_unedited.append(review)
    review = review.lower()
    review = "".join([char for char in review if char not in string.punctuation])
    review = "".join([char for char in review if not char.isdigit()])
    review = review.replace("\n", " ")
    pos_reviews.append(review)

  elif value[2] == 1.0 or value[2] == 2.0:
    review = value[1]
    neg_unedited.append(review)
    review = review.lower()
    review = "".join([char for char in review if char not in string.punctuation])
    review = "".join([char for char in review if not char.isdigit()])
    review = review.replace("\n", " ")
    neg_reviews.append(review)


# store reviews in a new variable to tie back to context using dependency parsing
pos_plus_neg_reviews = pos_reviews + neg_reviews
print(len(pos_plus_neg_reviews))

# tokenize
pos_words = []
neg_words = []

for i in pos_reviews:
    word_list = []
    pos_words += word_tokenize(i)
    
for i in neg_reviews:
    neg_words += word_tokenize(i)

print(pos_words)

# remove stopwords

stopwords = stopwords.words("english")

pos_updated = []
neg_updated = []

for word in pos_words:
  if word not in stopwords:
    pos_updated.append(word)

for word in neg_words:
  if word not in stopwords:
    neg_updated.append(word)

# list of all tokenized words
all_words = pos_updated + neg_updated

"""
2. Creating a customized data structure for dependency parsing
"""

# Function that creates dictionary of {word1:True, word2:False...} for all words in the review
def token_dictionary(review):
    review_dict = {}
    for token in all_words:
        if token in review.split():     
            review_dict[token] = True
        else:
            review_dict[token] = False
    return review_dict

# Call function on each review, append (dictionary, class) to a list.
list_of_dicts = []

for review in pos_reviews:
    pos_review_tuple = (token_dictionary(review), "pos")
    list_of_dicts.append(pos_review_tuple)

for review in neg_reviews:
    neg_review_tuple = (token_dictionary(review), "neg")
    list_of_dicts.append(neg_review_tuple)

# Function that goes through list of dictionaries of all words in each review, and converts boolean values into 0s and 1s.
def zeros_ones(dictionary):
  zeros_ones = []
  for key, val in dictionary.items():
    if val == True:
      zeros_ones.append(1)
    elif val == False:
      zeros_ones.append(0)
  return zeros_ones

tuple_list = []

for review in list_of_dicts: # review format: ({word1:True, word2:False, ...}, 'pos')
  tup = (zeros_ones(review[0]), review[1])
  tuple_list.append(tup)

# Randomize the order
random.shuffle(tuple_list)
shuffled_list = tuple_list

"""
3. Begin splitting for machine learning
"""

# Split into training and testing
training_ratio = 0.8
training = shuffled_list[0:(int)(training_ratio*len(shuffled_list))] # training is a list of dicts
testing = shuffled_list[(int)(training_ratio*len(shuffled_list)):] # testing is a list of dicts

X_train, y_train = [], []

for review in training:
  X_train.append(review[0])
  if review[1] == 'pos':
    y_train.append(1)
  elif review[1] == 'neg':
    y_train.append(0)

X_test, y_test = [], []
for review in testing:
  X_test.append(review[0])
  if review[1] == 'pos':
    y_test.append(1)
  elif review[1] == 'neg':
    y_test.append(0)


# Call on classifier

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter = 1_000_000).fit(X_train, y_train)

y_pred_train = logreg.predict(X_train)
y_pred = logreg.predict(X_test)
y_pred_probs = logreg.predict_proba(X_test)

print(y_pred)
y_pred_array = []
for i in y_pred:
  y_pred_array.append(i)

print(y_pred_array)

import numpy

# key is the coefficient of each word
# value is the index of each word in the original dictionary of words

coefficients = numpy.ndarray.tolist(logreg.coef_)
print(len(coefficients[0]))

tokens = {}

j = 0

for i in coefficients[0]:

  # retain words with strong intensity
  if i > .5 or i < -.5:
    tokens[i] = j
  j+=1

keys_list = list(list_of_dicts[0][0]) # keys are all the words

key_reviews = [] # list of reviews
key_words = []


for i in tokens:
  idx = tokens[i]
  word = keys_list[idx]
  key_words.append(word)

  k = 0
  for j, dct in enumerate(list_of_dicts):
    if dct[0][word] == True and pos_plus_neg_reviews[j] not in key_reviews:
      key_reviews.append(pos_plus_neg_reviews[j])

      k+=1

"""
4. calculate evaluation metrics for logistic regression
"""

# Calculate metrics
print('Precision score: ')
print(precision_score(y_test, y_pred_array, pos_label= 1))

print('Recall score: ')
print(recall_score(y_test, y_pred_array, pos_label= 1))

print('F-1 score: ')
print(f1_score(y_test, y_pred_array, pos_label= 1))

"""
5. use dependency parsing to tie logistic regression results back to the context in which they are used
"""

# load spacy for dependency parsing
import spacy
from nltk import Tree

# https://stackoverflow.com/questions/42824129/dependency-parsing-tree-in-spacy
# https://stackoverflow.com/questions/7443330/how-do-i-do-dependency-parsing-in-nltk
# https://stackoverflow.com/questions/36610179/how-to-get-the-dependency-tree-with-spacy
# https://stackoverflow.com/questions/39323325/can-i-find-subject-from-spacy-dependency-tree-using-nltk-in-python

en_nlp = spacy.load("en_core_web_sm")

# find sample spacy patterns that allow for context
# 'acomp', 'conj', 'attr', 'amod', 'advmod'
# attr nouns -> surgeon, doctor, practitioner, physician, Doctor, facility, pediatrician, surgery
dependency_tags = ['acomp']
pos_words_tagged = []
neg_words_tagged = []

# loop through reviews to find words associated with these tags
for review in pos_unedited:
  doc1 = en_nlp(review)

  for i in range(len(doc1)):
    if doc1[i].dep_ in dependency_tags:
      pos_words_tagged.append(str(doc1[i]))

pos_sorted = sorted(pos_words_tagged, key=pos_words_tagged.count, reverse=True)
pos_top = []
for word in pos_sorted:
  if word not in pos_top:
    pos_top.append(word)

for review in neg_unedited:
  doc1 = en_nlp(review)
  for i in range(len(doc1)):
    if doc1[i].dep_ in dependency_tags:
      neg_words_tagged.append(str(doc1[i]))

neg_sorted = sorted(neg_words_tagged, key=neg_words_tagged.count, reverse=True)
neg_top = []
for word in neg_sorted:
  if word not in neg_top:
    neg_top.append(word)

all_top = neg_top + pos_top

# see if keywords in acomp
acomp_words = []

for word in key_words:
  if word in all_top:
    acomp_words.append(word)

# add manual tags that are apparent to a human reader
manual_acomps = ['professional', 'easy', 'wonderful', 'kind', 'helpful', 'clean', 'horrible', 'thorough', 'friendly', 'comfortable', 'caring', 'terrible', 'efficient', 'unprofessional']

manual_acomps = manual_acomps + acomp_words # use both sets of words

# perform data structure manipulation for spacy dependency parsing
# islice goes through the iterable and stops at the nth position
# take(n, iterable) is meant to go through data.items() and give the first 100 key-value pairs from the realdata json file

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

# load in evaluation set here
# take a new evaluation set and use take(n, iterable) to get 100 reviews from it
# then, use dependency parsing to see if the words generated match up with our intuition found from train/test set
# this is evaluation that verifies the output of the words, not the metrics of the logistic regression performance

path = 'evaluation_set.json'

with open(path, encoding="utf8") as f:
    data = json.load(f)

realdata = take(100, data.items())
print(realdata)
print(type(realdata))

# preprocess evaluation set reviews
test_reviews = []

for value in realdata:
  review = value[1][1]
  review = review.lower()
  review = "".join([char for char in review if char not in string.punctuation])
  review = "".join([char for char in review if not char.isdigit()])
  review = review.replace("\n", " ")
  test_reviews.append(review)

# use dependency parsing on evaluation set to extract telling keywords from reviews
nlp = spacy.load("en_core_web_sm")

noun_phrase = []

for review in test_reviews:
  doc = nlp(review)
  for chunk in doc.noun_chunks:
    noun_phrase.append(chunk.text + ' | ' + chunk.root.text + ' | ' + chunk.root.dep_ + ' | ' + chunk.root.head.text)

good_phrases = []
for e in noun_phrase:
  for word in manual_acomps:
    if word in e:
      good_phrases.append(e)

# build a dictionary where key is the word and value is the frequency
dc = {}
for word in manual_acomps:
  dc[word] = 0

for phrase in good_phrases:
  for word in manual_acomps:
    if word in phrase:
      dc[word] += 1

