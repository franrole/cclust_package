import re
import string
from sklearn.datasets import fetch_20newsgroups
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from collections import Counter
from pprint import pprint

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords


DATA_PATH = "<PATH>/"

stopword_list = nltk.corpus.stopwords.words('english')


def get_texts_and_labels(corpus_name) :
  if corpus_name == "ng5" :
    categories = ['rec.motorcycles','rec.sport.baseball','comp.graphics','sci.space',
'talk.politics.mideast']
    ng5 = fetch_20newsgroups(subset='all', categories=categories, shuffle=True,remove=('headers', 'footers', 'quotes'))
    texts = ng5.data
    labels = ng5.target
    label_names = ng5.target_names 
  elif corpus_name == "classic3" :
    classic3 = pd.read_json(DATA_PATH  +  "classic3.json", lines = True)
    texts = classic3['raw']
    string_labels = classic3['label']
    lab_encoder = LabelEncoder()
    labels = lab_encoder.fit_transform(string_labels)
    label_names = np.unique(string_labels)
  elif corpus_filename == "classic4" :
    classic4 = pd.read_json(DATA_PATH  +  "classic4.json", lines = True)
    texts = classic4['raw']
    string_labels = classic4['label']
    lab_encoder = LabelEncoder()
    labels = lab_encoder.fit_transform(string_labels)
    label_names = np.unique(string_labels)
  elif corpus_name == "r8" :
    train_r8 = pd.read_csv(DATA_PATH  + "r8-train-all-terms.txt", sep = '\t', header = None)
    test_r8 = pd.read_csv(DATA_PATH  + "r8-test-all-terms.txt", sep = '\t', header = None)
    all_r8 = pd.DataFrame(np.vstack((train_r8,test_r8)))
    texts = data_r8[1]
    string_labels = data_r8[0]
    lab_encoder = LabelEncoder()
    labels = lab_encoder.fit_transform(string_labels)
    label_names = np.unique(string_labels)

  ###### Handle r40, r52, webkb (check empty docs ?)

  return texts, labels, label_names

def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):
        if doc != "":
            filtered_corpus.append(doc)
            filtered_labels.append(label)
    return filtered_corpus, filtered_labels
  
def remove_special_characters(text):
  tokens = tokenize_text(text)
  pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
  filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
  filtered_text = ' '.join(filtered_tokens)
  return filtered_text

def remove_stopwords(text):
  tokens = tokenize_text(text)
  filtered_tokens = [token for token in tokens if token not in stopword_list]
  filtered_text = ' '.join(filtered_tokens)
  return filtered_text


### Allow the user to choose a tokenizer (nltk, spacy, etc.)
def tokenize_text(text):
  tokens = nltk.word_tokenize(text)
  tokens = [token.strip() for token in tokens]
  return tokens

def clean_texts(texts, tokenize=True):
  """texts is a list of strings (documents"""
  cleaned_texts = []
  cleaned_tokenized_texts = []
  for text in texts:
    text = text.lower()
    text = remove_special_characters(text)
    text = remove_stopwords(text)
    cleaned_texts.append(text)
    if tokenize:
      text = tokenize_text(text)
      cleaned_tokenized_texts.append(text)
  return cleaned_texts, cleaned_tokenized_texts

### generic function for embedding docs using different models
def embed_documents(corpus_name, embedding_model) :
   texts, targets, labels = get_texts_and_labels(corpus_name)
   cleaned_texts, cleaned_tokenized_texts = clean_texts(texts, tokenize=True)
   if embedding_model == "word2vec" :
      word_model = Word2Vec(cleaned_tokenized_texts, size=50, window=10, min_count=1, workers=4, iter=50)
      ### I have the code for this one, so better to focus on the following two
   ### Other models 
   elif embedding_model == "doc2vec":
       pass
   elif embedding_model == "bert":
       pass


  

