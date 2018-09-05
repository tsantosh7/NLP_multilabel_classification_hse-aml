# Predict tags for posts from StackOverflow. To solve this task you will use multilabel classification approach.
#
# Libraries
#
# Numpy — a package for scientific computing.
# Pandas — a library providing high-performance, easy-to-use data structures and data analysis tools for the Python
# scikit-learn — a tool for data mining and data analysis.
# NLTK — a platform to work with natural language.


import sys
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from ast import literal_eval
import pandas as pd
import numpy as np
import re

def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data

train = read_data('data/train.tsv')
validation = read_data('data/validation.tsv')
test = pd.read_csv('data/test.tsv', sep='\t')

X_train, y_train = train['title'].values, train['tags'].values
X_val, y_val = validation['title'].values, validation['tags'].values
X_test = test['title'].values



pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))

from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()

def tweet_cleaner(text):
    stripped = re.sub(combined_pat, '', text)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()

# to maintain the assignment ethics, I will write a new code in the function
def text_prepare(text):

    norm_text = text.lower()
    norm_text = re.sub(r"([\.\",\(\)!\?;:])", " \\1 ", norm_text)
    return norm_text


X_train = [text_prepare(x) for x in X_train]
X_val = [text_prepare(x) for x in X_val]
X_test = [text_prepare(x) for x in X_test]