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


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join([word for word in text.split() if word not in (STOPWORDS)])  # delete stopwords from text

    return text



X_train = [text_prepare(x) for x in X_train]
X_val = [text_prepare(x) for x in X_val]
X_test = [text_prepare(x) for x in X_test]

tags_counts = {}
# Dictionary of all words from train corpus with their counts.
words_counts = {}

# ' '.join(X_train[:3])
rx = re.compile('([\[\],\'\'])')
rx.sub('', ' '.join(str(v) for v in y_train[:10])).split(' ')


from collections import Counter
tags_counts = Counter(rx.sub('', ' '.join(str(v) for v in y_train)).split(' '))
words_counts = Counter(' '.join(X_train).split(' '))




sorted_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:5000]




DICT_SIZE = 5000
sorted_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:DICT_SIZE]
INDEX_TO_WORDS = dict(enumerate(list(dict(sorted_words).keys())))
WORDS_TO_INDEX = dict(zip(INDEX_TO_WORDS.values(), INDEX_TO_WORDS.keys()))

ALL_WORDS = WORDS_TO_INDEX.keys()


def my_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary

        return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)
    for each_word in text.split():
        # print(each_word)
        if (each_word in words_to_index.keys()):
            result_vector[words_to_index.get(each_word)] = result_vector[words_to_index.get(each_word)] + 1

    return result_vector

def test_my_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (my_bag_of_words(ex, words_to_index, 4) != ans).any():
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'

print(test_my_bag_of_words())

from scipy import sparse as sp_sparse

X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])
print('X_train shape ', X_train_mybag.shape)
print('X_val shape ', X_val_mybag.shape)
print('X_test shape ', X_test_mybag.shape)


from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result


    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), use_idf=True, token_pattern=r'(\S+)')

    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_test = tfidf_vectorizer.transform(X_test)
    X_val = tfidf_vectorizer.transform(X_val)

    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_


X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}

print('c#' in tfidf_reversed_vocab.values())
print('c++' in tfidf_reversed_vocab.values())



from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
y_train = mlb.fit_transform(y_train)
y_val = mlb.fit_transform(y_val)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


# from sklearn.utils import class_weight
# # In order to calculate the class weight do the following
#
# class_weights = class_weight.compute_class_weight('balanced',
#                                                  np.unique(y_train),
#                                                  y_train)


from sklearn.preprocessing import StandardScaler

def train_classifier(X_train, y_train):
    """
      X_train, y_train — training data

      return: trained classifier
    """
    # scaler = StandardScaler(with_mean=False)
    # scaler.fit(X_train)

    # Create and fit LogisticRegression wraped into OneVsRestClassifier.
    clf = OneVsRestClassifier(LogisticRegression(C=1, multi_class='ovr', max_iter=1000, solver='liblinear',
                                                 class_weight='balanced'))
    clf.fit(X_train, y_train)

    return clf #, scaler


classifier_mybag = train_classifier(X_train_mybag, y_train)
classifier_tfidf = train_classifier(X_train_tfidf, y_train)

y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)


y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
y_val_inversed = mlb.inverse_transform(y_val)
for i in range(3):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        X_val[i],
        ','.join(y_val_inversed[i]),
        ','.join(y_val_pred_inversed[i])
    ))



from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def print_evaluation_scores(y_val, predicted):
    print(f1_score(y_val, predicted, average="weighted"))
    print(precision_score(y_val, predicted, average="weighted"))
    print(recall_score(y_val, predicted, average="weighted"))
    print(accuracy_score(y_val, predicted))


print('Bag-of-words')
print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
print('Tfidf')
print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)

