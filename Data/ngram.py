import csv
import re
import os
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


propTestData = 0.1
classes = ['alignment', 'monitoring', 'nonSafety', 'robustness', 'systemic']
for filename in classes:
    
docs_train = open('Processed')
def ngram_classifier (docs_train, y_train, docs_test, ngram_range):

    tfidfvec = TfidfVectorizer(stop_words = "english",
                                analyzer = 'word',
                                lowercase = True,
                                use_idf = True,
                                ngram_range = ngram_range)
    
    X_train = tfidfvec.fit_transform(docs_train)

    X_test = tfidfvec.transform(docs_test)
    
    
    clf = SGDClassifier(loss = "hinge", penalty = "l1")
    
    clf.fit(X_train, y_train)
    
    prediction = clf.predict(X_test)
    
    return prediction