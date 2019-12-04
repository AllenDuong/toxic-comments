'''
    File: improvement_01.py
    Description:
        Implements XG Boost to Classify Toxic Text
        Prints ROC AUC for Each Class and an Overall Mean AUC
    Author: Allen Duong
'''

# General Imports
import sys
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix  # For ROC AUC

# For XGB
from pandas import Series
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier


# Global Variables
CLASSES = ['toxic', 'severe_toxic', 'obscene',
           'threat', 'insult', 'identity_hate']

# Functions

# Main Execution
if __name__ == '__main__':

    # Read Training and Test Dataset
    train = pd.read_csv('./data/train.csv').fillna(' ')
    test = pd.read_csv('./data/test.csv').fillna(' ')
    text = pd.concat(
        [train['comment_text'], test['comment_text']]).reset_index(drop=True)

    # Build Sparse Matrix of TFIDF (Term Frequency–Inverse Document Frequency)
    tfidf = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'[a-z]{3,}',
        stop_words='english',
        ngram_range=(1, 2),
        max_df=50000,
        max_features=300)

    tfidf.fit(text)
    train_features = tfidf.transform(train['comment_text'])

    # Get Training and Testing Features
    X_train, X_test, y_train, y_test = train_test_split(
        train_features.toarray(), train[CLASSES], test_size=0.3, random_state=0)

    ############### XG Boost ###############
    # Build ROC Curve + Calculate AUC's
    d = {k:[] for k in y_test.columns.tolist()}
    plt.figure(0,figsize=(8,8)).clf()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    avg_auc = 0
    for class_ in CLASSES:
        model = XGBClassifier(n_estimators=500,n_jobs=2,oob_score=True)
        model.fit(X_train, y_train[class_])
        prediction = model.predict_proba(X_test)
        actual = y_test[class_]
        fpr, tpr, threshold = roc_curve(actual,prediction[:,1])
        d[class_] = d[class_] + np.where(prediction[:,1]>=threshold[np.argmax(tpr-fpr)],1,0).tolist()
        AUC = np.round(roc_auc_score(actual,prediction[:,1]),2)
        print('CV Score for {} = {}'.format(class_, np.round(AUC, 4)))
        avg_auc = avg_auc + AUC
        plt.plot(fpr,tpr,label=class_+" AUC = "+str(AUC))
        plt.legend(loc="lower right")
    plt.title('XGBoost | Mean AUC = {}'.format(np.round(float(avg_auc)/6.0,2)))
    print('XGBoost | Mean AUC = {}\n'.format(np.round(float(avg_auc)/6.0,2)))
    plt.show()
