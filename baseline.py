'''
    File: baseline.py
    Description: 
        Implements a Naive Bayes SVM to Classify Toxic Text
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

# For Naive Bayes SVM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from scipy import sparse

# Global Variables
CLASSES = ['toxic', 'severe_toxic', 'obscene',
           'threat', 'insult', 'identity_hate']

# Functions


def random(self, split):
    ''' Selects a random fraction amount from self dataframe specified by split
    Parameters
    ----------
    self : pandas dataframe
    split : fraction of dataframe to sample
    Results
    ---------
    returns sampled dataframe
    '''
    return self.sample(frac=split)


# Main Execution
if __name__ == '__main__':

    # Read Training and Test Dataset
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    x = train.iloc[:, 2:].sum()

    # Mark Comments Without Any Tags as "clean"
    rowsums = train.iloc[:, 2:].sum(axis=1)
    train['clean'] = (rowsums == 0)

    # Split Training Set into 30% Clean Comments and Non-Clean Comments Reduce Class Imbalance
    mdl_test = train.groupby(['clean'], group_keys=False).apply(random, 0.3)
    mdl_train = train.loc[~train.index.isin(mdl_test.index)]

    ############### Naive Bayes SVM ###############
    text = pd.concat([mdl_train['comment_text'], mdl_test['comment_text'],
                      test['comment_text']]).reset_index(drop=True)

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

    # Get Training and Testing Features 
    x_train = tfidf.transform(mdl_train['comment_text'])
    x_test = tfidf.transform(mdl_test['comment_text'])
    y_train = mdl_train[CLASSES]
    y_test = mdl_test[CLASSES]

    # Basic Naive Bayes Feature Equation
    def pr(y_i, y):
        p = x_train[y==y_i].sum(0)
        return (p+1) / ((y==y_i).sum()+1)

    # Fit a Model for Each Dependent
    def get_mdl(y):
        y = y.values
        r = np.log(pr(1,y) / pr(0,y))
        
        # Cross Validation 3-Fold similar to Nikhit's implementation 
        logR = LogisticRegression(dual=True)
        m = GridSearchCV(estimator=logR, cv=3, param_grid={'C':[0.01,0.1,1,10]},scoring='roc_auc')
        
        x_nb = x_train.multiply(r)
        
        return m.fit(x_nb, y), r

    # Build ROC Curve + Calculate AUC's
    d = {k:[] for k in y_test.columns.tolist()}
    plt.figure(0,figsize=(8,8)).clf()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    avg_auc = 0
    for class_ in CLASSES:
        model,r = get_mdl(y_train[class_])
        print('CV Score for {} = {}'.format(class_, np.round(model.best_score_, 4)))
        prediction = model.predict_proba(x_test.multiply(r))
        actual = y_test[class_]
        fpr, tpr, threshold = roc_curve(actual,prediction[:,1])
        d[class_] = d[class_] + np.where(prediction[:,1]>=threshold[np.argmax(tpr-fpr)],1,0).tolist()
        AUC = np.round(roc_auc_score(actual,prediction[:,1]),2)
        avg_auc = avg_auc + AUC
        plt.plot(fpr,tpr,label=class_+" AUC = "+str(AUC))
        plt.legend(loc="lower right")
    plt.title('Naive Bayes SVM | Mean AUC = {}'.format(np.round(float(avg_auc)/6.0,2)))
    print('Naive Bayes SVM | Mean AUC = {}'.format(np.round(float(avg_auc)/6.0,2)))
    plt.show()



