'''
    File: improvement_02.py
    Description: 
        Implements Long short-term memory (LSTM), an RNN architecture, to Classify Toxic Text
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
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix # For ROC AUC

# For LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import initializers, regularizers, constraints, optimizers, layers


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

    # Get Training and Testing Features 
    y_train = mdl_train[CLASSES]
    y_test = mdl_test[CLASSES]

    ############### LSTM RNN ###############
    # Initialize Variables
    Y = mdl_train[CLASSES].values
    train_sentences = mdl_train['comment_text']
    test_sentences = mdl_test['comment_text']

    embed_size = 128        # Size of Each Word Vector
    max_features = 20000    # Number of Unique Words to Use (i.e # Rows in Embedding Vector)
    maxlen = 200            # Max Number of Words in a Comment to Use

    # Tokenize + Pad the Datasets
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_sentences))
    tokenized_train = tokenizer.texts_to_sequences(train_sentences)
    tokenized_test = tokenizer.texts_to_sequences(test_sentences)
    X_t = pad_sequences(tokenized_train, maxlen=maxlen)
    X_te = pad_sequences(tokenized_test, maxlen=maxlen)

    # Define the LSTM Model
    def get_model():
        # Input Layer: Shape is Equal to the Max Length of Features Defined Above
        inp = Input(shape=(maxlen, ))

        # Embedding Layer: Basically a Dimensionality Reduction of the Sentences into a Defined Feature Space
        x = Embedding(max_features, embed_size)(inp)

        # Output Size as a Parameter (Default=60)
        output_size = 60 
        x = LSTM(output_size, return_sequences=True, name='LSTM_Layer')(x)

        x = GlobalMaxPool1D()(x)

        # Dropout Layer
        percent_drop = 0.15 #.10
        x = Dropout(percent_drop)(x)

        # Dense Layer
        dense_size = 50
        x = Dense(dense_size, activation="relu")(x)

        # Dropout Layer 2
        x = Dropout(percent_drop)(x)

        # Dense Output Activation Layer
        x = Dense(6, activation="sigmoid")(x)
        
        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        return model

    # Initialize Model
    model = get_model()
    print(model.summary())

    # Training the LSTM model
    batch_size = 32
    epochs = 2

    # Fit the model
    model.fit(X_t,Y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    # Predict 
    y_pred = model.predict(X_te, batch_size = 1024)

    # Build ROC Curve + Calculate AUC's
    d = {k:[] for k in y_test.columns.tolist()}
    plt.figure(0,figsize=(8,8)).clf()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    avg_auc = 0
    for i,class_ in enumerate(CLASSES):
        actual = y_test[class_]
        fpr, tpr, threshold = roc_curve(actual, y_pred[:,i])
        d[class_] = d[class_] + np.where(y_pred[:,i] >= threshold[np.argmax(tpr-fpr)],1,0).tolist()
        AUC = np.round(roc_auc_score(actual,y_pred[:,i]),2)
        print('CV Score for {} = {}'.format(class_, np.round(AUC, 4)))
        avg_auc = avg_auc + AUC
        plt.plot(fpr,tpr,label=class_+" AUC = "+str(AUC))
        plt.legend(loc="lower right")
        
    plt.title('LSTM | Mean AUC = {}'.format(np.round(float(avg_auc)/6.0,2)))
    print('LSTM | Mean AUC = {}'.format(np.round(float(avg_auc)/6.0,2)))
    plt.show()