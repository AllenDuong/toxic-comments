'''
    File: data_exploration.py
    Description: 
        Explores the Training and Testing Data Sets
        Print Graphics About Their Contents
    Author: Allen Duong
'''
##################### Setup #####################
from wordcloud import WordCloud, STOPWORDS
from nltk.stem.wordnet import WordNetLemmatizer
from PIL import Image
import keras
import tensorflow
from nltk import pos_tag, word_tokenize
from nltk import WordNetLemmatizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import time
import nltk
import re
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # Ignore Warnings

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

##################### Plot Functions #####################
# Plot Settings
colour = sns.color_palette()
sns.set_style("darkgrid")
eng_stopwords = set(stopwords.words("english"))


def label_freq(data):
    '''
        Plot The Number of Occurences of Different Labels
    '''
    x = data.iloc[:, 2:].sum()
    ax = sns.barplot(x.index, x.values, alpha=0.8)
    plt.title("Bar Plot: Number of Label Occurences")
    plt.ylabel('Total Occurrences')
    plt.xlabel('Type of Comment')
    rects = ax.patches
    labels = x.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2,
                height + 5, label, ha='center', va='bottom')

    plt.figure(figsize=(10, 6))
    plt.show()


def multi_label(data):
    '''
        Plot The Number of Occurences of Multiple Labels
    '''
    x = rowsums.value_counts()
    ax = sns.barplot(x.index, x.values, alpha=0.8, color=colour[4])
    plt.title("Multiple tags per comment")
    plt.ylabel('No of Occurrences')
    plt.xlabel('No of tags ')

    # adding the text labels
    rects = ax.patches
    labels = x.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2,
                height + 5, label, ha='center', va='bottom')
    plt.figure(figsize=(10, 6))
    plt.show()


def relations(data):
    temp_df = train.iloc[:, 2:-1]
    corr = temp_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, xticklabels=corr.columns.values,
                yticklabels=corr.columns.values, annot=True)


def clouds(label, data):
    '''
        Print a Wordcloud For a Set of Comments
    '''

    wc = WordCloud(background_color="black",
                   max_words=2000, stopwords=eng_stopwords)
    wc.generate(" ".join(data))
    plt.figure(figsize=(20, 10))
    plt.axis("off")
    plt.title("Words frequented in {} Comments".format(label), fontsize=20)
    plt.imshow(wc.recolor(colormap='viridis', random_state=17), alpha=0.98)
    plt.show()


##################### Exploratory Data Analysis #####################
# Load Data
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# Mark Non-Tagged Comments as Clean
rowsums = train.iloc[:, 2:].sum(axis=1)
train['clean'] = (rowsums == 0)

# Build Bags
clean = train[train.clean == True]
bag_clean = clean.comment_text.values

toxic = train[train.toxic == 1]
bag_toxic = toxic.comment_text.values

severe_toxic = train[train.severe_toxic == 1]
bag_severe_toxic = severe_toxic.comment_text.values

obscene = train[train.obscene == 1]
bag_obscene = obscene.comment_text.values

insult = train[train.insult == 1]
bag_insult = insult.comment_text.values

threat = train[train.threat == 1]
bag_threat = threat.comment_text.values

identity_hate = train[train.identity_hate == 1]
bag_identity_hate = identity_hate.comment_text.values

# Count Occurences of Each Label
label_freq(train)

# Count Multi-Labels
multi_label(train)

# When Multi-Labeled, What is Getting Grouped Together
relations(train)

# To segregate data into bags to understand whihch words for which label are the most important.
labels = ['clean', 'toxic', 'severe_toxic',
          'obscene', 'threat', 'insult', 'identity_hate']
data_bag = [bag_clean, bag_toxic, bag_severe_toxic,
            bag_obscene, bag_threat, bag_insult, bag_identity_hate]
for i, bag in enumerate(data_bag):
    clouds(labels[i], bag)
