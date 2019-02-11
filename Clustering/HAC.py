#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 15:14:08 2019

@author: parkerglenn
"""

import os
import pandas as pd
import numpy as np
import sklearn.manifold
import matplotlib.pyplot as plt
import nltk
import regex as re
import re
import codecs
import csv
import glob
import multiprocessing
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
stemmer = SnowballStemmer("english")
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import seaborn as sns
import sklearn

os.chdir("/Users/parkerglenn/Desktop/DataScience/Article_Clustering")

data = codecs.open('/Users/parkerglenn/Desktop/DataScience/Article_Clustering/csv/all_GOOD_articles.csv', encoding = 'utf-8')
data_with_labels = codecs.open("/Users/parkerglenn/Desktop/DataScience/Article_Clustering/Google_Drive/Article_Classification26.csv")

df = pd.read_csv(data)
labels_df= pd.read_csv(data_with_labels)

#Deletes unnecessary columns
df = df.drop(df.columns[:12], axis = 1)
#Sets manageable range for working data set
new_df = df[5000:6000]
#Gets info in list form to be later called in kmeans part

corpus = []
for text in new_df['content']:
    corpus.append(text)

titles = []
for title in new_df["title"]:
    titles.append(str(title))
#labels_df starts at df[5000] so we're good on the matching of labels to content
events = []
for event in labels_df["Event"][:1000]:
    events.append(str(event))

import os
os.chdir("/Users/parkerglenn/Desktop/DataScience/Article_Clustering/Pre-Processing")
from TFIDF import TFIDF
tfidf_matrix = TFIDF(corpus)



####################################################################
##########################HAC#######################################
####################################################################
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
hac = AgglomerativeClustering(n_clusters=500, affinity = "euclidean")
dense_matrix = tfidf_matrix.todense()
hac.fit_predict(dense_matrix)

from sklearn.externals import joblib
#Saves the model you just made
joblib.dump(hac, '350_euc_HAC.pkl')

y_pred = list(hac.labels_)

os.chdir("/Users/parkerglenn/Desktop/DataScience/Article_Clustering")
from SuccessMetrics import success
success(hac, y_pred, tfidf_matrix)





"""Dendogram Making"""
import pylab
fig = pylab.figure(figsize=(120,100))
children = hac.children_
distance = np.arange(children.shape[0])
no_of_observations = np.arange(2, children.shape[0]+2)
linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
dendrogram(linkage_matrix, labels = (events), truncate_mode = "level", leaf_font_size = 8)
fig.show




"""ScatterPlot"""
import mpld3
from mpld3 import display
from sklearn.decomposition import PCA
coords = PCA(n_components=2).fit_transform(dense_matrix)

fig, ax = plt.subplots(figsize = (14,8))
np.random.seed(0)
ax.plot(coords[:, 0], coords[:, 1],
        'or', ms=10, alpha=0.2)
ax.set_title('Truncated SVD with Cluster Assignments', size=14)
ax.grid(color='lightgray', alpha=0.7)
for i, txt in enumerate(events):
    print(i)
    plt.annotate(txt + ", " + str(y_pred[i]), (coords[:, 0][i], coords[:, 1][i]))
mpld3.show(fig)



