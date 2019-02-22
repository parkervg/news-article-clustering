#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 12:00:18 2019

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
df = pd.read_csv("/Users/parkerglenn/Desktop/DataScience/Article_Clustering/csv/all_GOOD_articles.csv")
labels_df= pd.read_csv("/Users/parkerglenn/Desktop/DataScience/Article_Clustering/Google_Drive/Article_Classification122.csv")
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
    
import spacy
from spacy import gold   
from spacy.gold import iob_to_biluo
nlp = spacy.load('en_core_web_md', disable=['parser','tagger','textcat'])
nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)
english_stopwords = stopwords.words('english')
from sklearn.feature_extraction import stop_words

##############################################################################
###################HYPER-PARAMETER TESTING####################################
##############################################################################
from hyper_parameter_functions import tokenize_and_stem_NER, do_tfidf, HAC, success

hyper_params = pd.DataFrame(columns = ["ents_rate", "person_rate","f_score","s_score","samples used"])

for person_rate in np.linspace(1,7,20):
    for ents_rate in np.linspace(1,7,20):
        tfidf_matrix = do_tfidf(ents_rate, person_rate)
        HAC(tfidf_matrix)
        cols = [pd.Series([ents_rate,person_rate,f_score,s_score,num2],index=hyper_params.columns)]
        hyper_params = hyper_params.append(cols)

hyper_params.plot.scatter(x="f_score", y = "s_score")


f_scores = [x for x in hyper_params["f_score"]]
s_scores = [x for x in hyper_params["s_score"]]
person_rate = [x for x in hyper_params["person_rate"]]
ents_rate = [x for x in hyper_params["ents_rate"]]

cmap = pyplot.cm.cubehelix
dimensions = (20,20)
fig, ax = pyplot.subplots(figsize=dimensions)
sns.heatmap(dist, vmin = 0, vmax = 1, cmap = cmap).set_title("Tfidf Distances Between Articles", fontsize = 15)



import seaborn as sns

"""F1 Score distribution across hyperparameters
Best F1 Score: ents_rate = 1.63, person_rate = 2.57, f_score = .927 on 412 articles. BUT s_score is a measly .0788.
"""
f_data = pd.DataFrame({"Person Rate": [round(x,2) for x in person_rate], "Entity Rate":[round(x,2) for x in ents_rate], "Z": hyper_params.f_score})
f_data_pivoted = f_data.pivot("Person Rate","Entity Rate","Z")
ax1 = sns.heatmap(f_data_pivoted, cmap = plt.cm.hot)
cbar = ax1.collections[0].colorbar
cbar.set_label('F1 Score', labelpad=15)
ax1.invert_yaxis()
plt.show()

"""Silhouette Score distribution across hyperparameters

More direct correlation here than in F1 Score. Most notably, as the rates increase, so does S Score. This is due to 
the fact that the rating of words in articles becomes more radical; articles that are different from each other, i.e. share no 
entities, are now much more distant than before. Even more specifically, when the Entity Rate and Person Rate are dissimilar, the
S Score is the highest. Again, the same radical rating phenomena: By limiting the amount of weighted features, those that are weighted
make the article more of an outlier than before."""
s_data = pd.DataFrame({"Person Rate": [round(x,2) for x in person_rate], "Entity Rate":[round(x,2) for x in ents_rate], "Z": hyper_params.s_score})
s_data_pivoted = s_data.pivot("Person Rate","Entity Rate","Z")
ax2 = sns.heatmap(s_data_pivoted, cmap = plt.cm.hot)
cbar = ax2.collections[0].colorbar
cbar.set_label('Silhouette Score', labelpad=15)
ax2.invert_yaxis()
plt.show()



"""Cumulative Total
Highest: ents_rate = 6.368, person_rate = 2.263, f_score = 0.922, s_score = 0.093

"""
sf_data = pd.DataFrame({"Person Rate": [round(x,2) for x in person_rate], "Entity Rate":[round(x,2) for x in ents_rate], "Z": hyper_params.s_score + hyper_params.f_score})
sf_data_pivoted = sf_data.pivot("Person Rate","Entity Rate","Z")
ax3 = sns.heatmap(sf_data_pivoted, cmap = plt.cm.hot)
cbar = ax3.collections[0].colorbar
cbar.set_label('Composite Score', labelpad=15)
ax3.invert_yaxis()
plt.show()

"""Little thing to find best cumulative score and rates that achieved it."""
hightot = 0
for tup in enumerate(hyper_params["f_score"]):
    tot = tup[1] + hyper_params.loc[tup[0]]["s_score"]
    if hightot < tot:
        hightot = tot
        best_scores = (hyper_params.loc[tup[0]]["ents_rate"],hyper_params.loc[tup[0]]["person_rate"])
best_scores



 