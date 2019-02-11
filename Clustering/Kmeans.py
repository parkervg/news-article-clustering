#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:03:36 2019

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
data_with_labels = codecs.open("/Users/parkerglenn/Desktop/DataScience/Article_Clustering/Google_Drive/Article_Classification122.csv")

df = pd.read_csv(data)
labels_df= pd.read_csv(data_with_labels)

for col in df:
    print ("column", col, ":", type(df[col][0]))
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


from TFIDF import TFIDF
#creates TFIDF matrix
TFIDF(corpus)

##############################################################################
###################KMEANS#####################################################
##############################################################################
from sklearn.externals import joblib
#Loads my pre-existing kmeans model
#Saves the model you just made
#joblib.dump(km, '700_No_Ngram.pkl')
km = joblib.load("/Users/parkerglenn/Desktop/DataScience/Article_Clustering/KMeans_Cluster_Models/350_no_Ngram.pkl")
clusters = km.labels_.tolist()



#Only to create a new kmeans model
from sklearn.cluster import KMeans
num_clusters = 350
km = KMeans(n_clusters = num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()


articles = {"title": titles, "date": new_df["date"], "cluster": clusters, "content": new_df["content"], "event": events[:1000]}
frame = pd.DataFrame(articles, index = [clusters] , columns = ['title', 'date', 'cluster', 'content', "event"])
frame['cluster'].value_counts()

order_centroids = km.cluster_centers_.argsort()[:, ::-1] 


from collections import Counter
#Creates a count dict (success) to see how many instances of the same event are clustered together
for i in clusters[:100]:
    print("Cluster %d words:" % i, end='')
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()
    print()
    print()
    counts = []
    for event in frame.loc[i]["event"].values.tolist():
        counts.append(event)
    counts = dict(Counter(counts))    
    print(counts)
    print()
    print()


#Allows you to zoom in on a specific cluster, see what words make that cluster unique
for i in clusters:
    if i == 244: #Change 2 to the cluster 
        print("Cluster %d words:" % i, end='')
        for ind in order_centroids[i, :5]: #replace 20 with n words per cluster
            print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
        counts = []
        for event in frame.ix[i]["event"].values.tolist():
            counts.append(event)
        counts = dict(Counter(counts))    
        print(counts)
        print()
        print()
        
        


