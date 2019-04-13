#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 15:50:28 2019

@author: parkerglenn
"""
import os
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

os.chdir("/Users/parkerglenn/Desktop/DataScience/Article_Clustering")

df = pd.read_csv("/Users/parkerglenn/Desktop/DataScience/Article_Clustering/csv/all_GOOD_articles.csv")
labels_df= pd.read_csv("/Users/parkerglenn/Desktop/DataScience/Article_Clustering/Google_Drive/Article_ClassificationFINAL.csv")

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

links = []
for link in new_df["url"]:
    links.append(link)

import os
os.chdir("/Users/parkerglenn/Desktop/toolbox")
from NERTokenizer import NERTokenizer
n = NERTokenizer(upper=True)
toks = n.transform(corpus)


from CustomTFIDF_tfidf_manipulated import CustomTFIDF
m = CustomTFIDF(ents_rate = 6.368, person_rate = 2.263, julian = False)
matrix= m.transform(toks)

os.chdir("/Users/parkerglenn/Desktop/DataScience/Article_Clustering")
from SuccessMetrics import success

#########################################################
####################BIRCH################################
#########################################################
from sklearn.cluster import Birch
brc = Birch(n_clusters = 520)
brc.fit(matrix)

y_pred = brc.labels_.tolist()
success(brc, y_pred, matrix)


#########################################################
####################HAC##################################
#########################################################
from sklearn.cluster import AgglomerativeClustering
hac = AgglomerativeClustering(n_clusters=520, affinity = "euclidean")
hac.fit(matrix)
#dense_matrix = tfidf_matrix.todense()

#from sklearn.externals import joblib
#Saves the model you just made
#joblib.dump(hac, '350_euc_HAC_ENTS.pkl')
#hac = joblib.load("/Users/parkerglenn/Desktop/DataScience/Article_Clustering/HAC_Cluster_Models/350_euc_HAC.pkl")

y_pred = hac.labels_.tolist()
success(hac, y_pred, matrix)


#########################################################
####################KEMANS###############################
#########################################################
from sklearn.cluster import KMeans
num_clusters = 520
km = KMeans(n_clusters = num_clusters)
km.fit(matrix)

y_pred = km.labels_.tolist()
success(km, y_pred, matrix)





#########################################################
###############KMEANS CLUSTER EXPLORING##################
#########################################################
def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


totalvocab_stemmed = []
totalvocab_tokenized = []
for i in corpus:
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

#Let's you search with stemmed word to see original format of word
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')


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

