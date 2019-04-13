#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:13:38 2019

@author: parkerglenn
"""
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from matplotlib import pyplot
import mpld3
from mpld3 import display
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pylab


###############################################################################
##################VISUALIZATION################################################
###############################################################################

#Date Distribution
out = pd.cut(df["joined"], bins = [20000000,20020000,20040000,20060000,20080000,20100000,20120000,20140000,20160000,20180000], include_lowest = True)
ax = out.value_counts(sort = False).plot.bar(rot = 0, color = "b", figsize = (20,10))
ax.set_ylim(bottom = 0, top = 450)
ax.set_xticklabels(x for x in ["2000 to 2002","2002 to 2004","2004 to 2006","2006 to 2008","2008 to 2010","2010 to 2012","2012 to 2014","2014 to 2016","2016 to 2018"])
plt.xlabel("Date Range", fontsize = 18)
plt.ylabel("Frequency", fontsize = 18)
plt.title("Date Distribution", fontsize = 25)
for i in ax.patches:
    ax.text(i.get_x() + .10, 5, str(i.get_height()), fontsize = 20, color = "black")
    

#Distance Heatmap
dist = 1 - cosine_similarity(matrix)

cmap = pyplot.cm.cubehelix
dimensions = (20,20)
fig, ax = pyplot.subplots(figsize=dimensions)
sns.heatmap(dist, vmin = 0, vmax = 1, cmap = cmap).set_title("Tfidf Distances Between Articles", fontsize = 15)

""" 
Notice hot spot around the 625:630 line. 
Those article titles:
['Dem Debate Blogging #1',
 'Dem Debate Blogging #2',
 'Dem Debate Blogging #3',
 'Dem Debate Blogging #4',
 'Dem Debate Blogging #5',
 'Dem Debate Blogging #6']


Around 90:160: large circle of relatively similar articles.
Reason: Iowa Caucuses.
"""


#Tfidf Matrix, in 2D SVD scatterplot 
svd = TruncatedSVD(n_components=2).fit(matrix)
data2D = svd.transform(matrix)
plt.title("Truncated SVD, 2 Components")
colors = rng.rand(1000)
plt.scatter(data2D[:,0], data2D[:,1], marker = "o", c = colors, cmap = "YlGnBu", s = 10)
"""
Seems to follow Zipf's law: Rank in corpus * number = constant
"""


######With clusters assigned as colors########
data2D = svd.transform(matrix)
kmeans = KMeans(n_clusters = 520)
kmeans.fit(data2D)
y_kmeans = kmeans.predict(data2D)
y_pred = kmeans.labels_.tolist()

success(kmeans,y_pred,matrix)

articles = {"title": titles, "date": new_df["date"], "cluster": y_pred, "content": new_df["content"], "event": events[:1000]}
frame = pd.DataFrame(articles, index = [y_pred] , columns = ['title', 'date', 'cluster', 'content', "event"])
frame['cluster'].value_counts()


"""Creates scalable points for cluster centers found within y_true.
The size of the center is dependent on how many events occur withing that specific cluster. """

centers = kmeans.cluster_centers_
fig, ax = plt.subplots(figsize = (14,8))
np.random.seed(0)
threshold = 4
for cluster, center in enumerate(centers):
    cluster+=1
    # Only maps cluster if it has more than "threshold" events in it
    if cluster in y_true and len(frame.loc[cluster]["event"].values.tolist()) > threshold:
        #Gets event name that the cluster center represents
        event = events[y_true.index(cluster)]
        #s scaled based on number of events in cluster
        ax.plot(center[0], center[1], markersize = float(len(frame.loc[cluster]["event"].values.tolist())),  marker = "o");
        plt.annotate(event, (center[0],center[1]))
ax.set_title('Cluster Centers with Predominant Event', size=14)
plt.show()

mpld3.show(fig)

# mpld3.save_html(fig, "Cluster_Centers.html")  
   

 
"""Zoomed out plot with colors"""
plt.title("Truncated SVD with Colored Clusters")
plt.scatter(data2D[:, 0], data2D[:, 1], c=y_kmeans, cmap = "tab20", s=30)
"""Example plot, zoomed in to visualize cluster centers"""

    
    
    
fig, ax = plt.subplots(figsize = (14,8))
np.random.seed(0)
ax.plot(data2D[:, 0], data2D[:, 1],
        'or', ms=10, alpha=0.2)
ax.set_title('Truncated SVD with Cluster Assignments', size=14)
ax.grid(color='lightgray', alpha=0.7)
for i, txt in enumerate(events):
    print(i)
    plt.annotate(txt + ", " + str(y_pred[i]), (data2D[:, 0][i], data2D[:, 1][i]))
mpld3.show(fig)






    
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=30, alpha=0.5, marker = 'x');
pyplot.axis(ymax = 0, ymin =-.25 , xmax = .2, xmin = .14)
plt.title("Truncated SVD with Cluster Centers")
pyplot.axis(ymax = 0, ymin =-.25 , xmax = .2, xmin = .14)
plt.scatter(data2D[:, 0], data2D[:, 1], c=y_kmeans, cmap = "tab20", s=30)




######Interactive ScatterPlot with SVD######
svd = TruncatedSVD(n_components=2).fit(matrix)
data2D = svd.transform(matrix)

fig, ax = plt.subplots(figsize = (14,8))
np.random.seed(0)
ax.plot(data2D[:, 0], data2D[:, 1],
        'or', ms=10, alpha=0.2)
ax.set_title('Truncated SVD with Cluster Assignments', size=14)
ax.grid(color='lightgray', alpha=0.7)
for i, txt in enumerate(events):
    print(i)
    plt.annotate(txt + ", " + str(y_pred[i]), (data2D[:, 0][i], data2D[:, 1][i]))
mpld3.show(fig)
#mpld3.save_html(fig, "Truncated_SVD_D3.html")




#####Interactive ScatterPlot with Dense Matrix and PCA######
"""Probably not the method to use. SVD seems better since it takes the sparse matrix "tfidf_matrix" directly."""
x = tfidf_matrix.todense()
coords = PCA(n_components=2).fit_transform(x)
fig, ax = plt.subplots(figsize = (14,8))

np.random.seed(0)
ax.plot(coords[:, 0], coords[:, 1],
        'or', ms=10, alpha=0.2)
ax.set_title('PCA with Cluster Assignments', size=14)
ax.grid(color='lightgray', alpha=0.7)
for i, txt in enumerate(events):
    plt.annotate(txt, (coords[:, 0][i], coords[:, 1][i]))
mpld3.show(fig)



"""Dendogram Making"""
fig = pylab.figure(figsize=(100,70))
children = hac.children_
distance = np.arange(children.shape[0])
no_of_observations = np.arange(2, children.shape[0]+2)
linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
dendrogram(linkage_matrix, labels = (events), truncate_mode = "level", leaf_font_size = 8)
fig.show




"""ScatterPlot"""
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



#############################################################
################Hyperparameter Testing#######################
#############################################################


hyper_params = pd.read_csv("/Users/parkerglenn/Desktop/DataScience/Article_Clustering/csv/Hyperparam_testing/Hyper_Params.csv")
hyper_params = hyper_params.reset_index()

f_scores = [x for x in hyper_params["f_score"]]
s_scores = [x for x in hyper_params["s_score"]]
person_rate = [x for x in hyper_params["person_rate"]]
ents_rate = [x for x in hyper_params["ents_rate"]]



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

s_data = pd.DataFrame({"Person Weighting": [round(x,2) for x in person_rate], "Non-Person Weighting":[round(x,2) for x in ents_rate], "Z": hyper_params.s_score})
s_data_pivoted = s_data.pivot("Person Weighting","Non-Person Weighting","Z")
ax2 = sns.heatmap(s_data_pivoted, cmap = plt.cm.hot)
cbar = ax2.collections[0].colorbar
cbar.set_label('Silhouette Score', labelpad=15)
ax2.invert_yaxis()
plt.show()



"""Cumulative Total
Highest: ents_rate = 6.368, person_rate = 2.263, f_score = 0.922, s_score = 0.093

"""
sf_data = pd.DataFrame({"Person Weighting": [round(x,2) for x in person_rate], "Entity Weighting":[round(x,2) for x in ents_rate], "Z": hyper_params.s_score + hyper_params.f_score})
sf_data_pivoted = sf_data.pivot("Person Weighting","Entity Weighting","Z")
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
