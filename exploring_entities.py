#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:27:43 2019

@author: parkerglenn
"""



"""
WITH NO ENTITY MANIPULATION:
97.9% of non-PERSON ents occur in at least one other cluster. 
51.5% of PERSON ents occur in at least one other cluster.



WITH ENTITY MANIPULATION:
96.2% of non-PERSON ents occur in at least one other cluster. 
52.1% of PERSON ents occur in at least one other cluster.


Average of 51.832 PERSONs per cluster
Average of 64.94 non-PERSONs per cluster

1795 PERSON tags only appear once.
6.9% of all PERSON tags are not helpful in determining cluster boundaries.
1683 NON_PERSON tags only appear once.
5.0% of all NON_PERSON tags are not helpful in determining cluster boundaries.

Non-People entities occured 6,554 more times than people entities.



"GPE": Countries, cities, states
"ORG": Organizations
"LOC": Non-GPE locations
"EVENT": Named hurricanes, sports events, etc.
"FAC": Buildings, airports, highways, bridges
"PERSON": All people, including fictional
"""

from collections import Counter
ent_corpus = []
unique_ents = []
for article in toks:
    article_ents = ([tok for tok in article if tok.startswith("*") == True or tok.startswith("&") == True and tok != "*’m"])
    for tok in article:
        if tok not in unique_ents:
            unique_ents.append(tok)
    ent_corpus.append(article_ents)

# Creates cluster_ents, a dict where the occurences of ents per cluster are counted.
cluster_ents = {}
for place, cluster in enumerate(y_pred):
    cluster_ents[cluster] = cluster_ents.get(cluster,ent_corpus[place]) + ent_corpus[place]
    
    
cluster_ents_count = {}
for cluster in cluster_ents:
    cluster_ents_count[cluster] = dict(Counter(cluster_ents[cluster]))

# Creates dict for the amount of times an entity is used across distinct clusters
dup_clusters1 = {}
for base_ent in unique_ents:
    for cluster, ents in cluster_ents.items():
        if base_ent in ents:
            dup_clusters1[base_ent] = dup_clusters1.get(base_ent, -1) + 1
dup_clusters = {k:v for k,v in dup_clusters1.items() if v != 0}


"""
How many one-off person ents/non person ents are there?

If more one-off person ents, this explains s score.
"""

ent_occurences = {}
for cluster in cluster_ents_count:
    for ent, value in cluster_ents_count[cluster].items():
        try:
            ent_occurences[ent] += value
        except:
             ent_occurences[ent] = value


one_off_persons = 0
one_off_notpersons = 0
delp = []
delnp = []
for k, v in ent_occurences.items():
    if k.startswith("*") and v == 1:
        one_off_persons += 1
        delp.append(k)
    elif k.startswith("&") and v == 1:
        one_off_notpersons += 1
        delnp.append(k)
one_off_persons   
one_off_notpersons 

       
event_ents = {}
for article, toks in enumerate(ent_corpus):
    event_ents[events[article]] = event_ents.get(events[article], toks) + toks   
for event,toks in event_ents.items():
    event_ents[event] = dict(Counter(toks))



distribution = pd.DataFrame(columns = ["event","ent","type","ratio"])
place = -1
for event, counts in event_ents.items():
    for ent, value in counts.items():
        place += 1
        if ent.startswith("*") and ent not in delp:
            distribution.loc[place] = [event, ent, "PERSON", (value / ent_occurences[ent])]
        if ent.startswith("&") and ent not in delnp:
            distribution.loc[place] = [event, ent, "NON_PERSON", (value / ent_occurences[ent])]


people_dist = distribution.loc[distribution["type"] == "PERSON"]
not_people_dist = distribution.loc[distribution["type"] == "NON_PERSON"]



pval = 0
for value in people_dist["ratio"]:
    if value > .7:
        pval +=1

npval = 0
for value in not_people_dist["ratio"]:
    if value > .7:
        npval +=1
           



        
import matplotlib.pyplot as plt
import seaborn as sns
fig, (ax1,ax2) = plt.subplots(ncols = 2)
fig.subplots_adjust(wspace = 0.01)
sns.heatmap(distribution, cmap = "rocket", ax = ax, cbar = False)

sns.heatmap(distribution.loc[distribution["type"] == "PERSON"])






"""
Entity Usage Across Distinct Clusters

"""
ents = pd.DataFrame(columns = ["ent","occurence"])
people = pd.DataFrame(columns = ["ent","occurence"])
p = []
n = []
p1 = []
n1 = []
for k in dup_clusters:
    if k.startswith("&"):
        n.append(k)
        n1.append(dup_clusters[k])
    elif k.startswith("*"):
        p.append(k)
        p1.append(dup_clusters[k])


ents["ent"] = n
ents["occurence"] = n1
ents["type"] = "ent"
people["ent"] = p
people["occurence"] = p1
people["type"] = "person"
ents = ents.sort_values(by = "occurence", ascending = False)
people = people.sort_values(by = "occurence", ascending = False)


people_2plt = people[1:6]
ents_2plt = ents[:5]
fig, ax = plt.subplots(1,2, sharey = True, figsize = (15,8))
fig.suptitle("Entity Usage Across Distinct Clusters", fontsize=14)
sns.set(style = "darkgrid")
sns.barplot(x = "ent", y = "occurence", hue = "type", data = people_2plt, ax = ax[0], palette=["C0"])
sns.barplot(x = "ent", y = "occurence", hue = "type", data = ents_2plt,ax = ax[1], palette=["C1"])
for a in ax:
    a.set_xlabel('Entity')
    a.set_ylabel('Occurences')
fig.show()


ents = pd.DataFrame(columns = ["ent","occurence"])
people = pd.DataFrame(columns = ["ent","occurence"])
p = []
n = []
p1 = []
n1 = []
for k, v in ent_occurences.items():
    if k.startswith("&"):
        n.append(k)
        n1.append(v)
    elif k.startswith("*"):
        p.append(k)
        p1.append(v)
ents["ent"] = n
ents["occurence"] = n1
ents["type"] = "Non-Person"
people["ent"] = p
people["occurence"] = p1
people["type"] = "Person"
ents = ents.sort_values(by = "occurence", ascending = False)
people = people.sort_values(by = "occurence", ascending = False)


from matplotlib.colors import ListedColormap

color1 = ["#13b23b"]
color2 = ["#ffa100"]  
people_2plt = people[:5]
ents_2plt = ents[:5]
fig, ax = plt.subplots(1,2, sharey = True, figsize = (15,8))
#fig.suptitle("Entity Usage Across Articles", fontsize=25)
sns.set(style = "darkgrid", font_scale = 1)
one = sns.barplot(x = "ent", y = "occurence", hue = "type", data = people_2plt, ax = ax[0], palette=color1)
two = sns.barplot(x = "ent", y = "occurence", hue = "type", data = ents_2plt,ax = ax[1], palette=color2)
for item in one.get_xticklabels():
    item.set_rotation(60)
for item in two.get_xticklabels():
    item.set_rotation(60)
for a in ax:
    a.set_xlabel('Entity')
    a.set_ylabel('Occurences')
fig.show()




person = 0
not_person = 0
for ent, value in dup_clusters.items():
    if ent.startswith("&"):
        not_person += value
    elif ent.startswith("*"):
        person += value
print("Overall, non-person entities occured across distinct clusters {} more times than person entities did.".format(not_person - person))

person
not_person




all_people = 0
all_not_people = 0
for ent, value in ent_occurences.items():
    if ent.startswith("&"):
        all_not_people += value
    if ent.startswith("*"):
        all_people += value
print("Non-People entities occured {} more times than people entities.".format(all_not_people - all_people))


all_people
all_not_people

x = 0
y = 0
for ent in unique_ents:
    if ent.startswith("&"):
        x += 1
    if ent.startswith("*"):
        y += 1

person / y
not_person / x

all_people / 500
all_not_people / 500

1710 / all_people
1772 / all_not_people

all_people
all_not_people - all_people


