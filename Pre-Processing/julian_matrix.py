#!/usr/bin/env python
# coding: utf-8

# In[60]:
import os
import pandas as pd
os.chdir("/Users/parkerglenn/Desktop/DataScience/Article_Clustering")
df = pd.read_csv("/Users/parkerglenn/Desktop/DataScience/Article_Clustering/csv/all_GOOD_articles.csv")
labels_df= pd.read_csv("/Users/parkerglenn/Desktop/DataScience/Article_Clustering/Google_Drive/Article_Classification26.csv")
#Deletes unnecessary columns
df = df.drop(df.columns[:12], axis = 1)
#Sets manageable range for working data set
new_df = df[5000:6000]


# In[63]:


import datetime
fmt = '%Y-%m-%d'
s = "2016-02-01"
s = datetime.datetime.strptime(s,fmt)


# In[64]:


new_df["dt"]=False
for i in enumerate(new_df["date"]):
    new_df["dt"].iloc[i[0]] = datetime.datetime.strptime(i[1],fmt)


# In[65]:


import julian
new_df["julian"] = False
for i in enumerate(new_df["dt"]):
    jd = julian.to_jd(i[1] + datetime.timedelta(hours=12), fmt = "jd")
    new_df["julian"].iloc[i[0]] = jd


# In[75]:


#Find amount of unique dates
#Set arbitrary value (maybe 1, do hyperparameting testing again) to index of that date
unique = []
for i in new_df["julian"]:
    unique.append(i)
unique = set(unique)
print(len(unique))


# In[80]:


unique_dict = {}
for place, date in enumerate(unique):
    unique_dict[date] = place



# In[96]:


import scipy
jul_matrix = []
for place, date in enumerate(new_df["julian"]):
    mini_matrix = [0.0] * len(new_df)
    #Change the .1 value to something appropriate
    #Use hyperparemeter testing
    mini_matrix[unique_dict[date]] = 0.1
    jul_matrix.append(mini_matrix)


jul_matrix




