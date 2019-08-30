import os
import pandas as pd
import glob
import csv
"""
Takes as input all original Kaggle CSVs, and filters the concatenated_df to only where the date and url are present.
"""

path = "/Users/parkerglenn/Desktop/DataScienceSets"


list_ = []
all_files = glob.glob(path + "/*.csv")
frame = pd.DataFrame()


df_from_each_file = (pd.read_csv(f) for f in all_files)
concatenated_df = pd.concat(df_from_each_file, ignore_index = True)

dates = concatenated_df['date']

url = concatenated_df['url']

df1 = concatenated_df[~concatenated_df['date'].isna()]
df = df1[~df1['url'].isna()]
df = df.reset_index()
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)

df = df.sort_values(by=['date'])

df.to_csv("all_good_articles.csv")
