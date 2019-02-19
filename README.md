# newsfeed-nlp

## Abstract 
This unsupervised learning project allows the average news consumer to experience a stream-lined information acquisition process, free of any useless repetition. Working in Python and using a Kaggle dataset (https://www.kaggle.com/snapcrack/all-the-news) of 85,000 news articles, we extract significance from the texts by utilizing a modified TFIDF-vectorizer to pre-process the data. We experiment with various clustering techniques (Kmeans, HAC, and KNN), paired with various success metrics to gauge effectiveness of the news consolidation. Visualizations are created using Seaborn and Matplotlib, along with D3 for ease of exploration. Spacy is used as the primary NLP.

## Motivation
In a political climate of intensely polarizing takes on the latest scandals, international relations, and national issues, it can be difficult to make sense of all the data. With the rise of social media, the ability to publish has been democratized and repetition in the newsfeed runs rampant. By grouping together news stories by topic, not only is the newsfeed browsing process streamlined, but groups created that provide differing perspective on the same story.

OriginDataset: https://www.kaggle.com/snapcrack/all-the-news

Possible Roadblocks:
* Specificity in clusters. Russia = Russian election hacking or Russian international relations or Russian Olympic ban?
* Intersections in broad topics. E.g. "Donald Trump speaks about Hurricane Matthew" about DT or hurricane?


Notes:
* Put special emphasis on dates and names when organizing clusters (use spacy entity recognition) 



