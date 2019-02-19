# newsfeed-nlp

## Abstract 
This unsupervised learning project allows the average news consumer to experience a stream-lined information acquisition process, free of any useless repetition. Working in Python and using a Kaggle dataset (https://www.kaggle.com/snapcrack/all-the-news) of 85,000 news articles, we extract significance from the texts by utilizing a modified TFIDF-vectorizer to pre-process the data. We experiment with various clustering techniques (Kmeans, HAC, and KNN), paired with various success metrics to gauge effectiveness of the news consolidation. Visualizations are created using Seaborn and Matplotlib, along with D3 for ease of exploration. Spacy is used as the primary NLP.

## Motivation
In a political climate of intensely polarizing takes on the latest scandals, international relations, and national issues, it can be difficult to make sense of all the data. With the rise of social media, the ability to publish has been democratized and repetition in the newsfeed runs rampant. By grouping together news stories by topic, not only is the newsfeed browsing process streamlined, but groups created that provide differing perspective on the same story.

## Methodology

![Distance Heatmap](/Visualizations/distance_heatmap.png)

|        |                                                                                   |             |                   |          |      |       |                                                                                                                         | 
|--------|-----------------------------------------------------------------------------------|-------------|-------------------|----------|------|-------|-------------------------------------------------------------------------------------------------------------------------| 
| id     | title                                                                             | publication | author           | date     | year | month | url       | content                                                                                                              | 
| 151908 | Alton Sterling‚Äôs son: ‚ÄôEveryone needs to protest the right way, with peace‚Äô | Guardian    | Jessica Glenza    | 7/13/16  | 2016 | 7     | https://www.theguardian.com/us-news/2016/jul/13/alton-sterling-son-cameron-protesters-baton-rouge                       | 
| 151909 | Shakespeare‚Äôs first four folios sell at auction for almost ¬£2.5m               | Guardian    |                   | 5/25/16  | 2016 | 5     | https://www.theguardian.com/culture/2016/may/25/shakespeares-first-four-folios-sell-at-auction-for-almost-25m           | 
| 151910 | My grandmother‚Äôs death saved me from a life of debt                             | Guardian    | Robert Pendry     | 10/31/16 | 2016 | 10    | https://www.theguardian.com/commentisfree/2016/oct/31/grandmothers-death-saved-me-life-of-debt                          | 
| 151911 | I feared my life lacked meaning. Cancer pushed me to find some                    | Guardian    | Bradford Frost    | 11/26/16 | 2016 | 11    | https://www.theguardian.com/commentisfree/2016/nov/26/cancer-diagnosis-existential-life-accomplishments-meaning         | 
| 151912 | Texas man serving life sentence innocent of double murder, judge says             | Guardian    |                   | 8/20/16  | 2016 | 8     | https://www.theguardian.com/us-news/2016/aug/20/texas-life-sentence-innocence-dna-richard-bryan-kussmaul                | 
| 



Possible Roadblocks:
* Specificity in clusters. Russia = Russian election hacking or Russian international relations or Russian Olympic ban?
* Intersections in broad topics. E.g. "Donald Trump speaks about Hurricane Matthew" about DT or hurricane?


Notes:
* Put special emphasis on dates and names when organizing clusters (use spacy entity recognition) 



