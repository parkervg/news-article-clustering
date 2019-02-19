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
| 151914 | My dad‚Äôs Reagan protests inspire me to stand up to Donald Trump                 | Guardian    | Steven W Thrasher | 11/28/16 | 2016 | 11    | https://www.theguardian.com/commentisfree/2016/nov/28/my-dads-reagan-protests-inspire-me-to-stand-up-to-donald-trump    | 
| 151915 | Flatmates of gay Syrian refugee beheaded in Turkey fear they will be next         | Guardian    | Patrick Kingsley  | 8/7/16   | 2016 | 8     | https://www.theguardian.com/world/2016/aug/07/flatmates-of-gay-syrian-refugee-beheaded-in-turkey-fear-they-will-be-next | 
| 151916 | Jaffas and daredevils: life on the world‚Äôs steepest street                      | Guardian    | Eleanor Ainge Roy | 7/22/16  | 2016 | 7     | https://www.theguardian.com/world/2016/jul/23/jaffas-and-daredevils-life-on-the-worlds-steepest-street                  | 
| 151917 | NSA contractor arrested for alleged theft of top secret classified information    | Guardian    | Ewen MacAskill    | 10/5/16  | 2016 | 10    | https://www.theguardian.com/us-news/2016/oct/05/nsa-contractor-arrested-harold-thomas-martin-edward-snowden             | 
| 151918 | Donald Trump to dissolve his charitable foundation after mounting complaints      | Guardian    | Ben Jacobs        | 12/24/16 | 2016 | 12    | https://www.theguardian.com/us-news/2016/dec/24/trump-university-shut-down-conflict-of-interest                         | 



Possible Roadblocks:
* Specificity in clusters. Russia = Russian election hacking or Russian international relations or Russian Olympic ban?
* Intersections in broad topics. E.g. "Donald Trump speaks about Hurricane Matthew" about DT or hurricane?


Notes:
* Put special emphasis on dates and names when organizing clusters (use spacy entity recognition) 



