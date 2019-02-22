#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:31:02 2019

@author: parkerglenn
"""

def tokenize_and_stem_NER(corpus):
    global tokenized_corpus
    tokenized_corpus = []
    good_ents = ["PERSON", "GPE","ORG", "LOC", "EVENT", "FAC"]
    continue_tags = ["B-","I-"]
    end_tags = ["L-","U-"]
    for text in corpus:
        toks = []
        iobs = [i.ent_iob_ for i in nlp(text)]
        biluos = list(iob_to_biluo(iobs))
        index = -1
        #Named entities variable
        ne = ""
        for tok in nlp(text):
            index += 1
            if biluos[index] in continue_tags and str(tok.ent_type_) in good_ents:
                #Checks if empty token
                #For some reason tok.whitespace_ doesn't include double token entities
                #like "JENNIFER LAWRENCE"
                if str(tok).split() != [] and str(tok.ent_type_) != "PERSON":
                    ne += " " + str(tok).upper()
                elif str(tok).split() != [] and str(tok.ent_type_) == "PERSON":
                    ne += " " +str(tok).title()
            elif biluos[index] in end_tags and str(tok.ent_type_) in good_ents:
                if str(tok).split() != [] and str(tok.ent_type_) != "PERSON":
                    ne += " " + str(tok).upper()
                    toks.append(ne.lstrip())
                    ne = " "
                elif str(tok).split() != [] and str(tok.ent_type_) == "PERSON":
                    ne += " " + str(tok).title()
                    ne = ne.replace("’S", "")
                    toks.append(ne.lstrip())
                    ne = " "
                ne = " "
            #If token is just a boring old word
            else:
                if tok.is_punct == False and str(tok).lower() not in list(stop_words.ENGLISH_STOP_WORDS):
                    toks.append(stemmer.stem(str(tok)))
        tokenized_corpus.append(toks)



def do_tfidf(ents_rate, person_rate):
    
    def TF_dict(article):
        article_tf = {}
        for word in article:
            if word in article_tf:
                article_tf[word] += 1
            else:
                article_tf[word] = 1
        for word in article_tf:
            """Manipulate word.isupper() to account for entity weighting."""
            if word.isupper():
                occurences = article_tf[word]
                article_tf[word] = (occurences / len(article)) * ents_rate
            #word.istitle() applies to PERSON tags
            elif word.istitle():
                occurences = article_tf[word]
                article_tf[word] = (occurences / len(article)) * person_rate
            else:
                occurences = article_tf[word]
                article_tf[word] = (occurences / len(article))            
        return article_tf
    
    TF = [TF_dict(article) for article in tokenized_corpus]     
    
    
    
    def Count_dict():
        countDict = {}
        for article in TF:
            found_words = []
            for word in article:
                if word in countDict and word not in found_words:
                    countDict[word] += 1
                    found_words.append(word)
                elif word not in found_words:
                    countDict[word] = 1
                    found_words.append(word)
        return countDict
        
    countDict = Count_dict()
    
    import operator
    sortCount = sorted(countDict.items(), key=operator.itemgetter(1), reverse = True)
    
    
    def IDF_dict():
        import math
        idfDict = {}
        for word in countDict:
            #len(corpus) is 1000, the total number of documents
            #countDict[word] is the number of articles the word appears in
            """
            From Sci-Kit code: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/text.py
                'smooth_idf: If ``smooth_idf=True`` (the default), the constant "1" is added to the
                numerator and denominator of the idf as if an extra document was seen
                containing every term in the collection exactly once, which prevents
                zero divisions: idf(d, t) = log [ (1 + n) / (1 + df(d, t)) ] + 1.'
                
                'The effect of adding "1" to
                the idf in the equation above is that terms with zero idf, i.e., terms
                that occur in all documents in a training set, will not be entirely
                ignored.'
                
                min_df: 'When building the vocabulary ignore terms that have a document
                frequency strictly lower than the given threshold. This value is also
                called cut-off in the literature.'
                
                max_df: 'When building the vocabulary ignore terms that have a document
                frequency strictly higher than the given threshold (corpus-specific
                stop words).'
                
                norm: (default='l2') Each output row will have unit norm, either:
                    * 'l2': Sum of squares of vector elements is 1. The cosine
                    similarity between two vectors is their dot product when l2 norm has
                    been applied.
            """
            #Implements min_df and max_df
            min_df = 0
            max_df = 1.0
            if countDict[word] > min_df and (countDict[word] / len(corpus)) < max_df:
                idfDict[word] = math.log((1 + len(corpus)) / (1 + countDict[word])) + 1
            else:
                idfDict[word] = 0
        return idfDict
    
    idfDict = IDF_dict()
    
    
    
    def TFIDF_list(article):
        article_tfidf = {}
        for word in article:
            #article[word] is the TF score for that word in the given article
            article_tfidf[word] = article[word] * idfDict[word]
        return article_tfidf
    
    
    
    tfidf = [TFIDF_list(article) for article in TF]
    
    
    
    
    from sklearn import preprocessing
    terms = sorted(countDict.keys())
    def compute_TFIDF_matrix(article):
        article_matrix = [0.0] * len(terms)
        for i, word in enumerate(terms):
            #Stores tfidf value of unique word in terms
            #if the word is in the article
            if word in article:
                #article[word] is the word's tfidf score
                article_matrix[i] = article[word]
        return article_matrix
    
    
    
    tfidf_matrix = [compute_TFIDF_matrix(article) for article in tfidf]
    
    #Normalized with the default l2 setting
    tfidf_matrix = preprocessing.normalize(tfidf_matrix, norm = 'l2')

    return tfidf_matrix

################################################################
####################HAC#########################################
################################################################

def HAC(matrix):
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import dendrogram
    hac = AgglomerativeClustering(n_clusters=500, affinity = "euclidean")
    hac.fit_predict(tfidf_matrix)
    y_pred = list(hac.labels_)
    success(hac, y_pred, tfidf_matrix)

def success(model, clusters, matrix):
    
    import os
    import pandas as pd
    import codecs
    
    os.chdir("/Users/parkerglenn/Desktop/DataScience/Article_Clustering")

    data = codecs.open('/Users/parkerglenn/Desktop/DataScience/Article_Clustering/csv/all_GOOD_articles.csv', encoding = 'utf-8')
    data_with_labels = codecs.open("/Users/parkerglenn/Desktop/DataScience/Article_Clustering/Google_Drive/Article_Classification26.csv")
    
    df = pd.read_csv(data)
    labels_df= pd.read_csv(data_with_labels)
    
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


    articles = {"title": titles, "date": new_df["date"], "cluster": clusters, "content": new_df["content"], "event": events[:1000]}
    frame = pd.DataFrame(articles, index = [clusters] , columns = ['title', 'date', 'cluster', 'content', "event"])
    frame['cluster'].value_counts()

    """
    BELOW THIS CREATES DICT OF CLUSTERS AND PREDOMINANT EVENT
    
    If multiple events occur the same amount of times in a single cluster,
    the ratio function is invoked to choose the event holding the most relative
    significance. If one ratio is not greater than the others (ex. a cluster 
    composed of 5 one-off events) then the cluster is disregared (labelled "nan").
    
    If the cluster only contains one event, it is assumed at this stage that it is 
    the main cluster for the event. 
    
    BUGS:
        If the cluster contains only "nan" events, it will not show up in y_trueDict
        (ex. Cluster 113 is not shown, consisting of {'nan': 2} )
    """ 
    from collections import Counter
    all_events = []
    #This fixes quirk where the same cluster was iterated over multiple times
    clusters_we_saw = []
    for cluster in clusters: 
        if cluster not in clusters_we_saw:
            clusters_we_saw.append(cluster)
            for event in frame.loc[cluster]["event"].values.tolist():
                if event != "nan" and event != "useless":
                    all_events.append(event)
    event_occurences = dict(Counter(all_events))
    
    all_clusters = []
    for cluster in clusters:
        if cluster not in all_clusters:
            all_clusters.append(cluster)
            
    y_trueDict = {}
    #This range needs to be changed depending on the cluster model
    for i in range(0,len(all_clusters)):
        ratios = []
        counts = []
        cluster_event = []
        ratio = 0
        
        #Counts occurence per cluster of event
        for event in frame.loc[i]["event"].values.tolist():
            if event != "nan" and event != "useless":
                counts.append(event)
        counts = Counter(counts) 
        
        
        if len(counts) > 1:
            score_1 = list(counts.most_common()[0])[1]
            score_2 = list(counts.most_common()[1])[1]
            #Check to see if there are multiple events with same frequency
            if score_1 == score_2:
                #Gets all events with same frequency
                tied_events = [k for k,v in dict(counts).items() if v == score_1]
                for event in tied_events:
                    #Gets the ratio of an occurence for an event in a cluster
                    #For example, if an event happens only once, it's ratio will be 1
                    #But if "iowa_caucuses" is used 100 times and only 20 times in a specific cluster,
                    #its ratio is .2
                    new_ratio = score_1 / int(event_occurences[event])
                    ratios.append(new_ratio)
                    if new_ratio > ratio:
                        cluster_event = event
                        ratio = new_ratio
                    #If result is an empty list, all ratios are unique. If not, there
                    #are repititions and the data point is thrown out.
                    if list(set([x for x in ratios if ratios.count(x) > 1])) != []:
                         y_trueDict[i] = "nan"
                         break
                    
                    #Dumb try and except sees if ytrueDict[i] is already set to something ("nan")
                    try: 
                        y_trueDict[i] 
                    except:
                        counts = dict(counts)
                        #Makes sure there's still the occurence in cluster attached to the cluster_event
                        y_trueDict[i] = [cluster_event, counts[cluster_event]]
        
    
        #If there is one obviously right event, i.e. score_1 != score_2
            else:
                y_trueDict[i] = list(counts.most_common()[0])
        
        #Catches the instance of only one item per cluster, i.e. len(counts) !> 1
        elif len(counts) == 1:
            y_trueDict[i] =list(counts.most_common()[0]) 
    
    
    #Re-analyzes y_trueDict, applying ratio again so there's one objectively "right" cluster per event
    a = []
    for k in y_trueDict:
        a.append(y_trueDict[k][0])
    a = dict(Counter(a))
    
    #Sees where the same event label is applied to multiple clusters
    duplicates = []
    for g in a:
        if a[g] > 1 and g != "n" and g != "unknown":
            duplicates.append(g)
    
    
    #Creates dup_eventsPLUSratio, where the duplicate events are stored by cluster number
    #with their ratio
    dup_eventsPLUSratio = {}
    for key in y_trueDict:
        if y_trueDict[key][0] in duplicates:
            event = y_trueDict[key][0]
            ratio = int(y_trueDict[key][1]) / int(event_occurences[event])
            eventPLUSratio = []
            eventPLUSratio.append(event)
            eventPLUSratio.append(ratio)
            dup_eventsPLUSratio[key] = eventPLUSratio
    dup_eventsPLUSratio
    
    #Dives into dup_eventsPLUSratio to see what cluster is more approrpiate for event
    for duplicate in duplicates:
        ratios = []
        for key in dup_eventsPLUSratio:
            if dup_eventsPLUSratio[key][0] == duplicate:
                ratios.append(dup_eventsPLUSratio[key][1])
        sort=sorted(ratios,reverse=True)
        highest = sort[0]
        theGood_one = [duplicate, highest]
        for key in dup_eventsPLUSratio:
            if dup_eventsPLUSratio[key][0] == duplicate:    
                if dup_eventsPLUSratio[key] != theGood_one or highest == sort[1]:
                    y_trueDict[key] = "nan"
            #If after all that there's still a tie between the top two ratios,
            #(like in hail_caesar_movie where its split 2 and 2 between clusters)
            #its given a "nan" label
            #COULD BE CHANGED TO FIT A WHILE LOOP THAT THEN FINDS score_2 AND
            #RELABELS CLUSTER TO SECOND MOST POPULAR EVENT IF THAT EVENT IS NOT
            #ALREADY ASSIGNED A CLUSTER
    
    #Gets y_true, the correct cluster assignments for each event
    bad_labels = ["useless","nan","unkown"]
    y_true = []          
    for event in events[:1000]:
        find = False
        for key in y_trueDict:
            #Used to see if there is a distinct cluster for that event
            #FIXED BUG: probably still some duplicates in y_trueDict somehow, bc output len is 10005
            #maybe the "unknown" or "useless" stuff?
            if y_trueDict[key][0] == event and y_trueDict[key][0] not in bad_labels:
                y_true.append(key)
                find = True
        if find == False:
            #Arbitrary value that's not going to return a match in t score
            y_true.append("nan")
    
    
    #Gets y_pred, the cluster where each individual event was actually clustered
    y_pred = []
    for cluster_assignment in frame["cluster"]:
        y_pred.append(cluster_assignment)
    
    #checks how events actually match up with definitively defined cluster
    num = 0
    for i in y_true:
        if i != "nan":
            num += 1
    print("Working with " + str(num) + " samples based on a corpus of " + str(len(corpus)) + " documents: ")
    print()
    
    #Re-Aligns two lists to only include good values (those not equalling "nan")
    filtered_y_true = []
    filtered_y_pred = []   
    
    for place in range(len(y_true)):
        if y_true[place] != "nan":
            filtered_y_true.append(y_true[place])
            filtered_y_pred.append(y_pred[place])
    
    
    
    
    
    """F1 score is the harmonic average of precision and recall. """
    
    from sklearn.metrics import f1_score
    print("The F1 score for the model is " + str(f1_score(y_true = filtered_y_true, y_pred = filtered_y_pred, average = "micro")))
    print()
    #500_no_ngrams F1 score micro: 0.8785046728971962 (also works off the most samples)
    #350_3_ngrams F1 score micro: 0.8718861209964412 (but goes off 281 samples rather than 303 in no ngrams)
    #700_no_ngrams F1 score micro: 0.8638392857142858
    #350_no_ngrams F1 score micro: 0.8576158940397351
    #300_3_ngrams F1 score micro: 0.8294573643410853
    
    """ Silhouette values lies in the range of [-1, 1]. A value of +1 indicates that the sample is far away 
    from its neighboring cluster and very close to the cluster its assigned. Similarly, value of -1 
    indicates that the point is close to its neighboring cluster than to the cluster its assigned. 
    And, a value of 0 means its at the boundary of the distance between the two cluster. Value of +1 
    is ideal and -1 is least preferred. Hence, higher the value better is the cluster configuration. """
    
    from sklearn.metrics import silhouette_score
    print("The sillhouette score for the model is " + str(silhouette_score(matrix, y_pred)))
    print()
    print()
    print()
    #500_no_ngrams: 0.07096239881264323
    #350_no_ngrams: 0.06777628195061947
    #700_no_ngrams: 0.06251251395097632
    #350_3_ngrams: 0.04969413068018369
    #300_3_ngrams: 0.04857286650243616
    
    global s_score 
    global f_score
    global num2
    
    s_score = silhouette_score(matrix, y_pred)
    f_score = f1_score(y_true = filtered_y_true, y_pred = filtered_y_pred, average = "micro")
    num2 = 0
    for i in y_true:
        if i != "nan":
            num2 += 1