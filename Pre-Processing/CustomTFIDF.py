#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:12:21 2019

@author: parkerglenn
"""
import sklearn

class CustomTFIDF(sklearn.base.TransformerMixin):
    
    def __init__(self, person_rate = 1,
                  ents_rate = 1, max_df = 1.0, min_df = 0, date_weight = .1, julian = False, df = False):
        self._person_rate = person_rate
        self._ents_rate = ents_rate
        self._min_df = min_df
        self._max_df = max_df
        self._date_weight = date_weight
        self._julian = julian
        self._df = df
        
        
    def fit(self, X, *_):
        return self
    
    
    def TF_dict(self, article):
            article_tf = {}
            for word in article:
                if word in article_tf:
                    article_tf[word] += 1
                else:
                    article_tf[word] = 1
            for word in article_tf:
                """Manipulate word.startswith() to account for entity weighting."""
                #word.startswith("*") applies to PERSON tags
                if word.startswith("*"):
                    occurences = article_tf[word]
                    article_tf[word] = (occurences / len(article)) * self._person_rate
                #word.startswith("&") applies to NON-PERSON tags
                elif word.startswith("&"):
                    occurences = article_tf[word]
                    article_tf[word] = (occurences / len(article)) * self._ents_rate
                else:
                    occurences = article_tf[word]
                    article_tf[word] = (occurences / len(article))            
            return article_tf


    def Count_dict(self):
        countDict = {}
        for article in self._TF:
            found_words = []
            for word in article:
                if word in countDict and word not in found_words:
                    countDict[word] += 1
                    found_words.append(word)
                elif word not in found_words:
                    countDict[word] = 1
                    found_words.append(word)
        return countDict

        


    def IDF_dict(self, X):
        import math
        idfDict = {}
        for word in self._countDict:
        #len(corpus) is 1000, the total number of documents for this project
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
            if self._countDict[word] > self._min_df and (self._countDict[word] / self._amount) < self._max_df:
                idfDict[word] = math.log((1 + self._amount) / (1 + self._countDict[word])) + 1
            else:
                idfDict[word] = 0
        return idfDict


    def TFIDF_list(self, article):
        article_tfidf = {}
        for word in article:
            #article[word] is the TF score for that word in the given article
            article_tfidf[word] = article[word] * self._idfDict[word]
        return article_tfidf


    
    def compute_TFIDF_matrix(self, article):
        terms = sorted(self._countDict.keys())
        article_matrix = [0.0] * len(terms)
        for i, word in enumerate(terms):
            #Stores tfidf value of unique word in terms
            #if the word is in the article
            if word in article:
                #article[word] is the word's tfidf score
                article_matrix[i] = article[word]
        return article_matrix


    def makeJulian(self,X):
        """X must be a df with a "date" column in '%Y-%m-%d' format.
        
        This takes a while to run.
        """
        import datetime
        fmt = '%Y-%m-%d'
        s = "2016-02-01"
        s = datetime.datetime.strptime(s,fmt)
        
        
        import julian
        
        julian_lst = []
        for date in X["date"]:
            dt = datetime.datetime.strptime(date,fmt)
            julian_lst.append(julian.to_jd(dt + datetime.timedelta(hours=12), fmt = "jd"))

        #Find amount of unique dates
        #Set arbitrary value (maybe 1, do hyperparameting testing again) to index of that date
        unique = list(set(julian_lst))
                        
        #Just to have easy access to indexes, ultimately to decide which place in the feature 
        #matrix corresponds to which date
        unique_dict = {}
        for place, date in enumerate(unique):
            unique_dict[date] = place
        
        jul_matrix = []
        for place, date in enumerate(julian_lst):
            #mini_matrix is the matrix for the individual article
            mini_matrix = [0.0] * len(julian_lst)
            for num in range(-4,4):
                if num == 0:
                    mini_matrix[unique_dict[date]] = self._date_weight
                else:
                    if (unique_dict[date] + num) > -1:
                        #Deterioation function as dates get further away from target
                        #Since dates within a proximity of about 4 seem to indicate some significance in similarity
                        #Can change, right now it's date_weight divided by absolute value of num
                        mini_matrix[unique_dict[date] + num] = (self._date_weight / (abs(num)+.5))
            jul_matrix.append(mini_matrix)
        return jul_matrix
        

    

    def transform(self, X, *_):
        self._amount = len(X)
        from sklearn import preprocessing
        self._TF = [self.TF_dict(article) for article in X]
        self._countDict = self.Count_dict()
        self._idfDict = self.IDF_dict(X)
        self._tfidf = [self.TFIDF_list(article) for article in self._TF]
        self._tfidf_matrix = [self.compute_TFIDF_matrix(article) for article in self._tfidf]
        self._tfidf_matrix = preprocessing.normalize(self._tfidf_matrix, norm = 'l2')
        #Decides whether or not to add date component
        if self._julian == True:
            import scipy
            from scipy.sparse import  hstack 
            self._jul_list = self.makeJulian(self._df)
            self._jul_matrix = scipy.sparse.csr_matrix(self._jul_list)
            self._combo_matrix = hstack([self._jul_matrix, self._tfidf_matrix]).toarray()
            return self._combo_matrix
        else:
            return self._tfidf_matrix