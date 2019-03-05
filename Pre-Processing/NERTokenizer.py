#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:59:22 2019

@author: parkerglenn
"""
import os
import sklearn.base
class NERTokenizer(sklearn.base.TransformerMixin):
    def __init__(self, upper = True):
        self._upper = upper
    
    def fit(self, X, *_):
        return self
    
    def transform(self, X, *_):
        from nltk.corpus import stopwords
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer("english")
        import spacy
        from spacy import gold   
        from spacy.gold import iob_to_biluo
        nlp = spacy.load('en_core_web_md', disable=['parser','tagger','textcat'])
        nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)
        english_stopwords = stopwords.words('english')
        from sklearn.feature_extraction import stop_words


        tokenized_corpus = []
        good_ents = ["PERSON","GPE","ORG", "LOC", "EVENT", "FAC", ]
        continue_tags = ["B-","I-"]
        end_tags = ["L-","U-"]
        
        for text in X:
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
                    if self._upper == False:
                        ne += " " + str(tok).lower()
                    elif self._upper == True:
                        if str(tok).split() != [] and str(tok.ent_type_) != "PERSON":
                            ne += " " + str(tok).upper()
                        elif str(tok).split() != [] and str(tok.ent_type_) == "PERSON":
                            ne += " " + str(tok).title()
                elif biluos[index] in end_tags and str(tok.ent_type_) in good_ents:
                    if self._upper == False:
                        ne += " " + str(tok).lower()
                        toks.append(ne.lstrip())
                        ne = " "
                    elif self._upper == True:
                        if str(tok).split() != [] and str(tok.ent_type_) != "PERSON":
                            ne += " " + str(tok).upper()
                            toks.append(ne.lstrip())
                            ne = " "
                        elif str(tok).split() != [] and str(tok.ent_type_) == "PERSON":
                            ne += " " + str(tok).title()
                            toks.append(ne.lstrip())
                            ne = " "
                #If token is just a boring old word
                else:
                    if tok.is_punct == False and tok.is_space == False  and str(tok).lower() not in english_stopwords:
                        toks.append(stemmer.stem(str(tok)))
            tokenized_corpus.append(toks)
        return tokenized_corpus