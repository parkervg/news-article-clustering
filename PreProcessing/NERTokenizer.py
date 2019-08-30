#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:59:22 2019

@author: parkerglenn
"""


"""
Known errors:
    Trump is sometimes tagged as an ORG
    U.S. is sometimes tagged as a PERSON
"""
import sklearn.base
class NERTokenizer(sklearn.base.TransformerMixin):
    """If 'tag' is True, Person entities .startswith("*") and other entities deemed "good" .startswith("&")"""
    def __init__(self, tag = False):
        self._tag = tag

    def fit(self, X, *_):
        return self

    def transform(self, X, *_):
        from nltk.corpus import stopwords
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer("english")

        import spacy
        from spacy.gold import iob_to_biluo
        nlp = spacy.load('en_core_web_md', disable=['parser','tagger','textcat'])
        from spacy.attrs import ORTH
        nlp.tokenizer.add_special_case("I'm", [{ORTH: "I'm"}])
        nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)

        english_stopwords = stopwords.words('english')
        english_stopwords.append("i'm")

        tokenized_corpus = []
        good_ents = ["PERSON","GPE","ORG", "LOC", "EVENT", "FAC"]
        continue_tags = ["B-","I-"]
        end_tags = ["L-","U-"]



        for text in X:
            toks = []
            iobs = [i.ent_iob_ for i in nlp(text)]
            biluos = list(iob_to_biluo(iobs))
            #Named entities variable
            ne = ""
            for index, tok in enumerate(nlp(text)):
                if biluos[index] in continue_tags and str(tok.ent_type_) in good_ents:
                    #str(tok).split() != [] Checks if empty token
                    #For some reason tok.whitespace_ doesn't include double token entities
                    #like "JENNIFER LAWRENCE"
                    if not self._tag:
                        ne += " " + str(tok).lower()
                    elif self._tag and str(tok).split() != []:
                        #Entity is the beginning of an entity set
                        if biluos[index] == "B-":
                            if str(tok.ent_type_) != "PERSON":
                                ne += " &" + str(tok).lower()
                            elif str(tok.ent_type_) == "PERSON":
                                ne += " *" + str(tok).lower()
                        else:
                            if str(tok.ent_type_) != "PERSON":
                                ne += " " + str(tok).lower()
                            elif str(tok.ent_type_) == "PERSON":
                                ne += " " + str(tok).lower()
                elif biluos[index] in end_tags and str(tok.ent_type_) in good_ents:
                    if not self._tag:
                        ne += " " + str(tok).lower()
                        toks.append(ne.lstrip())
                        ne = " "
                    elif self._tag and str(tok).split() != []:
                        #Entity is just a single unit
                        if biluos[index] == "U-":
                            if str(tok.ent_type_) != "PERSON":
                                ne += " &" + str(tok).lower()
                                toks.append(ne.lstrip())
                                ne = " "
                            elif str(tok.ent_type_) == "PERSON":
                                ne += " *" + str(tok).lower()
                                ne.replace("*’m", "")
                                toks.append(ne.lstrip())
                                ne = " "
                        else:
                            ne += " " + str(tok).lower()
                            # so that possesive tags are not stored with the '’s'
                            ne = ne.replace("’s", "")
                            toks.append(ne.lstrip())
                            ne = " "
                #If token is just a boring old word
                else:
                    if not tok.is_punct and not tok.is_space and str(tok).lower() not in english_stopwords:
                        toks.append(stemmer.stem(str(tok)))
            tokenized_corpus.append(toks)
        return tokenized_corpus
