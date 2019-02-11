#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:08:12 2019

@author: parkerglenn
"""
def TFIDF(corpus):
    def tokenize_and_stem(text):
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        stems = [stemmer.stem(t) for t in filtered_tokens]
        return stems
    
    def tokenize_only(text):
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        return filtered_tokens
    
    
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in corpus:
        allwords_stemmed = tokenize_and_stem(i)
        totalvocab_stemmed.extend(allwords_stemmed)
        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)
    
    #Let's you search with stemmed word to see original format of word
    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
    print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
    
    tf = TfidfVectorizer(analyzer='word', use_idf=True, stop_words = "english", tokenizer = tokenize_and_stem,min_df = 0)
    
    global tfidf_matrix
    tfidf_matrix =  tf.fit_transform(corpus)
    print(tfidf_matrix.shape)
    terms = tf.get_feature_names()
    
    
    from sklearn.metrics.pairwise import cosine_similarity
    dist = 1 - cosine_similarity(tfidf_matrix)


