# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:19:07 2019

@author: 
"""
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.datasets import fetch_20newsgroups

import nltk 
import re
from scipy import sparse
import string

stop_words = set(nltk.corpus.stopwords.words("english"))

def main():
    train, test = get_20_news_group(30)
    tf_idf(train, test)
    
def get_20_news_group(size_train):
    train = fetch_20newsgroups(subset = 'train', remove=('footers', 'quotes'), shuffle=True) #The F-score will be lower because it is more realistic.
    test  = fetch_20newsgroups(subset = 'test', remove=('footers', 'quotes') ) #The F-score will be lower because it is more realistic.
    
    return train.data[:30], [test.data[0]]

def tf_idf(train, test):
    candidates_train = list()
    candidates_tokanize_train = list()
    candidates_tokanize_test = list()
    #TRAIN
    for doc in train:
        phrases = nltk.sent_tokenize(doc)
        candidates_tokanize_train = sentence_preprocess(phrases)
        candidates_train = candidates_train + [candidates_tokanize_train]
            
    vectorizer_tfidf = tf_idf_train(candidates_train)   
    #TEST
    phrases = nltk.sent_tokenize(test[0])
    candidates_tokanize_test = sentence_preprocess(phrases)
    test_vector = tf_idf_test(vectorizer_tfidf, candidates_tokanize_test)
    
    feature_names = vectorizer_tfidf.get_feature_names()
    test_vector = tf_idf_scores(test_vector.tocoo(), feature_names,chars_or_words="words")
    
    #SORT
    sorted_terms = sort_terms(test_vector.tocoo())
    keyphrases = extract_keyphrases(feature_names ,sorted_terms)
    print(keyphrases)
    

def sentence_preprocess(phrases):
    candidates =[]
    for phrase in phrases:  
        phrase = phrase.lower()
        #Remove puntuation, except hifen
        remove = string.punctuation
        remove = remove.replace("-", "") # don't remove hyphens
        pattern = r"[{}]".format(remove) # create the pattern
        phrase = re.sub(pattern, "", phrase)
        
        phrase = re.sub(r'[^\D]'  ,'',phrase)
        phrase = re.sub(r'[\n]',' ', phrase)
        candidates.append(phrase)
        
    return candidates

def tf_idf_train(candidates_train):
     #Learn the vocabulary dictionary and return term-document matrix
    vectorizer_tfidf = TfidfVectorizer(use_idf = True, analyzer = 'word', ngram_range=(1, 3), stop_words = 'english')
    #vectorizer_tfidf.fit_transform(candidates_train)
    
    #print("candidates_train>>>", candidates_train)
    for doc in candidates_train:
       vectorizer_tfidf.fit_transform(doc)
    #vectorizer_tfidf.fit_transform(candidates_train)
    #DÁ O DICONÁRIO!!!
    
    #vectorizer_tfidf_fitted = map(vectorizer_tfidf.fit_transform, candidates_train)
    
    #print(vectorizer_tfidf.vocabulary_)
        
    return vectorizer_tfidf

def tf_idf_test(vectorizer_tfidf, candidates_tokanize_test):
    test_vector = vectorizer_tfidf.transform(candidates_tokanize_test)
    
    #print(vectorizer_tfidf.vocabulary_.keys())#DA TERMOS!!!!!!
#    #test_vector  = vectorizer_tfidf.transform(candidates_tokanize_test)
#    print(testvec)
    return test_vector

def tf_idf_scores(test_vector, feature_names,chars_or_words="words"):
    #print("type", type(testvec))
    #print("before >>>>testvec", testvec.toarray())

    test_vector = test_vector.toarray()
    
    for i in range(0, test_vector.shape[0]):
        for j in range(0, test_vector.shape[1]):
            if test_vector[i,j] != 0:
                if chars_or_words == 'chars':
                    test_vector[i,j] = test_vector[i,j] * len(feature_names[j])     
                    
                elif chars_or_words == 'words':
                    test_vector[i,j] =  test_vector[i,j] * len(feature_names[j].split())
    
    #print("after >>>>testvec", testvec)
    test_vector = sparse.csr_matrix(test_vector)
    return test_vector

def sort_terms(test_vector):
    tuples = zip(test_vector.col, test_vector.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_keyphrases(feature_names ,sorted_terms):
    sorted_terms = sorted_terms[:5]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_terms:
        
        #keep track of feature name and its corresponding score
        score_vals.append(score)
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results
    