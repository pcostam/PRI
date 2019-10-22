# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import nltk 
import re
from scipy import sparse
import string
#import sys
#np.set_printoptions(threshold=sys.maxsize)

stop_words = set(nltk.corpus.stopwords.words("english"))

def main():
    train, test = get_20_news_group(30)
    tf_idf(train, test)
    
def get_20_news_group(size_train):
    train = fetch_20newsgroups(subset = 'train', remove=('footers', 'quotes'),shuffle=True) #The F-score will be lower because it is more realistic.
    test  = fetch_20newsgroups(subset = 'test', remove=('footers', 'quotes') ) #The F-score will be lower because it is more realistic.
 
    return train.data[:30], [test.data[0]]

"""
returns vectorizer
"""
def tf_idf_train(candidates_train, th_min=1, th_max=1):
     #Learn the vocabulary dictionary and return term-document matrix
    vectorizer_tfidf = TfidfVectorizer(use_idf = True, analyzer = 'word', 
                                       ngram_range=(1, 3), stop_words = 'english', max_df=th_max, min_df=th_min) #Removing very rare words (3) and Removing very frequent words (90%)
    
    for txt in candidates_train:
        vectorizer_tfidf.fit_transform(txt)
        
    
    return vectorizer_tfidf

def tf_idf_aux(vectorizer_tfidf,doc_test):      
    testvec  = vectorizer_tfidf.transform(doc_test)                     
    #print(testvec.toarray())
    #find maximum for each of the terms over the dataset
    max_val = testvec.max(axis=1).toarray().ravel()
   
    feature_names = vectorizer_tfidf.get_feature_names()
  
    testvec = tf_idf_scores(testvec, feature_names,chars_or_words="words", scale_factor=20)
    
    sort_tfidf = max_val.argsort()
 
    result = list()
    for i in sort_tfidf[-5:]:
        result.append(feature_names[i])
    
    print(result)
    return result
    
    #Vocab of all docs
    #print('vocab: ', type(vectorizer_tfidf.vocabulary_))
    
    #Retrieve the idf of every term of the vocabulary (all docs)
    #print('tf : ', vectorizer_tfidf.idf_)   
           
def tf_idf(train, test):
    candidates_train = list()
    candidates_test = list()
    
    for doc in train:
        aux = text_preprocess(doc)
        candidates_train = candidates_train + [aux]
    tf_idf_vectorizer = tf_idf_train(candidates_train)   
    for doc in test:
        doc = text_preprocess(doc)
        candidates_test = candidates_test + [aux]
    
    for doc in candidates_test:
        tf_idf_aux(tf_idf_vectorizer,doc)
    
    
   
    
def text_preprocess(corpus, lemma=False, phrases=True):
    candidates =[]
    
    #Phrase division
    phrases = nltk.sent_tokenize(corpus)
 
    print(phrases)
 
  
    candidates = sentence_preprocess(phrases)
    
    return candidates

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

def tf_idf_scores(testvec, feature_names, chars_or_words = 'words', scale_factor=1):
    #print("type", type(testvec))
    #print("before >>>>testvec", testvec.toarray())

    testvec = testvec.toarray()
    
    for i in range(0, testvec.shape[0]):
        for j in range(0, testvec.shape[1]):
            if testvec[i,j] != 0:
                if chars_or_words == 'chars':
                    testvec[i,j] = testvec[i,j] * len(feature_names[j]) * scale_factor       
                    
                elif chars_or_words == 'words':
                    testvec[i,j] =  testvec[i,j] * len(feature_names[j].split()) * scale_factor
     
    #print("after >>>>testvec", testvec)
    testvec = sparse.csr_matrix(testvec)
    return testvec
                  
#def n_gram(n, feature_name):
#    n_grams_list = list()
    
#    for gram in feature_name:
#        if len(nltk.word_tokenize(gram)) == n:
#            if is_right_type(n, gram):
#                n_grams_list.append(gram)
    
#    return n_grams_list

#def is_right_type(n, gram):
#    tagged_grams = nltk.pos_tag_sents(gram)
    
#    if n == 2:
        
#    if n == 3:    
    
    
    
    

    
