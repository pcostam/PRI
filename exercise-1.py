# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import nltk 
import re

stop_words = set(nltk.corpus.stopwords.words("english"))

def main():
    train, test = get_20_news_group()
    tf_idf(train, test)
    
def get_20_news_group():
    train = fetch_20newsgroups(subset = 'train')
    test  = fetch_20newsgroups(subset = 'test' )
    np.random.shuffle(train.data)
    
    return train.data[:30], [test.data[0]]
                    
def tf_idf(train, test):
    candidates_train = list()
    candidates_test = list()
    
    for doc in train:
        aux = text_preprocess(doc)
        candidates_train = candidates_train + [aux]
        
    for doc in test:
        aux = text_preprocess(doc)
        candidates_test = candidates_test + [aux]
    
    #Learn the vocabulary dictionary and return term-document matrix
    vectorizer_tfidf = TfidfVectorizer(use_idf = True, analyzer = 'word', 
                                       ngram_range=(1, 3), stop_words = 'english')
    
    for txt in candidates_train:
        vectorizer_tfidf.fit_transform(txt)
    testvec  = vectorizer_tfidf.transform(candidates_test[0])
    
    #find maximum for each of the terms over the dataset
    max_val = testvec.max(axis=0).toarray().ravel()
    feature_names = vectorizer_tfidf.get_feature_names()
    
    tf_idf_scores(testvec, feature_names)
    
    sort_tfidf = max_val.argsort()
    
    for i in sort_tfidf[-5:]:
        print(feature_names[i])
    
    #Vocab of all docs
    #print('vocab: ', type(vectorizer_tfidf.vocabulary_))
    
    #Retrieve the idf of every term of the vocabulary (all docs)
    #print('tf : ', vectorizer_tfidf.idf_)
    
def text_preprocess(corpus, lemma=False, phrases=True):
    candidates =[]
    
    #Pass to lower case
    corpus = corpus.lower()

    #Phrase division
    phrases = nltk.sent_tokenize(corpus)
    
    for phrase in phrases:    
        #Remove puntuation
        phrase = re.sub(r'[^\w\s]','',phrase)
        phrase = re.sub(r'[^\D]'  ,'',phrase)
        phrase = re.sub(r'[\n]',' ', phrase)
        candidates.append(phrase)
        
    return candidates
   
def tf_idf_scores(testvec, feature_names, chars_or_words = 'words'):
    for row in testvec.toarray():
        j = 0
        for cell in row:
            j = j + 1
            if cell != 0:
                
                if chars_or_words == 'chars':
                    cell = cell * len(feature_names[j])
                    
                elif chars_or_words == 'words':
                    cell =  cell * len(feature_names[j].split())
                  
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
    
    
    
    

    