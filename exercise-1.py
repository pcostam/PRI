# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import nltk 
import re
from scipy import sparse

stop_words = set(nltk.corpus.stopwords.words("english"))

def main():
    train, test = get_20_news_group()
    tf_idf(train, test)
    
def get_20_news_group():
    train = fetch_20newsgroups(subset = 'train', remove=('headers', 'footers', 'quotes')) #The F-score will be lower because it is more realistic.
    test  = fetch_20newsgroups(subset = 'test', remove=('headers', 'footers', 'quotes') ) #The F-score will be lower because it is more realistic.
    np.random.shuffle(train.data)
    
    return train.data[:30], [test.data[0]]

def tf_idf_aux(candidates_train,doc_test): 
    #Learn the vocabulary dictionary and return term-document matrix
    vectorizer_tfidf = TfidfVectorizer(use_idf = True, analyzer = 'word', 
                                       ngram_range=(1, 3), stop_words = 'english')
    
    for txt in candidates_train:
        vectorizer_tfidf.fit_transform(txt)
    testvec  = vectorizer_tfidf.transform(doc_test)                     
    
    #find maximum for each of the terms over the dataset
    max_val = testvec.max(axis=0).toarray().ravel()
    feature_names = vectorizer_tfidf.get_feature_names()
    
    testvec = tf_idf_scores(testvec, feature_names)
    
    print("test_vec", testvec.toarray())
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
        
    for doc in test:
        doc = text_preprocess(doc)
        candidates_test = candidates_test + [aux]
    
    for doc in candidates_test:
        tf_idf_aux(candidates_train, doc)
    
    
   
    
def text_preprocess(corpus, lemma=False, phrases=True):
    candidates =[]
    
    #Phrase division
    phrases = nltk.sent_tokenize(corpus)
    
    candidates = sentence_preprocess(phrases)
    
    return candidates

def sentence_preprocess(phrases):
    candidates =[]
    for phrase in phrases:  
        phrase = phrase.lower()
        #Remove puntuation
        phrase = re.sub(r'[^\w\s]','',phrase)
        phrase = re.sub(r'[^\D]'  ,'',phrase)
        phrase = re.sub(r'[\n]',' ', phrase)
        candidates.append(phrase)
        
    return candidates

def tf_idf_scores(testvec, feature_names, chars_or_words = 'words'):
    print("type", type(testvec))
    print("before >>>>testvec", testvec.toarray())
    testvec = testvec.toarray()
    for i in range(0, testvec.shape[0]):
        for j in range(0, testvec.shape[1]):
            
            if testvec[i,j] != 0:
                if chars_or_words == 'chars':
                    testvec[i,j] = testvec[i,j] * len(feature_names[j])
            
                    
                elif chars_or_words == 'words':
                    testvec[i,j] =  testvec[i,j] * len(feature_names[j].split())
                    
    
    print("after >>>>testvec", testvec)   
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
    
    
    
    

    
