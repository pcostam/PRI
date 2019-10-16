# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer 
from sklearn.datasets import fetch_20newsgroups
import nltk 
import sys
import re

def read_file(filename):
    try:
        fp = open(filename, 'r')
        corpus = fp.read()
    
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
    except ValueError:
        print("Could not convert data to an integer.")
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
    finally:
        fp.close()

def text_preprocess(corpus, lemma=False, phrases=False):
    stop_words = set(nltk.corpus.stopwords.words("english"))
    candidates =[]
    
    #Pass to lower case
    corpus = corpus.lower()
    
    if phrases:
        #Phrase division
        phrases = nltk.sent_tokenize(corpus)
        
        for phrase in phrases:
            filtered_sentence = []
            #Remove puntuation
            phrase = re.sub(r'[^\w\s]','',phrase)
            phrase = re.sub(r'[^\D]'  ,'',phrase)
            #Tokenize text
            word_tokens = nltk.word_tokenize(phrase)
        
            #Remove Stop Words
            if not lemma:
                for w in word_tokens:
                    if w not in stop_words: 
                           filtered_sentence.append(w)
                candidates.append(filtered_sentence)
    else:
        #Remove puntuation
        corpus = re.sub(r'[^\w\s]','',corpus)
        corpus = re.sub(r'[^\D]'  ,'',corpus)
        
        #Tokenize text
        word_tokens = nltk.word_tokenize(corpus)
        
        #Remove Stop Words
        if not lemma:
            for w in word_tokens:
                if w not in stop_words: 
                    candidates.append(w)
                    
    
    candidates = " ".join(candidates)

    return candidates
                    
def tf_idf(train, test):
    candidates = []
    for doc in train:
        aux = text_preprocess(doc)
        candidates = candidates + [aux]
        
    vectorizer = CountVectorizer()
    
    #Learn the vocabulary dictionary and return term-document matrix (count)
    x = vectorizer.fit_transform(candidates)
    vectorizer_tfidf = TfidfVectorizer(use_idf = True, analyzer = 'word')
    
    trainvec = vectorizer_tfidf.fit_transform(train)
    testvec  = vectorizer_tfidf.transform(test)
    
    #find maximum for each of the terms over the dataset
    max_val = testvec.max(axis=0).toarray().ravel()
    
    sort_tfidf = max_val.argsort()
    print("sort", sort_tfidf)
    
    feature_names = vectorizer_tfidf.get_feature_names()
    print("sort_tfidf", sort_tfidf[-5:])
    print("fn", type(feature_names))
   
    for i in sort_tfidf[-5:]:
        print(feature_names[i])
    
    #Vocab of all docs
    #print('vocab: ', vectorizer_tfidf.vocabulary_)
    
    #Retrieve the idf of every term of the vocabulary (all docs)
    #print('tf : ', vectorizer_tfidf.idf_)
    #print(trainvec)
    
    
def get_20_news_group():
    train = fetch_20newsgroups(subset = 'train')
    test  = fetch_20newsgroups(subset = 'test' )
    
    return train.data[:10], test.data[:10]

def main():
    train, test = get_20_news_group()
    tf_idf(train, test)
    
    
    
    
    
    
    

    