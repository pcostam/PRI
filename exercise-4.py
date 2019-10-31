# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:43:00 2019

@author: anama
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import numpy as np
import pickle
import math
exercise3 = __import__('exercise-3')
exercise2 = __import__('exercise-2')

class candidate:
    def __init__(self, name, revelancy, position, length_candidate, doc_size, bm25 = 0, tfidf=0, term_frequency=0, inverse_document_frequency=0, zone=0):
        self.name = name
        self.position = position
        self.length_candidate = length_candidate
        self.doc_size = doc_size
        self.bm25 = bm25
        self.zone = zone
        self.term_frequency = term_frequency
        self.inverse_document_frequency = inverse_document_frequency
        self.tfidf = tfidf
        self.revelancy = revelancy
    
    def get_name(self):
        return self.name
    
    def get_position(self):
        return self.position
    
    def get_length_candidate(self):
        return self.length_candidate
    
    def get_doc_size(self):
        return self.doc_size
    
    def get_bm25(self):
        return self.bm25
    
    def get_zone(self):
        return self.zone
    
    def get_term_frequency(self):
        return self.term_frequency 
        
    def get_inverse_document_frequency(self):
        return self.inverse_document_frequency 
    
    def get_tfidf(self):
        return self.tfidf 
    
    def get_revelancy(self):
        return self.revelancy
    

def dummy_fun(doc):
    return doc

def tf_idf_tokens_fit(docs):
    tfidf = TfidfVectorizer(analyzer='word',
                            tokenizer=dummy_fun,
                            preprocessor=dummy_fun,
                            token_pattern=None)  
    
    tfidf.fit_transform(docs)
    return tfidf

def get_tfidf_candidate(vector, index):
    return vector.toarray()[index]
    
"""
Returns pandas dataframe
"""  
def do_dataframe(candidates):
    positions = []
    length_candidates = []
    doc_sizes = []
    bm25 = []
    tfidf = []
    idf = []
    tf = []
    zones = []
    revs = []
    names = []
    for candidate in candidates:
            names.append(candidate.get_name())
            positions.append(candidate.get_position())
            length_candidates.append(candidate.get_length_candidate())
            doc_sizes.append(candidate.get_doc_size())
            bm25.append(candidate.get_bm25())
            tfidf.append(candidate.get_tfidf())
            tf.append(candidate.get_term_frequency())
            idf.append(candidate.get_inverse_document_frequency())
            zones.append(candidate.get_zone())
            revs.append(candidate.get_revelancy())
        
        
           
    d = {'revelancy': revs, 'name': names, 'position': positions, 'length candidate': length_candidates, 'doc size': doc_sizes, 'bm25':bm25, 'tfidf': tfidf, 'term frequency': tf, 'inverse document frequency': idf, 'zone':  zones}
    df = pd.DataFrame(data=d)
    return df

def extract_target(df):
    print("df revelancy", df['revelancy'].tolist())
    return df['revelancy'].tolist()

def extract_data(df):
    X = np.array(df.iloc[:,1:])
    return X

def train(classifier, X, y):
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
    #feature_names = X_test[:,0]
    #print(feature_names)
    #X_train = X_train[:,1:]
    #X_test = X_test[:, 1:]
    X_train = X[:, 1:]
    y_train = y
    classifier.fit(X_train, y_train)
    
    #print("Accuracy: " + str(classifier.score(X_test, y_test)))
    return classifier


def pickle_file():
    docs = exercise3.get_dataset("train", t="lemma")
    with open("corpus.txt", "wb") as fp:   #Pickling
        pickle.dump(docs, fp)
        
def unpickle_file():
    with open("corpus.txt", "rb") as fp:   # Unpickling
        docs = pickle.load(fp)
    return docs

def df_train_test(docs, true_labels,vec, test_size=0.25):
    import itertools
    size_docs = len(docs)
    train_size = 1 - test_size
    index_train = math.floor(train_size * size_docs )
  
    candidates_train = list()
    candidates_test = list()
    docs_train = dict(itertools.islice(docs.items(), index_train))
    docs_test = dict(itertools.islice(docs.items(), index_train, len(docs)))
   
    for key, tokens in docs_train.items():
        position = 0
        for token in tokens:
            relevancy = 0
            if token in true_labels[key]:
                relevancy = 1
           
            #vector = vec.transform(tokens)
         
            
            #tfidf = get_tfidf_candidate(vector, position)
            
            candidates_train.append(candidate(token, relevancy, position, len(" ".split(token)), len(tokens)))
            position += 1    
    
    for key, tokens in docs_test.items():
        position = 0
        for token in tokens:
            relevancy = 0
        
            if token in true_labels[key]:
                relevancy = 1
            
            candidates_test.append(candidate(token, relevancy, position, len(" ".split(token)), len(tokens)))
            position += 1    
    
        
    return candidates_train, candidates_test
def get_keyphrases(y_pred, feature_names):
    keyphrases = list()
    i = 0
    for label in y_pred:
        if label == 1:
            keyphrases.append(feature_names[i])
            i += 1

    return keyphrases
    
def main():
    docs = unpickle_file()
    vec = tf_idf_tokens_fit(docs.values())
    true_labels = exercise2.json_references(stem_or_not_stem = "not stem")
    
    candidates_train, candidates_test = df_train_test(docs, true_labels,vec, test_size=0.25)
      
    df_train = do_dataframe(candidates_train)
    df_test = do_dataframe(candidates_test)
    y = extract_target(df_train)
    print("y>>>", np.unique(y))
    X = extract_data(df_train)
    
    clf = Perceptron(tol=1e-3, random_state=0)
    clf = train(clf, X, y)
    
    X_test = extract_data(df_test)
    feature_names = X_test[:,0]
    X_test = X_test[:, 1:]
    y_test = extract_target(df_test)
    print("y_test", np.unique(y_test))
    
    y_pred = clf.predict(X_test)
    
    print("logistic regression")
    clf = LogisticRegression(random_state=0, class_weight='balanced', multi_class='ovr')
    clf = train(clf, X, y)
    
    y_pred = clf.predict(X_test)
    keyphrases = get_keyphrases(y_pred, feature_names)
   
    
    print("Linear SVC")
    clf = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='squared_hinge', max_iter=1000,multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)
    clf = train(clf, X, y)
    
    y_pred = clf.predict(X_test)
    keyphrases = get_keyphrases(y_pred, feature_names)
    print("keyphrases", keyphrases)
  
    
   