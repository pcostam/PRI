# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:43:00 2019

@author: anama
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

class candidate:
    def __init__(self, name, position, length_candidate, doc_size, bm25, tfidf, revelancy, term_frequency=0, inverse_document_frequency=0, zone=0):
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
    
"""
Returns pandas dataframe
"""  
def do_dataframe(self, candidates):
    positions = []
    length_candidates = []
    doc_sizes = []
    bm25 = []
    tfidf = []
    idf = []
    tf = []
    zones = []
    revs = []
    for candidate in candidates:
        positions.append(candidate.get_position())
        length_candidates.append(candidate.get_length_candidate())
        doc_sizes.append(candidate.get_doc_size())
        bm25.append(candidate.get_bm25())
        tfidf.append(candidate.get_tfidf())
        tf.append(candidate.get_term_frequency())
        idf.append(candidate.get_inverse_document_frequency())
        zones.append(candidate.get_zone())
        revs.append(candidate.get_revelancy())
        
        
           
    d = {'revelancy': revs, 'position': positions, 'length candidate': length_candidates, 'doc size': doc_sizes, 'bm25':bm25, 'tfidf': tfidf, 'term frequency': tf, 'inverse document frequency': idf, 'zone':  zones}
    df = pd.DataFrame(data=d)
    return df

def extract_target(df):
    return df['revelancy'].values()

def extract_data(df):
    data = []
    return data

def train(classifier, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

    classifier.fit(X_train, y_train)
    
    print("Accuracy: " + str(classifier.score(X_test, y_test)))
    return classifier, X_test, y_test

def main():
    candidates = []
    df = do_dataframe(candidates)
    X = extract_target(df)
    y = extract_data(df)
    clf = Perceptron(tol=1e-3, random_state=0)
    clf, X_test, y_test = train(clf, X, y)
    y_pred = clf.predict(X_test)
   