# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:43:00 2019

@author: anama
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle
import math
import random
import itertools

exercise3 = __import__('exercise-3')
exercise2 = __import__('exercise-2')

class candidate:
    def __init__(self, doc_name, name, revelancy, position, length_candidate, doc_size, bm25 = 0, tfidf=0, term_frequency=0, inverse_document_frequency=0, zone=0):
        self.name = name
        self.doc_name = doc_name
        self.position = position
        self.length_candidate = length_candidate
        self.doc_size = doc_size
        self.bm25 = bm25
        self.zone = zone
        self.term_frequency = term_frequency
        self.inverse_document_frequency = inverse_document_frequency
        self.tfidf = tfidf
        self.revelancy = revelancy
    
    def get_doc_name(self):
        return self.doc_name
    
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

def tf_idf_tokens_fit(docs, vocab):
    tfidf = TfidfVectorizer(vocabulary = vocab,
                            use_idf = True, 
                            analyzer = 'word', 
                            ngram_range=(1,3), 
                            stop_words = 'english',
                            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]*[a-zA-Z]\b", 
                            lowercase = True,
                            max_df = 3,
                            norm = 'l2')  
    
    tfidf.fit_transform(docs)
    
    return tfidf

def get_tfidf_candidate(vector, index):
    return vector.toarray()[0][index]
    
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
    doc_names = []
    for candidate in candidates:
            doc_names.append(candidate.get_doc_name())
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
        
        
           
    d = {'revelancy': revs, 'name': names, 'docs_name': doc_names,  'position': positions, 'length candidate': length_candidates, 'doc size': doc_sizes, 'bm25':bm25, 'tfidf': tfidf, 'term frequency': tf, 'inverse document frequency': idf, 'zone':  zones}
    df = pd.DataFrame(data=d)
    return df

def extract_target(df):
    return df['revelancy'].tolist()

def extract_data(df):
    #remove revelancy
    X = np.array(df.iloc[:,1:])
    return X

def train(classifier, X, y):
    X_train = X[:, 2:]
    y_train = y
    classifier.fit(X_train, y_train)

    return classifier


def pickle_file():
    docs = exercise3.get_dataset("train", t="lemma")
    with open("corpus.txt", "wb") as fp:   #Pickling
        pickle.dump(docs, fp)
        
def unpickle_file():
    with open("corpus.txt", "rb") as fp:   # Unpickling
        docs = pickle.load(fp)
    return docs

def balance_train_set(candidates, no_relevants):
    from random import randrange
    threshold = randrange(20)
    no_not_relevants = len(candidates) - no_relevants
    no_to_remove = no_not_relevants - no_relevants + threshold
    positions_to_remove = []  #positions not relevant that can be removed
    idx = 0
    for candidate in candidates:
        if(candidate.get_revelancy() == 0):
                positions_to_remove.append(idx)
        idx += 1
    size_positions = len(positions_to_remove)
    
    idx_remove = list()
    for i in range(0, no_to_remove):
        rnd_index = random.randint(0, size_positions - 1)
        idx_remove.append(positions_to_remove[rnd_index])
    
    candidates = [i for j, i in enumerate(candidates) if j not in idx_remove]
    
    return candidates
   
def do_candidates(docs, true_labels, vec, stop, start = 0, is_train = True):
    candidates = list()
    docs = dict(itertools.islice(docs.items(), start, stop)) 
    
    no_relevants = 0
    for key, tokens in docs.items():
        position = 0
        doc_name = key
      
        string_not_tokenized = " ".join(tokens)
      
        
        for token in tokens:
            relevancy = 0
            if token in true_labels[key]:
                relevancy = 1
                no_relevants += 1
           
            vector = vec.transform([string_not_tokenized])
           

            tfidf = get_tfidf_candidate(vector, position)
          
            
            candidates.append(candidate(doc_name, token, relevancy, position, len(" ".split(token)), len(tokens), tfidf=tfidf))
            position += 1  
            
    if(is_train == True):
        candidates = balance_train_set(candidates, no_relevants)
        
    return candidates
        
def get_keyphrases(y_pred, feature_names):
    keyphrases = list()
    i = 0
    for label in y_pred:
        if label == 1:
            keyphrases.append(feature_names[i])
            i += 1

    return keyphrases

def do_classifiers():
       clfs = dict()
     
       clfs["KNN"] = KNeighborsClassifier(n_neighbors=5)
       clfs["Logistic Regression"] = LogisticRegression(random_state=0, class_weight='balanced', multi_class='ovr')
       clfs["SGD Classifier"] = SGDClassifier(max_iter=1000, tol=1e-3)
       return clfs
def main():
    docs = unpickle_file()
    
    doc_not_tokenized = []
    vocab = []
    for tokens in docs.values():
        vocab += tokens
        doc_not_tokenized.append(" ".join(tokens)) 
        
    vocab = list(set(vocab))
    
    vec = tf_idf_tokens_fit(doc_not_tokenized, vocab)
   
    true_labels = exercise2.json_references(stem_or_not_stem = "not stem")
    
    #25% of documents are used to test
    test_size = 0.25
    size_docs = len(docs)
    train_size = 1 - test_size
    index_train = math.floor(train_size * size_docs )
    
    #do candidates object
    candidates_train = do_candidates(docs, true_labels, vec, stop = index_train)
    candidates_test = do_candidates(docs, true_labels, vec, len(docs), start = index_train, is_train = False)
    
    #do separate dataframes for train set and test set
    df_train = do_dataframe(candidates_train)
    df_test = do_dataframe(candidates_test)
    
    y = extract_target(df_train)
    X = extract_data(df_train)
    
    X_test = extract_data(df_test)
    feature_names_all = X_test[:,0]
  
 
    files_to_test = dict()
    for line in X_test[:, 1:]:
        file_name = line[0]
        candidate = list(line[1:])
        if file_name not in files_to_test.keys():    
            files_to_test[file_name] = [candidate]
        else:     
            files_to_test[file_name].append(candidate)
  
    y_test_all = extract_target(df_test)
 
    clfs = do_classifiers()
    
    #train classifiers
    for name, clf in clfs.items():
            print("classifier_name>>>", name)
            clfs[name] = train(clf, X, y)
            
    #test classifiers
    idx_y_test_start = 0
    idx_y_test_stop = 0
    print("len all feature", len(feature_names_all))
    print("len keyphrases true", len(y_test_all))
    print("len", len(files_to_test.keys()))
    for file_name, candidates in files_to_test.items():
        print("file_name>>>", file_name)
        X_test = np.array(candidates)
        
        idx_y_test_stop += len(candidates)
        y_test = y_test_all[idx_y_test_start:idx_y_test_stop]
        feature_names = feature_names_all[idx_y_test_start:idx_y_test_stop]
        idx_y_test_start = idx_y_test_stop
        
        keyphrases_true = get_keyphrases(y_test, feature_names)
        print("keyphrases true", keyphrases_true)
      
        for name, clf in clfs.items():
            print("classifier_name>>>", name)
            y_pred = clf.predict(X_test)
      
            keyphrases = get_keyphrases(y_pred, feature_names)
            print("keyphrases", keyphrases)
            
            count = 0
            for keyphrase in keyphrases:
                if keyphrase in keyphrases_true:
                    count += 1
            print("count", count)
            y_score = list()
            if(name == "KNN"):
                y_score = clf.predict_proba(X_test)
            else:
                y_score = clf.decision_function(X_test)
            idx_features = np.argsort(y_score)
            top_5_keyphrases = list()
            for idx in idx_features[-5:]:
                top_5_keyphrases.append(feature_names[idx])
            print("top_5_keyphrases", top_5_keyphrases)
            print("keyphrases", keyphrases)
       
   
    
   