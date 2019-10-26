# -*- coding: utf-8 -*-
"""
Exercise 3
"""
import os
import xml.dom.minidom
from nltk import Tree, RegexpParser
exercise2 = __import__('exercise-2')
exercise1 = __import__('exercise-1')
import re
import string
import numpy as np
import math
from scipy.sparse import csr_matrix, lil_matrix
class Collection:
    def __init__(self):
        self.documents = dict()
        self.terms = dict()
        self.unique_index_doc = 0
        self.unique_index_term = 0
        
    def add_term(self, name, document_name):
     
        if name not in self.terms.keys():
            self.terms[name] = Term(name, self.unique_index_term)
            self.terms[name].inc_no_docs()
            self.unique_index_term += 1
            self.documents[document_name].add_term(name)
            
        if (name in self.terms.keys()) and (name not in self.documents[document_name].vocabulary.keys()):
            self.documents[document_name].add_term(name)
            self.terms[name].inc_no_docs()
        
        return self.terms[name]
        
         
            
    def add_document(self, name):
        if name not in self.documents.keys():
            self.documents[name] = Document(name, self.unique_index_doc)
            self.unique_index_doc += 1
            
            return self.documents[name]
        else:
            return None
        
    def get_documents(self):
        return self.documents.values()
        
    def get_terms(self):
        return self.terms.values()
        
      
        
class Document:
    def __init__(self, doc_name,index, vocabulary=""):
        self.doc_name = doc_name
        self.vocabulary = dict()
        self.index = index
        self.len_doc = 0
    def add_term(self, new_term):
        """
        Add term. Count occurences, inside document
        """

        if new_term not in self.vocabulary.keys():
            self.vocabulary[new_term] = 1
            
    
        else:
            self.vocabulary[new_term] += 1
    #frequence term i in doc j - fi,j
    def frequence_term(self, term):
        if term.name in self.vocabulary.keys(): 
            return self.vocabulary[term.name]    
        else:
            return 0
    def words_size_doc(self):
        return self.len_doc
    def set_size_doc(self):
        self.len_doc += 1
    
    def set_vocabulary(self, vocabulary):
        self.vocabulary = vocabulary
        
    def get_vocabulary(self):
        return self.vocabulary
 
 
        
class Term:
    def __init__(self, name,index,no_docs=0):
        self.no_docs = no_docs
        self.name = name
        self.index = index
    
   
    #number of docs containing term i  - ni
    def get_no_documents(self):
        return self.no_docs
    def inc_no_docs(self):
        self.no_docs += 1
    
class BM25:
    def __init__(self, docs, terms):
        size_doc = 0
        for doc in docs:
            size_doc += doc.words_size_doc()
        self.DocAvgLen = size_doc/len(docs) 
        #documents   -- collection of documents. List of object Document
        self.docs = docs
        self.terms = terms
        self.no_docs = len(docs)
        self.no_terms = len(terms)
        self.tfidf_matrix = lil_matrix((self.no_docs, self.no_terms), dtype=int)
     
    def TF(self, freq_term_i, len_doc_j, k1=1.5, b=0.75):
        upper = freq_term_i * (k1 + 1)
        below = freq_term_i + k1 * (1 - b + b*(abs(len_doc_j)/self.DocAvgLen))
        
        return upper/below
    def IDF(self, no_docs_with_term_i): 
        return math.log((self.no_docs - no_docs_with_term_i + 0.5)/(no_docs_with_term_i + 0.5))
    
    def TFIDF(self, freq_term_i, len_doc_j, no_docs_with_term_i, k1=1.5, b=0.75):
        return self.TF(freq_term_i, len_doc_j)*self.IDF(no_docs_with_term_i)
    
    def update_matrix_tfidf(self):
        print("UPDATE>>>")
        for doc in self.docs:
            for term in self.terms:
                no_docs_with_term_i = term.get_no_documents()
                len_doc_j = doc.words_size_doc()
                freq_term_i = doc.frequence_term(term)
                tfidf = self.TFIDF(freq_term_i, len_doc_j, no_docs_with_term_i)
                self.tfidf_matrix[doc.index, term.index] = tfidf
       
    
    def get_top_5(self, test_vector):
        tuples = zip(test_vector.col, test_vector.data)
        tuples_terms = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
        sorted_terms = tuples_terms[0:5]
        keyphrases = []
        for idx, score in sorted_terms:
            keyphrases.append(self.terms[idx].name)
        return keyphrases
        
        
    def calc_prediction(self, doc):
        test_vector = self.tfidf_matrix[doc.index,:]
        keyphrases = self.get_top_5(test_vector.tocoo())
        return keyphrases
        
        
def preprocess_word(word):
    word = word.lower()
  
    return word

def is_valid(word):
    remove = r"[{}]".format(string.punctuation)
    remove = remove.replace("-", "") # don't remove hyphens
    pattern_ponctuation = r"[{}]".format(remove) # create the pattern
    if re.match(pattern_ponctuation, word) or re.match(r'[^\D]', word) or re.match(r'[\n]', word):
        return False
    else:
        return True
   
  
       
#
#Returns: list of tuples where each corresponds to (word, tag). Each word
# is from train set
def get_tagged(t="word"):
    path = os.path.dirname(os.path.realpath('__file__')) + "\\SemEval-2010\\train"
 
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.xml' in file:
                files.append(os.path.join(r, file))
            
    collection = Collection()    
    word_tag_pairs = list()
    
    for f in files[:30]:
        text = xml.dom.minidom.parse(f)
        base_name=os.path.basename(f)
        key = os.path.splitext(base_name)[0]
        document = collection.add_document(key) 

        # get a list of XML tags from the document and print each one
        sentences = text.getElementsByTagName("sentence")
    
        for sentence in sentences:
                tokens = sentence.getElementsByTagName("token")
                for token in tokens:
                    word = token.getElementsByTagName(t)[0].firstChild.data
                    #don't consider if stop words or punctuation
                    if(is_valid(word)):
                        document.set_size_doc()
                        tag = token.getElementsByTagName("POS")[0].firstChild.data 
                        word_tag_pair = (word,tag)
                        word_tag_pairs.append(word_tag_pair)
        #end iterating sentences
        #Generate valid grams - vocabulary for each document
     
        grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
        cp = RegexpParser(grammar)
        parsed_word_tag = cp.parse(word_tag_pairs)
     
        for child in parsed_word_tag:
            if isinstance(child, Tree):               
                if child.label() == 'KT':
                    group_words = []
                    size_child = len(child)
                    if size_child <= 3:
                        for num in range(size_child):
                                group_words.append(child[num][0])
                        n_gram = " ".join(group_words)
                        n_gram = exercise1.preprocess(n_gram)
                        collection.add_term(n_gram, document.doc_name)
        #end iterating tree
      
    #END iterating docs
    
    
  
    return collection

def main():
     true_labels = exercise2.json_references()
     collection = get_tagged(t="lemma")
     print("collection", len(collection.get_documents()))
     print("collection", collection.unique_index_term)
     heuristic = BM25(list(collection.get_documents()), list(collection.get_terms()))
     heuristic.update_matrix_tfidf()
     doc_test = list(collection.get_documents())[0]
     print("TESTING>>>")
     print("doc_test", doc_test.doc_name)
     keyphrases = heuristic.calc_prediction(doc_test)
     print("keyphrases", keyphrases)
     print("y_true", true_labels)
     
     
     
     
     

     
                
                
                    