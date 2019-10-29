# -*- coding: utf-8 -*-
"""
Exercise 3
"""
import os
import xml.dom.minidom
from nltk import Tree, RegexpParser
from nltk.corpus import stopwords
exercise2 = __import__('exercise-2')
exercise1 = __import__('exercise-1')
import re
import string
import math
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import TfidfVectorizer 
import numpy as np

def preprocess(text):
    text = text.lower()

   
    return text

class BM25:
    def get_tf_matrix(self, docs, vocabulary):     
        vectorizer = TfidfVectorizer(vocabulary = vocabulary,
                                     use_idf = False, 
                                     analyzer = 'word', 
                                     ngram_range=(1,3), 
                                     stop_words = 'english',
                                     token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]*[a-zA-Z]\b", 
                                     lowercase = True)
        tf_matrix = vectorizer.fit_transform(docs).toarray()   
        
        for idx in range(0, len(self.feature_names)):
            if self.feature_names[idx] == "adaptive resource management":
                print("first keyword", idx)
                print("test", tf_matrix[0,idx])
           
        print(tf_matrix)        
        return tf_matrix
     

    def term_frequency(self, term_index, doc_index):
        return self.tf_matrix[doc_index, term_index] 
    
    def document_length(self, doc_index):
        return np.sum(self.tf_matrix[doc_index, :])
   

    def get_binary_matrix(self, docs, vocabulary):
        vectorizer = TfidfVectorizer(vocabulary = vocabulary,
                                     binary=True,
                                     use_idf=False,
                                     norm=None,
                                     analyzer = 'word', 
                                     ngram_range=(1,3), 
                                     stop_words = 'english',
                                     token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]*[a-zA-Z]\b", 
                                     lowercase = True)
        binary_matrix = vectorizer.fit_transform(docs).toarray()
        self.feature_names = vectorizer.get_feature_names()
        for idx in range(0, len(self.feature_names)):
            if self.feature_names[idx] == "adaptive resource management":
                print("first keyword", idx)
                print("test", binary_matrix[0,idx])
                print("test", np.sum(binary_matrix[:, idx]))
                
     
        return binary_matrix

    def document_frequency(self, term_index):
        return np.sum(self.binary_matrix[:,term_index])
    
    def docAvgLen(self):
        acc = 0
        for doc_index in range(0, self.no_docs):
            acc += np.sum(self.tf_matrix[doc_index, :])
        return acc/self.no_docs
        
    def __init__(self, docs, vocabulary):
        self.docs = docs
        self.length_docs = []
        for doc in self.docs:
            self.length_docs.append(len(doc.split()))
        
        self.no_docs = len(docs) 
        self.feature_names = ""
        self.vocabulary = vocabulary
        self.binary_matrix = self.get_binary_matrix(docs, self.vocabulary)
        self.tf_matrix = self.get_tf_matrix(self.docs, self.vocabulary)
        self.no_terms = self.binary_matrix.shape[1]
        self.tfidf_matrix = lil_matrix((self.no_docs, self.no_terms), dtype=float)
        self.DocAvgLen = self.docAvgLen()
   
      
     
    def TF(self, freq_term_i, len_doc_j, k1=1.5, b=0.75):
        upper = freq_term_i * (k1 + 1)
        below = freq_term_i + k1 * (1 - b + b*(abs(len_doc_j)/self.DocAvgLen))
    
        return upper/below
    def IDF(self, nt): 
        return math.log((self.no_docs - nt + 0.5)/(nt + 0.5))
    
    def TFIDF(self, freq_term_i, len_doc_j, nt):
        return self.TF(freq_term_i, len_doc_j)*self.IDF(nt)
    
    def update_matrix_tfidf(self):
        print("UPDATE>>>")
        for doc_index in range(0, self.no_docs):
            for term_index in range(0, self.no_terms):
                nt = self.document_frequency(term_index)
                tf = self.term_frequency(term_index, doc_index)
                doc_length = self.document_length(doc_index)
                tfidf = self.TFIDF(tf, doc_length, nt)
                self.tfidf_matrix[doc_index, term_index] = tfidf
     
    def get_top_5(self, test_vector):
        tuples = zip(test_vector.col, test_vector.data)
        print("test_vector col", test_vector.col)
        tuples_terms = sorted(tuples, key=lambda x: x[1], reverse=True)
        sorted_terms = tuples_terms[0:5]
        print("sorted_terms", sorted_terms)
        keyphrases = []
        for idx, score in sorted_terms:
            keyphrases.append(self.feature_names[idx])
        return keyphrases
        
        
    def calc_prediction(self, doc_index):
        test_vector = self.tfidf_matrix[doc_index,:]
        keyphrases = self.get_top_5(test_vector.tocoo())
        return keyphrases
    


#
#Returns: list of tuples where each corresponds to (word, tag). Each word
# is from train set
def get_tagged(folder, t="word"):
    path = os.path.dirname(os.path.realpath('__file__')) + "\\SemEval-2010\\" + folder
 
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.xml' in file:
                files.append(os.path.join(r, file))
            

    word_tag_pairs = list()
  
   
    for f in files[:30]:
        doc = xml.dom.minidom.parse(f)

        # get a list of XML tags from the document and print each one
        sentences = doc.getElementsByTagName("sentence")
    
        for sentence in sentences:
                tokens = sentence.getElementsByTagName("token")
                for token in tokens:
                    word = token.getElementsByTagName(t)[0].firstChild.data
                    #don't consider if stop words, punctuation or digits
                    if(is_valid(word)):
                        tag = token.getElementsByTagName("POS")[0].firstChild.data 
                        word_tag_pair = (word,tag)
                        word_tag_pairs.append(word_tag_pair)
  
    return word_tag_pairs

def is_valid(word):
    word = preprocess(word)
    remove = string.punctuation
    pattern_ponctuation = r'[{}]'.format(remove) # create the pattern
  
    stop_words = set(stopwords.words('english')) 
  
    if re.match(pattern_ponctuation, word) or re.match(r'[^\D]', word) or re.match(r'[\n]', word) or (word in stop_words):
        return False
    else:
        return True
 
def parse(tagged_words, n):
    grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
    cp = RegexpParser(grammar)

    parsed_word_tag = cp.parse(tagged_words)  
    results = []
    
    for child in parsed_word_tag:
            if isinstance(child, Tree):               
                if child.label() == 'KT':
                    #group_words = []
                    size_child = len(child)
                    if size_child == n:
                        #for num in range(size_child):
                        #      group_words.append(child[num][0])
                        #n_gram = " ".join(group_words)
                        return True
                else:
                    return False
    return results
                     
  

    
def candidates(tagged_words):
    bigrams = list()
    trigrams = list()
    unigrams = list()
   
    for i  in range(0, len(tagged_words)):
        unigram = [tagged_words[i]]
        if(parse(unigram, 1)):
                ngram = preprocess(tagged_words[i][0])
                unigrams.append(ngram)
                
        if(i + 1 < len(tagged_words)):
            pos_tag = [tagged_words[i], tagged_words[i + 1]]
            if(parse(pos_tag, 2)):
                bigram = [tagged_words[i][0], tagged_words[i + 1][0]]
                ngram = " ".join(bigram)
                ngram = preprocess(ngram)
                bigrams.append(ngram)
          
        if(i + 2 < len(tagged_words)):
            pos_tag = [tagged_words[i], tagged_words[i + 1], tagged_words[i + 2]]
            
            if(parse(pos_tag, 3)):
                trigram = [tagged_words[i][0], tagged_words[i + 1][0], tagged_words[i + 2][0]]
                ngram = " ".join(trigram)
                ngram = preprocess(ngram)
                trigrams.append(ngram)
     
   
    res = list(set(bigrams)) + list(set(trigrams)) + list(set(unigrams))
    
    
    return res
            
def size_sentences_words(sentences):
     for sentence in sentences:
          size = len(sentence.getElementsByTagName("token"))
     return size

#
#Returns: list of tuples where each corresponds to (word, tag). Each word
# is from train set
def main():
 
     true_labels = exercise2.json_references()
     print("BM25>>>>")
     docs, test_set = exercise2.get_dataset("train",t="lemma", test_size=5)
     tagged_words = get_tagged("train", t="lemma")
     vocabulary = candidates(tagged_words)
     algorithm = BM25(docs, vocabulary)
     algorithm.update_matrix_tfidf()
     keyphrases = algorithm.calc_prediction(0)
     print("keyphrases", keyphrases)
     
     
     
     
     
     

     
                
                
                    