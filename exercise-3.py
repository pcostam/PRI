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
import math
from scipy.sparse import lil_matrix
class Collection:
    def __init__(self):
        self.documents = dict()
        self.terms = dict()
        self.zones = list()
        self.unique_index_doc = 0
        self.unique_index_term = 0
        
    def add_term(self, name, document_name):  
     
        if name not in self.terms.keys():
            self.terms[name] = Term(name, self.unique_index_term)
            self.terms[name].inc_no_docs()
            self.unique_index_term += 1
            self.documents[document_name].add_term(name)
            
        elif(name in self.terms.keys()) and (name not in self.documents[document_name].vocabulary.keys()):
            self.documents[document_name].add_term(name)
            self.terms[name].inc_no_docs()
            
        elif(name in self.documents[document_name].vocabulary.keys()):
              self.documents[document_name].add_term(name)

        return self.terms[name]
        
         
            
    def add_document(self, name):
        if name not in self.documents.keys():
            self.documents[name] = Document(name, self.unique_index_doc)
            self.unique_index_doc += 1
            
            return self.documents[name]
        else:
            return None
        
    def get_documents_values(self):
        return self.documents.values()
    def get_documents(self):
        return self.documents
    def get_terms_values(self):
        return self.terms.values()
    def get_terms(self):
        return self.terms
    def get_zones(self):
        return self.zones
    def add_zone(self, zone):
        return self.zones.append(zone)
        
      
count_x = 1      
class Document:
    def __init__(self, doc_name,index, vocabulary=""):
        self.doc_name = doc_name
        self.vocabulary = dict()
        self.index = index
        self.len_doc = 0
        #list of object zone
        self.zones = list()
    def add_term(self, new_term):
        global count_x
        """
        Add term. Count occurences, inside document
        """

        if new_term not in self.vocabulary.keys():
            self.vocabulary[new_term] = 1
            
    
        else:
            if(new_term == "resource" and self.doc_name == "C-41"):
                count_x += 1
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
        
    def get_zones(self):
        return self.zones
    
    def set_vocabulary(self, vocabulary):
        self.vocabulary = vocabulary
        
    def get_vocabulary(self):
        return self.vocabulary
    def add_zone(self, zone):
        self.zones.append(zone)
 
 
        
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
        self.tfidf_matrix = lil_matrix((self.no_docs, self.no_terms), dtype=float)
     
    def TF(self, freq_term_i, len_doc_j, k1=1.2, b=0.75):
        upper = freq_term_i * (k1 + 1)
        below = freq_term_i + k1 * (1 - b + b*(abs(len_doc_j)/self.DocAvgLen))
    
        return upper/below
    def IDF(self, no_docs_with_term_i): 
        return math.log(1 + (self.no_docs - no_docs_with_term_i + 0.5)/(no_docs_with_term_i + 0.5))
    
    def TFIDF(self, freq_term_i, len_doc_j, no_docs_with_term_i):
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
        print("test_vector col", test_vector.col)
        tuples_terms = sorted(tuples, key=lambda x: x[1], reverse=True)
        sorted_terms = tuples_terms[0:5]
        print("sorted_terms", sorted_terms)
        keyphrases = []
        for idx, score in sorted_terms:
            keyphrases.append(self.terms[idx].name)
        return keyphrases
        
        
    def calc_prediction(self, doc):
        test_vector = self.tfidf_matrix[doc.index,:]
        keyphrases = self.get_top_5(test_vector.tocoo())
        return keyphrases
        
class Zone:
     def __init__(self, weight, size):
         self.size = size
         self.weight = weight
         #term name : count
         self.term_frequency = dict()
         self.b = 0.75
         
     def get_frequency(self, term_name):
        if term_name in self.term_frequency.keys(): 
            return self.term_frequency[term_name]
        else:
            return 0
    
     def add_term(self, term_name):
        if term_name in self.term_frequency.keys():
            self.term_frequency[term_name] += 1
        else:
            self.term_frequency[term_name] = 1

class BM25F(BM25):
    def __init__(self, zones, docs, terms):
        size_zone = 0
        for zone in zones:
            size_zone += zone.size
        self.avgdl_zone = size_zone/len(zones) 
        #documents   -- collection of documents. List of object Document
        self.docs = docs
        self.terms = terms
        self.no_docs = len(docs)
        self.no_terms = len(terms)
        self.tfidf_matrix = lil_matrix((self.no_docs, self.no_terms), dtype=float)
        
    def TF(self, doc, term, k1=1.5):
        total_sum = 0
        for zone in doc.get_zones():
            tf = zone.weight * zone.get_frequency(term.name)
            avgdl = self.avgdl_zone
            dl = zone.size * zone.weight
            b = zone.b
            upper = tf * (k1 + 1)
            below = tf + k1 * (1 - b + b*((dl)/avgdl))
            total_sum += upper/below
        return total_sum
    
    def TFIDF(self, doc, term, no_docs_with_term_i, k1=1.2, b=0.75):
        return self.TF(doc, term)*self.IDF(no_docs_with_term_i)
    
    def update_matrix_tfidf(self):
        print("UPDATE>>>")
        for doc in self.docs:
            for term in self.terms:
                no_docs_with_term_i = term.get_no_documents()
                tfidf = self.TFIDF(doc, term, no_docs_with_term_i)
                self.tfidf_matrix[doc.index, term.index] = tfidf
   
    
def preprocess_word(word):
    word = word.lower()
  
    return word

def is_valid(word):
    remove = string.punctuation
    remove = remove.replace("-", "") # don't remove hyphens
    pattern_ponctuation = r'[{}]'.format(remove) # create the pattern
    if re.match(pattern_ponctuation, word) or re.match(r'[^\D]', word) or re.match(r'[\n]', word):
        return False
    else:
        return True
 
def parse(tagged_words, collection, document, zone=None):
    grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
    cp = RegexpParser(grammar)
 

    parsed_word_tag = cp.parse(tagged_words)
     
            
    for child in parsed_word_tag:
            if isinstance(child, Tree):               
                if child.label() == 'KT':
                    group_words = []
                    size_child = len(child)
                    if size_child == 2 or size_child == 3:
                        for num in range(size_child):
                                group_words.append(child[num][0])
                        n_gram = " ".join(group_words)
                        n_gram = exercise1.preprocess(n_gram)
                        collection.add_term(n_gram, document.doc_name)
                        if zone != None:
                             zone.add_term(n_gram)
  
                  
def candidates(tagged_words, collection, document, zone=None):
    bigrams = list()
    trigrams = list()

    if zone != None:
            document.add_zone(zone)
            collection.add_zone(zone)
    
    
    for i  in range(0, len(tagged_words)):
        unigram = tagged_words[i][0]
        unigram = exercise1.preprocess(unigram)
        
        collection.add_term(unigram, document.doc_name)
        if zone != None:
            zone.add_term(unigram)
        
        if(i + 1 < len(tagged_words)):
            bigram = [tagged_words[i], tagged_words[i + 1]]
            bigrams.append(bigram)
          
        if(i + 2 < len(tagged_words)):
            trigram = [tagged_words[i], tagged_words[i + 1], tagged_words[i + 2]]
            trigrams.append(trigram)
            
    
    for gram in trigrams:
        parse(gram, collection, document, zone=zone)
    for gram in bigrams:
        parse(gram, collection, document, zone=zone)
        
            
def xml_tags_document(sentences, document, t="lemma"):
        word_tag_pairs = list()
        # get a list of XML tags from the document and print each one
        for sentence in sentences:
                tokens = sentence.getElementsByTagName("token")
                for token in tokens:
                    word = token.getElementsByTagName(t)[0].firstChild.data
                    document.set_size_doc()
                    #don't consider if stop words or punctuation
                    if(is_valid(word)):
                    
                        tag = token.getElementsByTagName("POS")[0].firstChild.data 
                        word_tag_pair = (word, tag)
                        word_tag_pairs.append(word_tag_pair) 
        return word_tag_pairs

def size_sentences_words(sentences):
     for sentence in sentences:
          size = len(sentence.getElementsByTagName("token"))
     return size
#
#Returns: list of tuples where each corresponds to (word, tag). Each word
# is from train set
def get_tagged(t="word", has_zones=False):
    path = os.path.dirname(os.path.realpath('__file__')) + "\\SemEval-2010\\train"
 
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.xml' in file:
                files.append(os.path.join(r, file))
            
    collection = Collection()    
  
    
    for f in files[:30]:
        text = xml.dom.minidom.parse(f)
        base_name=os.path.basename(f)
        key = os.path.splitext(base_name)[0]
        document = collection.add_document(key) 
        sentences = text.getElementsByTagName("sentence")
     
        
        if has_zones == True:
            text_zone_1 = sentences[:10]
            text_zone_2 = sentences[10:]
            size_1 = size_sentences_words(text_zone_1)
            size_2 = size_sentences_words(text_zone_2)    
              
            zone_1 = Zone(0.8, size_1)
            zone_2 = Zone(0.3, size_2)
            zones = [zone_1, zone_2]
            text_zones = [text_zone_1, text_zone_2]
  
            for zone,text_zone in zip(zones, text_zones): 
                word_tag_pairs = xml_tags_document(text_zone, document)
                candidates(word_tag_pairs, collection, document, zone=zone)
         
        else:
             word_tag_pairs = xml_tags_document(sentences, document)
             candidates(word_tag_pairs, collection, document)
       
      
    #END iterating docs
    
    
  
    return collection

def main():
 
     true_labels = exercise2.json_references()
     print("BM25>>>>")
     collection = get_tagged(t="lemma", has_zones=False)
     documents = collection.get_documents()
     document = documents["C-41"]

     terms = list(collection.get_terms_values())
     
     for term in terms:
         if(term.name == "adaptive resource management"):
             print("ALSO TRUE")
             print("no_docs", term.no_docs)
             print("count", document.get_vocabulary()["adaptive resource management"])
         if(term.name == "adaptive resource"):
             print("second keyword")
         if(term.name == "resource reservation mechanism"):
             print("third keyword")
             print("no_docs", term.no_docs)
             print("count", document.get_vocabulary()["resource reservation mechanism"])
         if(term.name == "resource"):
             print("fourth keyword")
             print("count", document.get_vocabulary()["resource"])
     
     print("fim")
     heuristic = BM25(list(collection.get_documents_values()), list(collection.get_terms_values()))
     heuristic.update_matrix_tfidf()
     doc_test = list(collection.get_documents_values())[0]
     print("TESTING>>>")
     keyphrases = heuristic.calc_prediction(doc_test)
     print("keyphrases", keyphrases)
     print("y_true", true_labels["C-41"])
     
     
     print("BM25F>>>>")
     collection = get_tagged(t="lemma", has_zones=True)
     print("collection", len(collection.get_documents()))
     print("collection", collection.unique_index_term)
     #heuristic = BM25(list(collection.get_documents()), list(collection.get_terms()))
     heuristic = BM25F(collection.get_zones(), list(collection.get_documents_values()),  list(collection.get_terms_values()))
     heuristic.update_matrix_tfidf()
     doc_test = list(collection.get_documents_values())[0]
    
     print("TESTING>>>")
     print("doc_test", doc_test.doc_name)
     keyphrases = heuristic.calc_prediction(doc_test)
     print("keyphrases", keyphrases)
     print("y_true", true_labels["C-41"])
     
     
     
     
     
     

     
                
                
                    