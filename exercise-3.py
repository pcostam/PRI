# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 18:48:43 2019

@author: anama
"""
from scipy.sparse import lil_matrix
import math
from six.moves import range
from six import iteritems
import xml.dom.minidom
from nltk import word_tokenize
from nltk import Tree, RegexpParser
from nltk.corpus import stopwords
import re
import string
import os

PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25
class BM25(object):
    def __init__(self, corpus):
        self.corpus_size = 0
        self.doc_len = []
        self.avgdl = 0
        self.occurences_terms = []
        self.idf = {}
        self.terms = []
        self.no_terms = 0
        self.corpus = corpus
        self._initialize(corpus)
    
    def _initialize(self, corpus):
        #number of documents with word
        nd={}
        sum_num_doc = 0
        
        for document in corpus:
                self.corpus_size += 1
                size = len(document)
                self.doc_len.append(size)
                sum_num_doc += size
                
                occurences={}
                for word in document:
                    #i think we dont need this if, we could just pass the terms
                    #pass terms with repetition... do a different tokenizer...
                    if word not in self.terms:
                        self.terms.append(word)
                        self.no_terms +=1
                    if word not in occurences:
                        occurences[word] = 0
                    occurences[word] += 1
                self.occurences_terms.append(occurences)
                
                for word, freq in iteritems(occurences):
                    if word not in nd:
                        nd[word] = 0
                    nd[word] += 1
                    
        self.avgdl = float(sum_num_doc)/self.corpus_size
        
        idf_sum = 0
        #collect words with negative idf to set them a special epsilon value
        #idf can be negative if word is contained in more than half of the documents
        negative_idfs = []
        for word, freq in iteritems(nd):
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = float(idf_sum)/len(self.idf)
        eps = EPSILON  * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps
            
        self.tfidf_matrix = lil_matrix((self.corpus_size, self.no_terms), dtype=float)
            
    def get_tf(self, doc_index, index_term):
        score = 0
 
        word = self.terms[index_term]
        if word in self.occurences_terms[doc_index].keys():
            term_freqs = self.occurences_terms[doc_index][word]
            score = (term_freqs * (PARAM_K1 + 1)
                    / (term_freqs + PARAM_K1 * (1- PARAM_B + PARAM_B * self.doc_len[doc_index]/ self.avgdl)))
        return score
    
    def get_scores(self):
        for doc_index in range(0, self.corpus_size):
            for term_index in range(0, self.no_terms):
                tf = self.get_tf(doc_index, term_index)
                idf = self.idf[self.terms[term_index]]
                tfidf = tf * idf
                self.tfidf_matrix[doc_index, term_index] = tfidf
        print("tfidf_matrix", self.tfidf_matrix)
        return self.tfidf_matrix
            

    
    def get_top_5(self, test_vector):
        tuples = zip(test_vector.col, test_vector.data)
        print("test_vector col", test_vector.col)
        tuples_terms = sorted(tuples, key=lambda x: x[1], reverse=True)
        sorted_terms = tuples_terms[0:10]
        print("sorted_terms", sorted_terms)
        keyphrases = []
        for idx, score in sorted_terms:
            keyphrases.append(self.terms[idx])
        return keyphrases
        
        
    def calc_prediction(self, doc_index):
        test_vector = self.tfidf_matrix[doc_index,:]
        keyphrases = self.get_top_5(test_vector.tocoo())
        return keyphrases          
 
def get_dataset(folder, t="word"):
    path = os.path.dirname(os.path.realpath('__file__')) + "\\SemEval-2010\\" + folder
 
    files = []
    docs = dict()
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.xml' in file:
                files.append(os.path.join(r, file))
           
    for f in files[:30]:
        word_tag_pairs = list()
        text = str()
        base_name=os.path.basename(f)
        key = os.path.splitext(base_name)[0]
        doc = xml.dom.minidom.parse(f)

        # get a list of XML tags from the document and print each one
        sentences = doc.getElementsByTagName("sentence")
        text = ""
        for sentence in sentences:
                tokens = sentence.getElementsByTagName("token")
                sentence_string = ""
                for token in tokens:
                    word = token.getElementsByTagName(t)[0].firstChild.data
                    word = preprocess(word)
                    if(is_valid(word)):
                        #tag = token.getElementsByTagName("POS")[0].firstChild.data
                        #word_tag_pair = (word,tag)
                        #word_tag_pairs.append(word_tag_pair)     
                        sentence_string = sentence_string + " " + word
                #ended iterating tokens from sentence
                text += sentence_string
        #ended iterating sentences
        docs[key] = text
        #review this! not sure
        #terms += candidates(terms, word_tag_pairs)
     
    #terms = list(set(terms))
    

    return docs
 

def candidates(terms, tagged_words):
    bigrams = list()
    trigrams = list()
    unigrams = list()

    for i  in range(0, len(tagged_words)):
        unigram = [tagged_words[i]]
        if(is_valid_semantic(unigram, 1)):
                ngram = preprocess(tagged_words[i][0])
                unigrams.append(ngram)

        if(i + 1 < len(tagged_words)):
            pos_tag = [tagged_words[i], tagged_words[i + 1]]
            if(is_valid_semantic(pos_tag, 2)):
                bigram = [tagged_words[i][0], tagged_words[i + 1][0]]
                ngram = " ".join(bigram)
                ngram = preprocess(ngram)
                bigrams.append(ngram)

        if(i + 2 < len(tagged_words)):
            pos_tag = [tagged_words[i], tagged_words[i + 1], tagged_words[i + 2]]

            if(is_valid_semantic(pos_tag, 3)):
                trigram = [tagged_words[i][0], tagged_words[i + 1][0], tagged_words[i + 2][0]]
                ngram = " ".join(trigram)
                ngram = preprocess(ngram)
                trigrams.append(ngram)


    res = list(set(bigrams)) + list(set(trigrams)) + list(set(unigrams))


    return res

def preprocess(text):
    text = text.lower()


    return text


def is_valid(word):
    word = preprocess(word)
    remove = string.punctuation
    pattern_ponctuation = r'[{}]'.format(remove) # create the pattern

    stop_words = set(stopwords.words('english'))

    if re.match(pattern_ponctuation, word) or re.match(r'[^\D]', word) or re.match(r'[\n]', word) or (word in stop_words):
        return False
    else:
        return True
    
def is_valid_semantic(tagged_words, n):
    grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
    cp = RegexpParser(grammar)

    parsed_word_tag = cp.parse(tagged_words)
   
    for child in parsed_word_tag:
            if isinstance(child, Tree):
                if child.label() == 'KT':
                    size_child = len(child)
                    if size_child == n:
                        
                        return True
                else:
                    return False

    
def main():
     docs = get_dataset("train",t="lemma")
     corpus = docs.values()
     corpus = [word_tokenize(doc) for doc in corpus]
     bm25 = BM25(corpus)
     bm25.get_scores()
     keyphrases = bm25.calc_prediction(0)
     print("keyphrases", keyphrases)