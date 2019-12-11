#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module contains function of computing rank scores for documents in
corpus and helper class `BM25` used in calculations. Original algorithm
descibed in [1]_, also you may check Wikipedia page [2]_.
.. [1] Robertson, Stephen; Zaragoza, Hugo (2009).  The Probabilistic Relevance Framework: BM25 and Beyond,
       http://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf
.. [2] Okapi BM25 on Wikipedia, https://en.wikipedia.org/wiki/Okapi_BM25
Examples
--------
.. sourcecode:: pycon
    >>> from gensim.summarization.bm25 import get_bm25_weights
    >>> corpus = [
    ...     ["black", "cat", "white", "cat"],
    ...     ["cat", "outer", "space"],
    ...     ["wag", "dog"]
    ... ]
    >>> result = get_bm25_weights(corpus, n_jobs=-1)
Data:
-----
.. data:: PARAM_K1 - Free smoothing parameter for BM25.
.. data:: PARAM_B - Free smoothing parameter for BM25.
.. data:: EPSILON - Constant used for negative idf of document in corpus.
"""


import math
from six import iteritems
from six.moves import range
from scipy.sparse import lil_matrix
import xml.dom.minidom
import os
import string
from nltk.corpus import stopwords
import re
from nltk.util import ngrams
import itertools
from nltk import Tree, RegexpParser, pos_tag
exercise2 = __import__('exercise-2')

PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25


class BM25(object):
    """Implementation of Best Matching 25 ranking function.
    Attributes
    ----------
    corpus_size : int
        Size of corpus (number of documents).
    avgdl : float
        Average length of document in `corpus`.
    doc_freqs : list of dicts of int
        Dictionary with terms frequencies for each document in `corpus`. Words used as keys and frequencies as values.
    idf : dict
        Dictionary with inversed documents frequencies for whole `corpus`. Words used as keys and frequencies as values.
    doc_len : list of int
        List of document lengths.
    """

    def __init__(self, corpus):
        """
        Parameters
        ----------
        corpus : list of list of str
            Given corpus.
        """
        self.corpus_size = 0
        self.avgdl = 0
        self.no_terms = 0
        self.terms= []
        self.doc_freqs = []
        self.idf = []
        self.doc_len = []
        self._initialize(corpus)

    def _initialize(self, corpus):
        """Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""
        nd = {}  # word -> number of documents with word
        num_doc = 0
        
        for document in corpus:
            self.corpus_size += 1
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in self.terms:
                        self.terms.append(word)
                        self.no_terms +=1
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in nd:
                    nd[word] = 0
                nd[word] += 1

        self.avgdl = float(num_doc) / self.corpus_size
        # collect idf sum to calculate an average idf for epsilon value
        
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        
        for document in corpus:
            inv_doc_freq = {}
         
            for word in document:
                for word, freq in iteritems(nd):
                    inv_doc_freq[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
                    if(inv_doc_freq[word] < 0):
                        inv_doc_freq[word] = EPSILON
            self.idf.append(inv_doc_freq)
                
        
        self.tfidf_matrix = lil_matrix((self.corpus_size, self.no_terms), dtype=float)

    def get_score(self, doc_index, index_term):
        score = 0
 
        word = self.terms[index_term]
        if word in self.idf[doc_index].keys():
            if word in self.doc_freqs[doc_index].keys():           
                term_freqs = self.doc_freqs[doc_index][word]
                score = (self.idf[doc_index][word]) * (term_freqs * (PARAM_K1 + 1)
                        / (term_freqs + PARAM_K1 * (1- PARAM_B + PARAM_B * self.doc_len[doc_index]/ self.avgdl)))
        return score
    
    def get_scores(self):
        for doc_index in range(0, self.corpus_size):
            for term_index in range(0, self.no_terms):
                tfidf = self.get_score(doc_index, term_index)  #* len(" ".split(self.terms[term_index]))
                self.tfidf_matrix[doc_index, term_index] = tfidf
        #print("tfidf_matrix", self.tfidf_matrix)
        return self.tfidf_matrix

    def get_top_5(self, test_vector):
        tuples = zip(test_vector.col, test_vector.data)
        #print("test_vector col", test_vector.col)
        tuples_terms = sorted(tuples, key=lambda x: x[1], reverse=True)
        sorted_terms = tuples_terms[0:10]
        #print("sorted_terms", sorted_terms)
        keyphrases = []
        for idx, score in sorted_terms:
            keyphrases.append(self.terms[idx])
        return keyphrases
        
        
    def calc_prediction(self, doc_index):
        test_vector = self.tfidf_matrix[doc_index,:]
        keyphrases = self.get_top_5(test_vector.tocoo())
        return keyphrases

def is_valid(word):
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
                
def get_dataset(folder, t="word"):
    path = os.path.dirname(os.path.realpath('__file__')) + "\\Inspec\\" + folder
 
    files = []
    docs = dict()
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f[:30]:
            if '.xml' in file:
                files.append(os.path.join(r, file))
           
    for f in files:
        base_name=os.path.basename(f)
        key = os.path.splitext(base_name)[0]
        doc = xml.dom.minidom.parse(f)

        # get a list of XML tags from the document and print each one
        sentences = doc.getElementsByTagName("sentence")
        sentence_string = []
        for sentence in sentences:
                tokens = sentence.getElementsByTagName("token")
             
                for token in tokens:
                    word = token.getElementsByTagName(t)[0].firstChild.data
                    word = word.lower()
                    if(is_valid(word)):
                        sentence_string.append(word)
                #ended iterating tokens from sentence
                
        #ended iterating sentences
        res = []
       
        unigram_it= ngrams(sentence_string, 1)
        bigram_it  = ngrams(sentence_string, 2)
        trigram_it = ngrams(sentence_string, 3)
        
        for tokens in itertools.chain(bigram_it, trigram_it, unigram_it):
            tagged_word = pos_tag(tokens)
            #print("tagged_word", tagged_word)
            if(is_valid_semantic(tagged_word, len(tokens))):
                res.append(" ".join(tokens))
          
                
        docs[key] = res
    
    return docs

 
def main():
     doc_count = 0
     all_ap = list()
     all_p5 = list()
     docs = get_dataset("test",t="lemma")
     corpus = docs.values()
     print("BM25>>>")
     bm25 = BM25(corpus)
     print("UPDATE>>>")
     bm25.get_scores()
     doc_index = 0
     true_labels = exercise2.json_references(stem_or_not_stem = "not stem")
     for doc_name in docs.keys():
         if doc_name in true_labels.keys():
             if len(true_labels[doc_name]) >= 5:
                 y_true = true_labels[doc_name]
                 doc_count += 1
                 y_pred = bm25.calc_prediction(doc_index)
                 exercise2.metrics(y_true, y_pred)
                 ap, p5 = exercise2.average_precision_score(y_true, y_pred)
                 all_ap.append(ap)
                 all_p5.append(p5)
         doc_index +=1
     mAP = exercise2.global_metrics(all_ap, doc_count, global_metric = 'mAP')
     print(">>>Mean average precision ", mAP)
    
     mP5 = exercise2.global_metrics(all_p5, doc_count, global_metric = 'mP5')
     print(">>>Mean precision@5 ", mP5)

