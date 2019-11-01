# -*- coding: utf-8 -*-
"""
Exercise 1
"""
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.datasets import fetch_20newsgroups
from scipy import sparse

def main():
    docs,test = get_20_news_group(18000)
    keys = tf_idf(docs,test)
    
    return keys

#loads train set and test doc
#@input:size disered for the docs set
#@return: set of docs set and one test doc 
def get_20_news_group(size):
    docs = fetch_20newsgroups(subset = 'all') 
    doc_test  = fetch_20newsgroups(subset = 'test', shuffle=False)
    
    return docs.data[:size + 1], [doc_test.data[0]]

#Tokenizes set train and test doc by sentence
#calls: tf_idf_train, tf_idf_test and calc_prediction
#@input:train set and doc test
#@return: set of keyphrases 
def tf_idf(docs, test):
    #TRAIN
    vectorizer_tfidf = tf_idf_train(docs)   
    
    #TEST
    test_vector = tf_idf_test(vectorizer_tfidf, test)
    keys = calc_prediction(test_vector, vectorizer_tfidf)
    
    return keys

#Creates vectorizer and fits it to the docs
#@input:train set, parameter (optional) removes the n most frequent words 
#@return: vectorizer 
def tf_idf_train(docs, maxdf = 1, mindf = 1):      
    vectorizer_tfidf = TfidfVectorizer(use_idf = True, 
                                           analyzer = 'word', 
                                           ngram_range=(1,3), 
                                           stop_words = 'english',
                                           token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]*[a-zA-Z]\b", 
                                           lowercase = True,
                                           max_df = maxdf,
                                           min_df = mindf,
                                           norm = 'l1')
        
    vectorizer_tfidf.fit_transform(docs)
                 
    return vectorizer_tfidf
  
#applies the learned vocabulary to the test doc
#@input:vectorizer and test tokanized
#@return: vectorizer 
def tf_idf_test(vectorizer_tfidf, doc_test):
    
    return vectorizer_tfidf.transform(doc_test)

#computes tf-idf final scores according to it len 
#@input: matrix with tf-idf scores, feature_names, char_or_word
#@return: matrix with tf-idf final scores 
def tf_idf_scores(test_vector, feature_names,chars_or_words="words", scale_factor=1):

    test_vector = test_vector.toarray()
   
    for i in range(0, test_vector.shape[0]):
        for j in range(0, test_vector.shape[1]):
            if test_vector[i,j] != 0:
                if chars_or_words == 'chars':
                    test_vector[i,j] = test_vector[i,j] * len(feature_names[j])*scale_factor   
                 
                    
                elif chars_or_words == 'words':
                    test_vector[i,j] =  test_vector[i,j] * len(feature_names[j].split())*scale_factor
    
    test_vector = sparse.csr_matrix(test_vector)
    
    return test_vector

#sortes matrix with tf-idf final scores
#@input: all the nonzero entries of matrix with tf-idf final scores
#@return: sorted nonzero entries of matrix with tf-idf final scores
def sort_terms(test_vector):
    tuples = zip(test_vector.col, test_vector.data)
    
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

#extracts keyphrases 
#@input: feature_names and sorted_terms
#@return: dictionary with top 5 terms and corresponding scores
def extract_keyphrases(feature_names ,sorted_terms):
    sorted_terms = sorted_terms[0:5]
  
    keyphrases = list()
    results= dict()
    # word index and corresponding tf-idf score
    for idx, score in sorted_terms:
        results[feature_names[idx]] = score
        keyphrases.append(feature_names[idx])

    return keyphrases

#computes predictions of top 5 keyphrases
#@input: matrix with tf-idf final scores and vectorizer
#@return: top 5 keyphrases
def calc_prediction(test_vector, vectorizer_tfidf, scale_factor=1):
    feature_names = vectorizer_tfidf.get_feature_names()
    test_vector = tf_idf_scores(test_vector.tocoo(), feature_names,chars_or_words="words")
    
    #SORT
    sorted_terms = sort_terms(test_vector.tocoo())
    keyphrases = extract_keyphrases(feature_names ,sorted_terms)
    
    return keyphrases