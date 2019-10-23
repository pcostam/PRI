# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.datasets import fetch_20newsgroups
from scipy import sparse
import string
import nltk 
import re

#load stopwords to set
stop_words = set(nltk.corpus.stopwords.words("english"))

def main():
    train, test = get_20_news_group(30)
    keys = tf_idf(train, test)
    print(keys)

#loads train set and test doc
#@input:size disered for the train set
#@return: set of train set and one test doc 
def get_20_news_group(size_train):
    train = fetch_20newsgroups(subset = 'train', shuffle = True) #The F-score will be lower because it is more realistic.
    test  = fetch_20newsgroups(subset = 'test') #The F-score will be lower because it is more realistic.
    
    return train.data[:30], [test.data[0]]

#Tokanizes set train and test doc by sentence
#calls: sentence_preprocess, tf_idf_train, tf_idf_test and calc_prediction
#@input:train set and doc test
#@return: set of keyphrases 
def tf_idf(train, test):
    candidates_train = list()
    candidates_tokanize_train = list()
    candidates_tokanize_test = list()
    result = list()
    
    #TRAIN
    for doc in train:
        phrases = nltk.sent_tokenize(doc)
        candidates_tokanize_train = sentence_preprocess(phrases)
        candidates_train = candidates_train + candidates_tokanize_train
            
    vectorizer_tfidf = tf_idf_train(candidates_train)   
    
    #TEST
    phrases = nltk.sent_tokenize(test[0])
    candidates_tokanize_test = sentence_preprocess(phrases)
    test_vector = tf_idf_test(vectorizer_tfidf, candidates_tokanize_test)
    
    keys = calc_prediction(test_vector, vectorizer_tfidf)
    
    for key in keys:
        result.append(key)

    return result
    
#cleaning each sentence
#@input:phrase
#@return: phrases in lowercase with no pontuation and no digits 
def sentence_preprocess(phrases):
    candidates =[]
    for phrase in phrases:  
        phrase = phrase.lower()
        #Remove puntuation, except hifen
        remove = string.punctuation
        remove = remove.replace("-", "") # don't remove hyphens
        pattern = r"[{}]".format(remove) # create the pattern
        phrase = re.sub(pattern, "", phrase)
        
        phrase = re.sub(r'[^\D]'  ,'',phrase)
        phrase = re.sub(r'[\n]',' ', phrase)
        candidates.append(phrase)
        
    return candidates


#Creates vectorizer and fits it to the train set
#@input:train set already pre-processed 
#@return: vectorizer 
def tf_idf_train(candidates_train, vocabulary=[]):
     #Learn the vocabulary dictionary and return term-document matrix
    if(vocabulary == []):
        vectorizer_tfidf = TfidfVectorizer(use_idf = True, analyzer = 'word', ngram_range=(1, 3), stop_words = 'english')
        vectorizer_tfidf.fit_transform(candidates_train)
    else:
        vectorizer_tfidf = TfidfVectorizer(vocabulary=vocabulary, use_idf = True, analyzer = 'word', ngram_range=(1, 3), stop_words = 'english')
        vectorizer_tfidf.fit_transform(candidates_train)
        
    return vectorizer_tfidf

#applies the learned vocabulary to the test doc
#@input:vectorizer and test tokanized
#@return: vectorizer 
def tf_idf_test(vectorizer_tfidf, candidates_tokanize_test):
    return vectorizer_tfidf.transform(candidates_tokanize_test)

#computes tf-idf final scores according to it len 
#@input: matrix with tf-idf scores, feature_names, char_or_word
#@return: matrix with tf-idf final scores 
def tf_idf_scores(test_vector, feature_names,chars_or_words="words"):

    test_vector = test_vector.toarray()
    #count = 0
    for i in range(0, test_vector.shape[0]):
        for j in range(0, test_vector.shape[1]):
            if test_vector[i,j] != 0:
                #count += 1
                if chars_or_words == 'chars':
                    test_vector[i,j] = test_vector[i,j] * len(feature_names[j])     
                    
                elif chars_or_words == 'words':
                    test_vector[i,j] =  test_vector[i,j] * len(feature_names[j].split())
    
    #print("count", count)
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
def calc_prediction(test_vector, vectorizer_tfidf):
    feature_names = vectorizer_tfidf.get_feature_names()
    test_vector = tf_idf_scores(test_vector.tocoo(), feature_names,chars_or_words="words")
    
    #SORT
    sorted_terms = sort_terms(test_vector.tocoo())
    keyphrases = extract_keyphrases(feature_names ,sorted_terms)
    
    return keyphrases
