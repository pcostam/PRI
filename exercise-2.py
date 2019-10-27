# -*- coding: utf-8 -*-
"""
Exercise 2
"""
import xml.dom.minidom
import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from scipy import sparse
exercise1 = __import__('exercise-1')
"""
Process XML file. Preprocesses text
Arguments:
token (optional): must be "word" or "lemma"
Returns:
a list where each element corresponds to a list(document) containing strings (sentences) 
"""
def get_dataset(folder,t="word", test_size=5):
    path = os.path.dirname(os.path.realpath('__file__')) + "\\SemEval-2010\\" + folder
 
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.xml' in file:
                files.append(os.path.join(r, file))
                
    docs = dict()
    test_set = dict()
    i = 0
  
    for f in files:
        i += 1
        text = str()
        base_name=os.path.basename(f)
        key = os.path.splitext(base_name)[0]
        doc = xml.dom.minidom.parse(f)

        # get a list of XML tags from the document and print each one
        sentences = doc.getElementsByTagName("sentence")
      
        for sentence in sentences:
                tokens = sentence.getElementsByTagName("token")
                sentence_string = ""
                for token in tokens:
                    word = token.getElementsByTagName(t)[0].firstChild.data
                    sentence_string = sentence_string + " " + word
                #ended iterating tokens from sentence
                text += sentence_string
        #ended iterating sentences
        #text =  exercise1.preprocess(text)
        #add dictionary. key is name of file.
        docs[key] = text
        
        test_set[key] = [text]

    return docs.values(), test_set
       
"""
Returns dictionary with only n-grams where n < 4. Key is filename and value is list containing lists 
with keyphrases. 
"""
def json_references():
    path = os.path.dirname(os.path.realpath('__file__')) + "\\SemEval-2010\\references"
    filename = os.path.join(path,"train.combined.json")
    data = dict()
    
    with open(filename) as f:    
        data = json.load(f)
        
        for key, value in data.items():
            i = 0
            aux_list = []
            for gram in value:
                size = len(gram[0].split(" "))
                if(size==1 or size == 2 or size == 3):
                    aux_list.append(gram[0])
                    i += 1
                #if i == 5:
                    #break
            value = aux_list
            data[key] = value
            
    return data

#computes predictions of top 5 keyphrases
#@input: matrix with tf-idf final scores and vectorizer
#@return: top 5 keyphrases
def calc_prediction(test_vector, vectorizer_tfidf, num_known_relevants,scale_factor=1):
    feature_names = vectorizer_tfidf.get_feature_names()
    test_vector = exercise1.tf_idf_scores(test_vector.tocoo(), feature_names,chars_or_words="words")
    
    #SORT
    sorted_terms = sort_terms(test_vector.tocoo())
    keyphrases = extract_keyphrases(feature_names ,sorted_terms, num_known_relevants)
    
    return keyphrases


#sortes matrix with tf-idf final scores
#@input: all the nonzero entries of matrix with tf-idf final scores
#@return: sorted nonzero entries of matrix with tf-idf final scores
def sort_terms(test_vector):
    tuples = zip(test_vector.col, test_vector.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_keyphrases(feature_names ,sorted_terms, num_known_relevants):
    keyphrases = list()
    results= dict()
    
    sorted_terms = sorted_terms[0:num_known_relevants]
  
    # word index and corresponding tf-idf score
    for idx, score in sorted_terms:
        results[feature_names[idx]] = score
        keyphrases.append(feature_names[idx])
    
    return keyphrases

def average_precision_score(y_true, y_pred, no_relevant_collection):
    nr_relevants = 0
    i = 0
    Ap_at_sum = 0
    for el in y_pred:
        i += 1
        #is relevant
        if el in y_true:
            nr_relevants += 1
            Ap_at_sum += nr_relevants/i
        #if not relevant, count as 0

    return Ap_at_sum/no_relevant_collection

#Arguments:
# ap: numpy array of all average precision scores
#nr queries: corresponds to the number of documents 
def mean_average_precision_score(ap, nr_queries):
    sum_all_ap = np.sum(ap)
    return sum_all_ap/nr_queries

#Arguments:
#Grams predicted and Real grams
#Prints precision, recall and f1 score per individual document
def metrics(y_true, y_pred):
    
    relevant_retrieved = len(list(set(y_pred) & set(y_true)))
    
    #Precision = num grams relevants retrieved / total of grams retrieved
    precision = relevant_retrieved / len(y_pred)
    print(">>> precision score", precision)
    
    #Recall = num grams relevants retrieved / num relevant grams
    recall = relevant_retrieved / len(y_true)
    print(">>> recall score"   , recall)
    
    #F1-measure = 2 * (Precision * Recall )/(Precision + Recall)
    if(precision + recall != 0):
        f1_scores = 2 * (precision * recall) / (precision + recall)
        print(">>> f1 score"       , round(f1_scores, 2))
    else:
        f1_scores = 0
        
    return precision, recall, f1_scores
    
def main():
    precisions = list()
    all_ap = list()
    
    docs, test_set = get_dataset("train",t="lemma", test_size=5)
    true_labels = json_references()
    
    precisions = np.array(precisions)
    
    vectorizer_tfidf = exercise1.tf_idf_train(docs)
    
    #for key, doc in test_set.items():
    #    print(">>>>Doc to be tested", key)
    #    testvec = exercise1.tf_idf_test(vectorizer_tfidf, doc)
        
    #    y_true = true_labels[key]
    #    y_pred = calc_prediction(testvec,vectorizer_tfidf, len(y_true))
    #    print(">>>Predicted", y_pred)
    #    print(">>>Known Relevant", y_true)
        
    #    metrics(y_true, y_pred)
        
    #   ap = average_precision_score(y_true, y_pred, len(y_true))
    #    all_ap.append(ap)
    
    key = "C-76"
    doc = test_set[key]
    
    print(">>>>doc to be tested", key)
    testvec = exercise1.tf_idf_test(vectorizer_tfidf, doc)
    
    y_true = true_labels[key]
    y_pred = calc_prediction(testvec,vectorizer_tfidf, len(y_true))
    
    print(">>>y_pred", y_pred)
    print(">>>y_true", y_true)
    
    metrics(y_true, y_pred)

    #GLOBAL        
    #mean value for the precision@5 evaluation metric
    #precisions = np.array(precisions)
    #mean_precision_5 = np.mean(precisions)
    #print(">>> mean precision@5", mean_precision_5)
	
    #mean average precision
    #all_ap = np.array(all_ap)
    #mAP = mean_average_precision_score(all_ap,len(test_set.keys()))
    #print(">>> mean average precision", mAP)