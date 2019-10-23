# -*- coding: utf-8 -*-
"""
Exercise 2
"""
import xml.dom.minidom
import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer 
from scipy import sparse

"""
Process XML file.
Arguments:
token (optional): must be "word" or "lemma"
Returns:
a list where each element corresponds to a list(document) containing strings (sentences) 
"""
def get_dataset(folder,t="word", test_size=0.25):
    path = os.path.dirname(os.path.realpath('__file__')) + "\\SemEval-2010\\" + folder
 
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.xml' in file:
                files.append(os.path.join(r, file))
                
    train_set = list()
    i = 0
    files = files[:30]
    size_dataset = len(files)
    index_test = size_dataset*test_size
    test_set = dict()
    
    for f in files[:30]:
        base_name=os.path.basename(f)
        key = os.path.splitext(base_name)[0]
        doc = xml.dom.minidom.parse(f)

        # get a list of XML tags from the document and print each one
        sentences = doc.getElementsByTagName("sentence")
        i +=1
        for sentence in sentences:
                tokens = sentence.getElementsByTagName("token")
                sentence_string = ""
                for token in tokens:
                    word = token.getElementsByTagName(t)[0].firstChild.data  
                    sentence_string = sentence_string + " " + word
                #ended iterating tokens from sentence
                #GENERATE CLAUSES
                clauses_list = sentence_string.split(",")
                for clause in clauses_list:
                    if i <= (size_dataset-index_test):    
                        train_set.append(clause)
                
                    elif(i > size_dataset - index_test):
                        if key in test_set:
                            test_set[key].append(clause)
                        else:
                            test_set[key] = [clause]
                    
      
    return test_set, train_set
       
    
    
    
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
                if i == 5:
                    break
            value = aux_list
            data[key] = value
            
    return data

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
"""
returns vectorizer
"""
def tf_idf_train(candidates_train, th_min=1, th_max=1):
     #Learn the vocabulary dictionary and return term-document matrix
    vectorizer_tfidf = TfidfVectorizer(use_idf = True, analyzer = 'word', 
                                       ngram_range=(1, 3), stop_words = 'english', max_df=th_max, min_df=th_min) #Removing very rare words (3) and Removing very frequent words (90%)
    
    #for txt in candidates_train:
    vectorizer_tfidf.fit_transform(candidates_train)
        
    
    return vectorizer_tfidf
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

def tf_idf_test(vectorizer_tfidf, candidates_tokanize_test):
    test_vector = vectorizer_tfidf.transform(candidates_tokanize_test)
    
    #print(vectorizer_tfidf.vocabulary_.keys())#DA TERMOS!!!!!!
#    #test_vector  = vectorizer_tfidf.transform(candidates_tokanize_test)
#    print(testvec)
    return test_vector

def sort_terms(test_vector):
    tuples = zip(test_vector.col, test_vector.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def calc_prediction(test_vector, vectorizer_tfidf):
    feature_names = vectorizer_tfidf.get_feature_names()
    test_vector = tf_idf_scores(test_vector.tocoo(), feature_names,chars_or_words="words")
    
    #SORT
    sorted_terms = sort_terms(test_vector.tocoo())
    keyphrases = extract_keyphrases(feature_names ,sorted_terms)
    
    return keyphrases.keys()
def tf_idf_scores(test_vector, feature_names,chars_or_words="words"):
    #print("type", type(testvec))
    #print("before >>>>testvec", testvec.toarray())

    test_vector = test_vector.toarray()
    
    for i in range(0, test_vector.shape[0]):
        for j in range(0, test_vector.shape[1]):
            if test_vector[i,j] != 0:
                if chars_or_words == 'chars':
                    test_vector[i,j] = test_vector[i,j] * len(feature_names[j])     
                    
                elif chars_or_words == 'words':
                    test_vector[i,j] =  test_vector[i,j] * len(feature_names[j].split())
    
    #print("after >>>>testvec", testvec)
    test_vector = sparse.csr_matrix(test_vector)
    return test_vector

def extract_keyphrases(feature_names ,sorted_terms):
    sorted_terms = sorted_terms[:5]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_terms:
        
        #keep track of feature name and its corresponding score
        score_vals.append(score)
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

def main():
    test_set, train_set = get_dataset("train",t="lemma", test_size=0.25)
    true_labels = json_references()
    
    for sentence in train_set:
        sentence = sentence_preprocess(sentence)
    
        
    precisions = list()
    precisions = np.array(precisions)
    all_ap = list()
   
    vectorizer_tfidf = tf_idf_train(train_set)
    for key, doc in test_set.items():
            doc = sentence_preprocess(doc)
            y_pred = list()
            print(">>>>doc to be tested", key)
            testvec = tf_idf_test(vectorizer_tfidf, doc)
            keys_pred = calc_prediction(testvec,vectorizer_tfidf )
            for key_pred in keys_pred:
                y_pred.append(key_pred)
        
            
            print(">>>y_pred", y_pred)
            y_true = true_labels[key]
            print(">>>y_true", y_true)
            #print precision, recall per individual document
            precision_5 = precision_score(y_true, y_pred, average='micro')
            print(">>> precision score", precision_5)
            print(">>> recall score", recall_score(y_true, y_pred, average='micro'))
            print(">>> f1 score", f1_score(y_true, y_pred, average='micro'))
            print(">>> average precision score", )
            #Q: relevant documents in collection, should consider more than
            #y_true? Just 5?
            ap = average_precision_score(y_true, y_pred, len(y_true))
            all_ap.append(ap)
  
    #GLOBAL        
    #mean value for the precision@5 evaluation metric
    precisions = np.array(precisions)
    mean_precision_5 = np.mean(precisions)
    print(">>> mean precision@5", mean_precision_5)
	
    #mean average precision
    all_ap = np.array(all_ap)
    mAP = mean_average_precision_score(all_ap,len(test_set.keys()))
    print(">>> mean average precision", mAP)
   
 
   