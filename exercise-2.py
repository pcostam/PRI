# -*- coding: utf-8 -*-
"""
Exercise 2
"""
import xml.dom.minidom
import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
preprocess = __import__('exercise-1')

"""
Process XML file.
Arguments:
token (optional): must be "word" or "lemma"
Returns:
a list where each element corresponds to a list(document) containing strings (sentences) 
"""
def get_dataset(t="word"):
    sentences_doc_train = get_dataset_aux("train",t)
    
    train = dict()
    test = dict()
    size_train = len(sentences_doc_train)
    size_test = size_train/3
    
    i = 0
    #25% test set, 75% train set
    for key, value in sentences_doc_train.items():
            train[key] = value
            i += 1
            if i > (size_train-size_test):
                test[key] = value
        
        
       
    print("train>>>", len(train))
    print("test>>>", len(test))

  
    return train,test

def get_dataset_aux(folder,t="word"):
    path = os.path.dirname(os.path.realpath('__file__')) + "\\SemEval-2010\\" + folder
 
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.xml' in file:
                files.append(os.path.join(r, file))
                
    sentences_doc = dict()
    for f in files[:30]:
        base_name=os.path.basename(f)
        key = os.path.splitext(base_name)[0]
        doc = xml.dom.minidom.parse(f)

        # get a list of XML tags from the document and print each one
        sentences = doc.getElementsByTagName("sentence")
        sentences_list = list()
        for sentence in sentences:
                tokens = sentence.getElementsByTagName("token")
                sentence_string = ""
                for token in tokens:
                    word = token.getElementsByTagName(t)[0].firstChild.data
                   
                    sentence_string = sentence_string + " " + word
                
                sentences_list.append(sentence_string)
        sentences_doc[key] = sentences_list
        
    return sentences_doc
       
    
    
    
"""
Returns dictionary. Key is filename and value is list containing lists 
with keyphrases
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
    
def main():
    train, test = get_dataset(t="lemma")
    true_labels = json_references()
    train_set = dict()

    for key, doc in train.items():
            doc = preprocess.sentence_preprocess(doc)
            train_set[key] = doc
                        
        
    precisions = list()
    precisions = np.array(precisions)
    all_ap = list()
    vectorizer_tfidf = preprocess.tf_idf_train(train_set.values())
    for key, doc in test.items():
            doc = preprocess.sentence_preprocess(doc)
            print(">>>>doc to be tested", key)
            y_pred = preprocess.tf_idf_aux(vectorizer_tfidf, doc)
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
    mAP = mean_average_precision_score(all_ap,len(test.keys()))
    print(">>> mean average precision", mAP)
   
 
   