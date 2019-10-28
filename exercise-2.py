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

def average_precision_score(y_true, y_pred):
    nr_relevants = 0
    i = 0
    ap_at_sum = 0
    
    for el in y_pred:
        i += 1
        #is relevant
        if el in y_true:
            nr_relevants += 1
            ap_at_sum += nr_relevants/i
        
        if i == 5 :
            precision_at_5 = nr_relevants/i
    
    #if not relevant, count as 0
    if nr_relevants == 0:
        return 0, precision_at_5
    else:
        return ap_at_sum/nr_relevants, precision_at_5

#Arguments:
# ap: numpy array of all average precision scores
#nr queries: corresponds to the number of documents 
def global_metrics(ap_or_p5_lst, nr_queries, global_metric):
    
    if global_metric == 'mAP':
        sum_all_ap = np.sum(ap_or_p5_lst)
        return sum_all_ap/nr_queries
    
    elif global_metric == 'mP5':
        sum_all_p5 = np.sum(ap_or_p5_lst)
        return sum_all_p5/nr_queries
    
    else:
        return ">>>>WRONG GLOBAL METRIC<<<<"
#Arguments:
#Grams predicted and Real grams
#Prints precision, recall and f1 score per individual document
def metrics(y_true, y_pred):
    
    y_pred_set = set(y_pred)
    y_true_set = set(y_true)
    
    y_intersection = len(y_pred_set.intersection(y_true_set))
    
    #Precision = num grams relevants retrieved / total of grams retrieved
    precision = y_intersection / len(y_pred)
    #print(">>> precision score", precision)
    
    #Recall = num grams relevants retrieved / num relevant grams
    recall = y_intersection / len(y_true)
    #print(">>> recall score"   , recall)
    
    #F1-measure = 2 * (Precision * Recall )/(Precision + Recall)
    if precision + recall != 0:
        f1_scores = 2 * (precision * recall) / (precision + recall)
        #print(">>> f1 score"   , round(f1_scores, 2))
    else:
        print(">>>>WRONG GLOBAL METRIC<<<<")
    
def main():
    precisions = list()
    all_ap = list()
    all_p5 = list()
    
    docs, test_set = get_dataset("train",t="lemma", test_size=5)
    true_labels = json_references()
    
    precisions = np.array(precisions)
    
    #for i in range(2, 100):
        
        #print("INDEX : ", i )
        
    vectorizer_tfidf = exercise1.tf_idf_train(docs, 17)
    
    for key, doc in test_set.items():
        #print(">>>>Testing document ", key)
        testvec = exercise1.tf_idf_test(vectorizer_tfidf, doc)
        
        y_true = true_labels[key]
        y_pred = exercise1.calc_prediction(testvec,vectorizer_tfidf)
        #print(">>>Predicted ", y_pred)
        #print(">>>Known Relevant ", y_true)
         
        metrics(y_true, y_pred)

        ap, p5 = average_precision_score(y_true, y_pred)
        all_ap.append(ap)
        all_p5.append(p5)
        
    mAP = global_metrics(all_ap, len(test_set.keys()), global_metric = 'mAP')
    print(">>>Mean average precision ", mAP)
    
    mP5 = global_metrics(all_p5, len(test_set.keys()), global_metric = 'mP5')
    print(">>>Mean average precision@5 ", mP5)
        
        #all_ap.clear()
        #all_p5.clear()
        #vectorizer_tfidf.vocabulary_.clear()
    
    #key = "C-76"
    #doc = test_set[key]
        
    #print(">>>>doc to be tested", key)
    #testvec = exercise1.tf_idf_test(vectorizer_tfidf, doc)
        
     #y_true = true_labels[key]
     #y_pred = exercise1.calc_prediction(testvec,vectorizer_tfidf)
        
     #print(">>>y_pred", y_pred)
     #print(">>>y_true", y_true)
        
     #metrics(y_true, y_pred)