# -*- coding: utf-8 -*-
"""
Exercise 2
"""
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import xml.dom.minidom
import numpy as np
import json
import os
exercise1 = __import__('exercise-1')

def main():
    precisions = list()
    all_ap = list()
    all_p5 = list()
    precision_curve_plot = list()
    recall_curve_plot = list()
    doc_count = 0
    
    docs, test_set = get_dataset("test", t="word", stem_or_not_stem = "not stem")
    true_labels = json_references(stem_or_not_stem = "not stem")
    precisions = np.array(precisions)
    
    #for i in range(1, 100, 1):   
        #print("MIN INDEX : ", i )
    #for j in range (3, 20, 1):
     #   print("MAX INDEX : ", j )
        #MAX INDEX :  9
        #>>>Mean average precision  0.16658163265306122
        #>>>Mean precision@5  0.05994897959183673
    
    vectorizer_tfidf = exercise1.tf_idf_train(docs,9, 1)
    
    for key, doc in test_set.items():
        #print(">>>>Testing document ", key)
        testvec = exercise1.tf_idf_test(vectorizer_tfidf, doc)
        
        y_true = true_labels[key]
        
        y_pred = exercise1.calc_prediction(testvec, vectorizer_tfidf)

        #print(">>>Predicted ", y_pred)
        #print(">>>Known Relevant ", y_true)
        
        if len(y_true) >= 5:
            precision, recall = metrics(y_true, y_pred)
            precision_curve_plot.append(precision)
            recall_curve_plot.append(recall)
            
            ap, p5 = average_precision_score(y_true, y_pred)
            all_ap.append(ap)
            all_p5.append(p5)
            doc_count += 1
            
    mAP = global_metrics(all_ap, doc_count, global_metric = 'mAP')
    print(">>>Mean average precision ", mAP)
    
    mP5 = global_metrics(all_p5, doc_count, global_metric = 'mP5')
    print(">>>Mean precision@5 ", mP5)

    #plot_precision_recall(precision_curve_plot, recall_curve_plot)
    
       # all_ap.clear()
        #all_p5.clear()

#Process XML file. Preprocesses text
#Input:path to file, token (optional)"word" or "lemma", token(optional)"stem" or "not stem"
#Output: list where each element corresponds to a document (string) and list of lists where 
#intern lists has the test documents (strings) as elements
#Notes: can't be "lemma" and "stem" at the same time
def get_dataset(folder, t="word", stem_or_not_stem = "not stem"):
    path = os.path.dirname(os.path.realpath('__file__')) + "\\Inspec\\" + folder
    ps = PorterStemmer()
    test_set = dict()
    files = list()
    docs = dict()
    file_counter = 0
    i = 0
    
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.xml' in file:
                files.append(os.path.join(r, file))
                
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
                    if (t == 'word' and stem_or_not_stem == 'stem'):
                        word = ps.stem(word)
                        
                    sentence_string = sentence_string + " " + word
          
                text += sentence_string
        
        #add dictionary. key is name of file.
        if(file_counter <= 375):
            docs[key] = text
        else:
            test_set[key] = [text]

        file_counter += 1
        
    return docs.values(), test_set
       

#From "\SemEval-2010\references" extracts the real keyphrases of the documents
#Input: token (optional) "stem" or "not stem"
#Output: dictionary with n-grams where n < 4.
#Notes: Key is filename; value is a list containing lists of keyphrases. 
def json_references(stem_or_not_stem = 'not stem'):
    data = dict()    
    
    path = os.path.dirname(os.path.realpath('__file__')) + "\\Inspec\\references"
    if stem_or_not_stem == 'not stem':
        filename = os.path.join(path,"test.uncontr.json")
    elif stem_or_not_stem == 'stem':
        filename = os.path.join(path,"test.uncontr.stem.json")
        
    with open(filename) as f:    
        docs = json.load(f)
        
        for key, value in docs.items():
            aux_list = []
            for gram in value:
                size = len(gram[0].split(" "))
                if(size==1 or size == 2 or size == 3):
                    aux_list.append(gram[0])
            value = aux_list
            data[key] = value

    return data

#Input: Real grams of the document, Predicted grams of the document
#Output: , Interpolated precision@5 of a document 
def average_precision_score(y_true, y_pred): 
    nr_relevants = 0
    i = 0
    ap_at_sum = 0
    precision_at_5 = 0
    
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

#Input: Numpy array of all average precision scores, Number of documents,
#Token selects between calculation of "mAP" or "mP5"
#Output: Mean Average Precision or Mean Precision@5
def global_metrics(ap_or_p5_lst, nr_queries, global_metric):
    
    if global_metric == 'mAP':
        sum_all_ap = np.sum(ap_or_p5_lst)
        return sum_all_ap/nr_queries
    
    elif global_metric == 'mP5':
        sum_all_p5 = np.sum(ap_or_p5_lst)
        return sum_all_p5/nr_queries

#Prints precision, recall and f1 score per individual document    
#Input: Real grams of the document, Predicted grams of the document
#Output:Precision and Recall measures
def metrics(y_true, y_pred):
    
    y_pred_set = set(y_pred)
    y_true_set = set(y_true)
    
    y_intersection = len(y_pred_set.intersection(y_true_set))
    
    #Precision = num grams relevants retrieved / total of grams retrieved
    precision = y_intersection / len(y_pred)
    print(">>> precision score", precision)
    
    #Recall = num grams relevants retrieved / num relevant grams
    recall = y_intersection / len(y_true)
    print(">>> recall score"   , recall)
    
    #F1-measure = 2 * (Precision * Recall )/(Precision + Recall)
    if precision + recall != 0:
        f1_scores = 2 * (precision * recall) / (precision + recall)
        print(">>> f1 score"   , f1_scores)
        
    return precision, recall
         
def plot_precision_recall(precision_curve_plot, recall_curve_plot):
    precision_curve_plot.sort()
    recall_curve_plot.sort(reverse = True)
    i = len(recall_curve_plot)-2
    precision_copy=precision_curve_plot.copy()

    while(i >= 0):
        if precision_curve_plot[i+1] > precision_curve_plot[i]:
            precision_curve_plot[i] = precision_curve_plot[i+1]
        i=i-1
    
    #fig, ax = plt.subplots()
    for i in range(len(recall_curve_plot)-1):
        plt.plot((recall_curve_plot[i],recall_curve_plot[i]),(precision_curve_plot[i],precision_curve_plot[i+1]),'k-',label='',color='red') #vertical
        plt.plot((recall_curve_plot[i],recall_curve_plot[i+1]),(precision_curve_plot[i+1],precision_curve_plot[i+1]),'k-',label='',color='red') #horizontal
    
    plt.plot(recall_curve_plot, precision_copy,'k--',color='blue')
    plt.xlabel("recall")
    plt.ylabel("precision")