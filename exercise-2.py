# -*- coding: utf-8 -*-
"""
Exercise 2
"""
import xml.dom.minidom
import os
import json
preprocess = __import__('exercise-1')

"""
Process XML file.
Arguments:
token (optional): must be "word" or "lemma"
Returns:
a list where each element corresponds to a list(document) containing strings (sentences) 
"""
def get_dataset(t="word"):
    path = os.path.dirname(os.path.realpath('__file__')) + "\\SemEval-2010\\train"
 
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.xml' in file:
                files.append(os.path.join(r, file))
                
    sentences_doc = []
    for f in files[:5]:
        print(f)
        doc = xml.dom.minidom.parse(f);

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
        sentences_doc.append(sentences_list)
       
      
    print("doc", len(sentences_doc))
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
            print(key, value)
    return data

def main():
    sentences_doc = get_dataset()
    print(">>>>>", sentences_doc[0])
    for doc in sentences_doc:
            preprocess.sentence_preprocess(doc)