# -*- coding: utf-8 -*-
"""
Exercise 2
"""
import xml.dom.minidom
import os

"""
Process XML file.
Arguments:
token (optional): must be "word" or "lemma"
Returns:
a list where each element corresponds to a list(document) containing strings (sentences) 
"""
def get_dataset():
    path = os.path.dirname(os.path.realpath('__file__')) + "\\SemEval-2010\\train"
 
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.xml' in file:
                files.append(os.path.join(r, file))

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
                    word = token.getElementsByTagName("word")[0].firstChild.data
                    sentence_string = sentence_string + " " + word
                
                sentences_list.append(sentence_string)
        print(">>>>>>>>>>>>>>>>", sentences_list)
        
                
                    
       