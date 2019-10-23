# -*- coding: utf-8 -*-
"""
Exercise 3
"""
import os
import xml.dom.minidom
from nltk import Tree, RegexpParser
exercise2 = __import__('exercise-2')
exercise1 = __import__('exercise-1')
#
#Returns: list of tuples where each corresponds to (word, tag). Each word
# is from train set
def get_tagged(t="word"):
    path = os.path.dirname(os.path.realpath('__file__')) + "\\SemEval-2010\\train"
 
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.xml' in file:
                files.append(os.path.join(r, file))
            
                

    files = files[:30]
    size_dataset = len(files)
  
    test_size=0.25
    index_test = size_dataset*test_size
    word_tag_pairs = list()
    index = int(size_dataset - index_test)
   
    for f in files[:index]:
        doc = xml.dom.minidom.parse(f)

        # get a list of XML tags from the document and print each one
        sentences = doc.getElementsByTagName("sentence")
    
        for sentence in sentences:
                tokens = sentence.getElementsByTagName("token")
                for token in tokens:
                    word = token.getElementsByTagName(t)[0].firstChild.data
                    tag = token.getElementsByTagName("POS")[0].firstChild.data 
                    word_tag_pair = (word,tag)
                    word_tag_pairs.append(word_tag_pair)
    #END iterating train set
    #Generate valid grams
    grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
    cp = RegexpParser(grammar)
    parsed_word_tag = cp.parse(word_tag_pairs)
    valid_grams = list()
    for child in parsed_word_tag:
        if isinstance(child, Tree):               
            if child.label() == 'KT':
                group_words = []
                for num in range(len(child)):
                        group_words.append(child[num][0])
                n_gram = " ".join(group_words)
                valid_grams.append(n_gram)
    print(">>>valid_grams", valid_grams)
    return valid_grams

def main():
     test_set, train_set = exercise2.get_dataset("train",t="lemma", test_size=0.25)
     vectorizer_tfidf = exercise1.tf_idf_train(train_set)
     grams = vectorizer_tfidf.get_feature_names()
     valid_grams = get_tagged(t="lemma")
     to_delete = list(set(grams) - set(valid_grams))
     #delete terms that do not match from vocabulary
     for term in to_delete:
         del vectorizer_tfidf.vocabulary_[term] 
     print(">>>after", vectorizer_tfidf.vocabulary_)
     print(">>>after", vectorizer_tfidf.get_feature_names())
     
     
     
     

     
                
                
                    