# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 18:13:54 2018

@author: Diksha
"""


import os
import math
import numpy as np
import random
import sys

def get_data(spam_dir):
    data_spam=[]
    for file in os.listdir(spam_dir):
        f = spam_dir + '/'+ file
        
        with open(f, encoding="Latin1") as txt_file:
            data = txt_file.read().split()
            for word in data:
                data_spam.append(word)

    return data_spam


def naive_bayes(train_spam_dir,train_ham_dir, test_spam_dir, test_ham_dir):
    spam_files =  len([name for name in os.listdir(train_spam_dir) if os.path.isfile(os.path.join(train_spam_dir, name))])
    ham_files = len([name for name in os.listdir(train_ham_dir) if os.path.isfile(os.path.join(train_ham_dir, name))])
    
    test_spam_files =  len([name for name in os.listdir(test_spam_dir) if os.path.isfile(os.path.join(test_spam_dir, name))])
    test_ham_files = len([name for name in os.listdir(test_ham_dir) if os.path.isfile(os.path.join(test_ham_dir, name))])
    
    prob_spam_files = spam_files / (spam_files+ham_files)
    prob_ham_files = 1-prob_spam_files
    
    data_spam=get_data(train_spam_dir)
    data_ham = get_data(train_ham_dir)
    
    spam_ham_distinct = len(list(set(data_ham + data_spam)))
    
    #main algo
    correct_classification = 0
    
    for file in os.listdir(test_spam_dir):
        f = test_spam_dir + '/'+ file
      
        prob_spam_word = math.log10(prob_spam_files)
        prob_ham_word = math.log10(prob_ham_files)
        
        with open(f, encoding="Latin1") as txt_file:
            data = txt_file.read().split()
            for word in data:
                prob_word_spam = (data_spam.count(word) + 1)/ (len(data_spam)+spam_ham_distinct)
                prob_spam_word = prob_spam_word + math.log10(prob_word_spam)
                
                prob_word_ham = (data_ham.count(word) + 1)/ (len(data_ham)+spam_ham_distinct)
                prob_ham_word = prob_ham_word + math.log10(prob_word_ham)
        
        if prob_spam_word > prob_ham_word:
            correct_classification = correct_classification + 1
            
    
    for file in os.listdir(test_ham_dir):
        f = test_ham_dir + '/'+ file
        
        prob_spam_word = math.log10(prob_spam_files)
        prob_ham_word = math.log10(prob_ham_files)
        
        with open(f, encoding="Latin1") as txt_file:
            data = txt_file.read().split()
            for word in data:
                prob_word_spam = (data_spam.count(word) + 1)/ (len(data_spam)+spam_ham_distinct)
                prob_spam_word = prob_spam_word + math.log10(prob_word_spam)
                
                prob_word_ham = (data_ham.count(word) + 1)/ (len(data_ham)+spam_ham_distinct)
                prob_ham_word = prob_ham_word + math.log10(prob_word_ham)
        
        if prob_spam_word < prob_ham_word:
            correct_classification = correct_classification + 1
    
    accuracy = correct_classification/(test_spam_files+test_ham_files)
    print("\n#### accuracy for naive bayes with stopwords #####")
    print(accuracy)
    print();
            
def create_inverted_index(spam_ham_distinct, file):
    data = []
    with open(file, encoding="Latin1") as txt_file:
            data1 = txt_file.read().split()
            for word in data1:
                data.append(word)
    
    inverted_index = []
    #appending the x0 for w0
    inverted_index.append(1)
    for word in spam_ham_distinct:
        count = data.count(word)
        inverted_index.append(count)
    
    return inverted_index

def logistic_regression_train(train_spam_dir, train_ham_dir, l, iterate):
    data_spam=get_data(train_spam_dir)
    data_ham = get_data(train_ham_dir)
    spam_ham_distinct = list(set(data_ham + data_spam))
    
    # a dictionary like { <word> : <freq_
    #inverted_index_dict = create_inverted_index(spam_ham_distinct, )
    #initialize random weights with w0 also
    w = []
    for i in range(len(spam_ham_distinct)+1):
        val = random.randint(-10,10)
        w.append(val/10000)
    
    num_iteration = iterate
    Lambda = l
    eta = 0.1
    #training for the weights with L2 regularization
    for iteration in range(num_iteration):
        #starting the loop for spam train files
        true = 0
        error = np.array(np.zeros(len(spam_ham_distinct)+1))
        
        for file in os.listdir(train_spam_dir):
            f = train_spam_dir + '/'+ file
          
            
            spam_train_word_freq = create_inverted_index(spam_ham_distinct, f )
            #ham = 1 and spam = 0  #ham = exp(wix1) / 1+ exp(wixi)
            decision = np.dot(np.array(spam_train_word_freq), np.array(w))
            if decision > 0:
                predicted = 1
                
            else:
                predicted = 0
            
            error = error + (true - predicted)*np.array(spam_train_word_freq)
         
        true = 1
        for file in os.listdir(train_ham_dir):
            f = train_ham_dir + '/'+ file
            
            
            ham_train_word_freq = create_inverted_index(spam_ham_distinct, f )
            
            #ham = 1 and spam = 0
            #ham = exp(wix1) / 1+ exp(wixi)
            
            decision = np.dot(np.array(ham_train_word_freq), np.array(w))
            if decision > 0:
                predicted = 1
                
            else:
                predicted = 0
                
            error = error + (true - predicted)*np.array(ham_train_word_freq)
    
        #since batch gradient descent
        # update weights after going through the whole training data once
        w = np.array(w) + eta*np.array(error) - eta*Lambda*np.array(w)
    
    return w,spam_ham_distinct

def logistic_regression_test(test_spam_dir,test_ham_dir, trained_weight, spam_ham_distinct):
    test_spam_files =  len([name for name in os.listdir(test_spam_dir) if os.path.isfile(os.path.join(test_spam_dir, name))])
    test_ham_files = len([name for name in os.listdir(test_ham_dir) if os.path.isfile(os.path.join(test_ham_dir, name))])
    total_data = test_spam_files + test_ham_files
    w = trained_weight
    #starting the loop for spam train files
    true = 0
    correct_classify = 0
    
    for file in os.listdir(test_spam_dir):
        f = test_spam_dir + '/'+ file
       
        
        spam_test_word_freq = create_inverted_index(spam_ham_distinct, f )
        #ham = 1 and spam = 0  #ham = exp(wix1) / 1+ exp(wixi)
        decision = np.dot(np.array(spam_test_word_freq), np.array(w))
        if decision > 0:
            predicted = 1
            
        else:
            predicted = 0  
            correct_classify = correct_classify + 1
     
    true = 1
    for file in os.listdir(test_ham_dir):
        f = test_ham_dir + '/'+ file
        
        
        ham_test_word_freq = create_inverted_index(spam_ham_distinct, f )
        
        #ham = 1 and spam = 0
        #ham = exp(wix1) / 1+ exp(wixi)
   
        decision = np.dot(np.array(ham_test_word_freq), np.array(w))
        if decision > 0:
            predicted = 1
            correct_classify = correct_classify + 1
            
        else:
            predicted = 0
            
    
    accuracy = correct_classify/total_data
    print("########## Logistic accuracy with stopwords #####")
    print(accuracy)
    print();
    
    

########### with removing the stop words

def remove_stopwords(stopwords, data_list):
    for word in stopwords:
        data_list = list(filter((word).__ne__, data_list))
        
    return data_list
        
    
def naive_bayes_stopwords(train_spam_dir,train_ham_dir, test_spam_dir, test_ham_dir, stopwords):
    spam_files =  len([name for name in os.listdir(train_spam_dir) if os.path.isfile(os.path.join(train_spam_dir, name))])
    ham_files = len([name for name in os.listdir(train_ham_dir) if os.path.isfile(os.path.join(train_ham_dir, name))])
    
    test_spam_files =  len([name for name in os.listdir(test_spam_dir) if os.path.isfile(os.path.join(test_spam_dir, name))])
    test_ham_files = len([name for name in os.listdir(test_ham_dir) if os.path.isfile(os.path.join(test_ham_dir, name))])
    
    prob_spam_files = spam_files / (spam_files+ham_files)
    prob_ham_files = 1-prob_spam_files
    
    data_spam = get_data(train_spam_dir)
    data_ham = get_data(train_ham_dir)
    
    ## removing stopwords
    data_spam = remove_stopwords(stopwords, data_spam)
    data_ham = remove_stopwords(stopwords, data_ham)
    
    spam_ham_distinct = len(list(set(data_ham + data_spam)))
    
    #main algo
    correct_classification = 0
    
    for file in os.listdir(test_spam_dir):
        f = test_spam_dir + '/'+ file
       
        prob_spam_word = math.log10(prob_spam_files)
        prob_ham_word = math.log10(prob_ham_files)
        
        with open(f, encoding="Latin1") as txt_file:
            data = txt_file.read().split()
            data =  remove_stopwords(stopwords, data)
            for word in data:
                prob_word_spam = (data_spam.count(word) + 1)/ (len(data_spam)+spam_ham_distinct)
                prob_spam_word = prob_spam_word + math.log10(prob_word_spam)
                
                prob_word_ham = (data_ham.count(word) + 1)/ (len(data_ham)+spam_ham_distinct)
                prob_ham_word = prob_ham_word + math.log10(prob_word_ham)
        
        if prob_spam_word > prob_ham_word:
            correct_classification = correct_classification + 1
            
    
    for file in os.listdir(test_ham_dir):
        f = test_ham_dir + '/'+ file
       
        prob_spam_word = math.log10(prob_spam_files)
        prob_ham_word = math.log10(prob_ham_files)
        
        with open(f, encoding="Latin1") as txt_file:
            data = txt_file.read().split()
            data =  remove_stopwords(stopwords, data)
            for word in data:
                prob_word_spam = (data_spam.count(word) + 1)/ (len(data_spam)+spam_ham_distinct)
                prob_spam_word = prob_spam_word + math.log10(prob_word_spam)
                
                prob_word_ham = (data_ham.count(word) + 1)/ (len(data_ham)+spam_ham_distinct)
                prob_ham_word = prob_ham_word + math.log10(prob_word_ham)
        
        if prob_spam_word < prob_ham_word:
            correct_classification = correct_classification + 1
    
    accuracy = correct_classification/(test_spam_files+test_ham_files)
    print("\n#### accuracy for naive bayes without stopwords #####")
    print(accuracy)
    print("\n")
            

def logistic_regression_train_stopwords(train_spam_dir, train_ham_dir, l, stopwords, iterate):
    data_spam=get_data(train_spam_dir)
    data_ham = get_data(train_ham_dir)
    spam_ham_distinct = list(set(data_ham + data_spam)- set(stopwords))
    
    # a dictionary like { <word> : <freq_
    #inverted_index_dict = create_inverted_index(spam_ham_distinct, )
    #initialize random weights with w0 also
    w = []
    for i in range(len(spam_ham_distinct)+1):
        val = random.randint(-10,10)
        w.append(val/10000)
    
    num_iteration = iterate
    Lambda = l
    eta = 0.1
    #training for the weights with L2 regularization
    for iteration in range(num_iteration):
        #starting the loop for spam train files
        true = 0
        error = np.array(np.zeros(len(spam_ham_distinct)+1))
        
        for file in os.listdir(train_spam_dir):
            f = train_spam_dir + '/'+ file
            
            print(f)
            spam_train_word_freq = create_inverted_index(spam_ham_distinct, f )
            #ham = 1 and spam = 0  #ham = exp(wix1) / 1+ exp(wixi)
            decision = np.dot(np.array(spam_train_word_freq), np.array(w))
            if decision > 0:
                predicted = 1
                
            else:
                predicted = 0
            
            error = error + (true - predicted)*np.array(spam_train_word_freq)
         
        true = 1
        for file in os.listdir(train_ham_dir):
            f = train_ham_dir + '/'+ file
          
            
            ham_train_word_freq = create_inverted_index(spam_ham_distinct, f )
            
            #ham = 1 and spam = 0
            #ham = exp(wix1) / 1+ exp(wixi)
            
            decision = np.dot(np.array(ham_train_word_freq), np.array(w))
            if decision > 0:
                predicted = 1
                
            else:
                predicted = 0
                
            error = error + (true - predicted)*np.array(ham_train_word_freq)
    
        #since batch gradient descent
        # update weights after going through the whole training data once
        w = np.array(w) + eta*np.array(error) - eta*Lambda*np.array(w)
    
    return w,spam_ham_distinct

def logistic_regression_test_stopwords(test_spam_dir,test_ham_dir, trained_weight, spam_ham_distinct, stopwords):
    test_spam_files =  len([name for name in os.listdir(test_spam_dir) if os.path.isfile(os.path.join(test_spam_dir, name))])
    test_ham_files = len([name for name in os.listdir(test_ham_dir) if os.path.isfile(os.path.join(test_ham_dir, name))])
    total_data = test_spam_files + test_ham_files
    w = trained_weight
    #starting the loop for spam train files
    true = 0
    correct_classify = 0
    
    for file in os.listdir(test_spam_dir):
        f = test_spam_dir + '/'+ file
        
        
        spam_test_word_freq = create_inverted_index(spam_ham_distinct, f )
        #ham = 1 and spam = 0  #ham = exp(wix1) / 1+ exp(wixi)
        decision = np.dot(np.array(spam_test_word_freq), np.array(w))
        if decision > 0:
            predicted = 1
            
        else:
            predicted = 0  
            correct_classify = correct_classify + 1
     
    true = 1
    for file in os.listdir(test_ham_dir):
        f = test_ham_dir + '/'+ file
       
        
        ham_test_word_freq = create_inverted_index(spam_ham_distinct, f )
        
        #ham = 1 and spam = 0
        #ham = exp(wix1) / 1+ exp(wixi)
   
        decision = np.dot(np.array(ham_test_word_freq), np.array(w))
        if decision > 0:
            predicted = 1
            correct_classify = correct_classify + 1
            
        else:
            predicted = 0
            
    
    accuracy = correct_classify/total_data
    print("\n#######")
    print("\nLogistic accuracy without stopwords")
    print(accuracy)
    print("\n");
    
    
    

    
# main function
'''
train_spam_dir = 'hw2_train/train/spam'
train_ham_dir = 'hw2_train/train/ham'

test_spam_dir = 'hw2_test/test/spam'
test_ham_dir = 'hw2_test/test/ham'
'''

train_spam_dir = sys.argv[1]
train_ham_dir = sys.argv[2]

test_spam_dir = sys.argv[3]
test_ham_dir = sys.argv[4]

Lambda = float(sys.argv[5])
iterate = int(sys.argv[6])

######## removing the stopwords
stopwords = open("stopwords.txt",'r')
stopwords = stopwords.read().split('\n')

##### naive bayes ######
naive_bayes(train_spam_dir, train_ham_dir,test_spam_dir,test_ham_dir)
naive_bayes_stopwords(train_spam_dir, train_ham_dir,test_spam_dir,test_ham_dir,stopwords)

####### logistic with stopwords present #######
trained_weight, spam_ham_distinct = logistic_regression_train(train_spam_dir, train_ham_dir, Lambda, iterate)
logistic_regression_test(test_spam_dir,test_ham_dir, trained_weight, spam_ham_distinct)

####### Logistic with stopwords removal #####
trained_weight, spam_ham_distinct = logistic_regression_train_stopwords(train_spam_dir, train_ham_dir, Lambda, stopwords, iterate)
logistic_regression_test_stopwords(test_spam_dir,test_ham_dir, trained_weight, spam_ham_distinct, stopwords)
