#!/usr/bin/env python
# coding: utf-8

# In[62]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from numpy import asarray
import numpy as np
from sklearn.model_selection import KFold
file1 = open("C:\\Users\\Public\\HIV_protein_pair_label.txt", 'r')
file2 = open("C:\\Users\\Public\\HIV_protein_sequences.txt", 'r')
Lines = file1.readlines()
Lines2 = file2.readlines()
Dict1=dict()
for line in Lines2:
    x=line.split()
    ids=x[0]
#    print(ids)
    seq=x[1]
#    w1=w1[0:6]
#    w2=w2[0:6]
#    output=x[2]
#    output=int(output)
    Dict1[ids]=seq
        
    
x_train=[]
y_train=[]
x_test=[]
y_test=[]
x_list=[]
x_list1=[]
x_list2=[]
y_list=[]
for line in Lines:
    x=line.split()
    w1=x[0]
    w2=x[1]
    output=x[2]
    output=int(output)
    seq1=w1
    seq2=w2
#    seq1=Dict1[w1]
#    seq2=Dict1[w2]
#    w3=seq1+seq2   
#    x_list.append(str(w3))
    y_list.append(output)
    x_list1.append(seq1)
    x_list2.append(seq2)
    
#print(x_list2[0:10])

#trainDF = pandas.DataFrame()
#trainDF['text'] = x_list

x_list1=asarray(x_list1)
x_list2=asarray(x_list2)
y_list=asarray(y_list)
num_folds=9
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(x_list1, x_list2, y_list):
    x_list3=[]
    c1_seq1=[]
    c1_seq2=[]
    c1_seq1_neg=[]
    c1_seq2_neg=[]
    c1_seq1_neg_test=[]
    c1_seq2_neg_test=[]
    c1_y=[]
    c1_y_neg=[]
    c2_seq1=[]
    c2_seq2=[]
    c2_seq1_neg=[]
    c2_seq2_neg=[]
    c2_seq1_neg_test=[]
    c2_seq2_neg_test=[]
    c2_y=[]
    c2_y_neg=[]
    c3_seq1=[]
    c3_seq2=[]
    c3_seq1_neg=[]
    c3_seq2_neg=[]
    c3_seq1_neg_test=[]
    c3_seq2_neg_test=[]
    c3_y=[]
    c3_y_neg=[]   
    train_seq1=[]
    train_seq2=[]
    train_seq1_neg=[]
    train_seq2_neg=[]
    train_seq1_neg_test=[]
    train_seq2_neg_test=[]
    train_y=[]
    train_y_neg=[]
#    print(x_list1[train])
#    x_list3.append(list(x_list1[train]))
#    x_list3.append(list(x_list2[train]))
#    x_list3=x_list1[train]
#    x_list3.append(x_list2[train])
    x_list3=np.concatenate([x_list1[train], x_list2[train]])
    for j in range(len(x_list1[train])):
        if(y_list[train][j]==1):
            train_seq1.append(x_list1[train][j])
            train_seq2.append(x_list2[train][j])
            train_y.append(1)
        else:
            train_seq1_neg.append(x_list1[train][j])
            train_seq2_neg.append(x_list2[train][j])            
    for i in range(len(x_list1[test])):
        if((x_list1[test][i] in x_list3) and(x_list2[test][i] in x_list3)):
            if(y_list[test][i]==1):
                c1_seq1.append(x_list1[test][i])
                c1_seq2.append(x_list2[test][i])
                c1_y.append(1)
            else:
                c1_seq1_neg.append(x_list1[test][i])
                c1_seq2_neg.append(x_list2[test][i])  
#            c1_y.append(y_list[test][i])
        if((x_list1[test][i] not in x_list3) and (x_list2[test][i] not in x_list3)):
            if(y_list[test][i]==1):
                c3_seq1.append(x_list1[test][i])
                c3_seq2.append(x_list2[test][i])
                c3_y.append(1)
            else:
                c3_seq1_neg.append(x_list1[test][i])
                c3_seq2_neg.append(x_list2[test][i])
        else:
            if(y_list[test][i]==1):
                c2_seq1.append(x_list1[test][i])
                c2_seq2.append(x_list2[test][i])
                c2_y.append(1)
            else:
                c2_seq1_neg.append(x_list1[test][i])
                c2_seq2_neg.append(x_list2[test][i])
    print("train_seq1:",len(train_seq1)) 
    print("c1_seq1:",len(c1_seq1)) 
    print("c2_seq1:",len(c2_seq1)) 
    print("c3_seq1:",len(c3_seq1))
    train_seq1_neg_index=[]
    c1_seq1_neg_index=[]
    c2_seq1_neg_index=[]
    c3_seq1_neg_index=[]
    for i in range(len(train_seq1_neg)):
        train_seq1_neg_index.append(i)
    for i in range(len(c1_seq1_neg)):
        c1_seq1_neg_index.append(i)
    for i in range(len(c2_seq1_neg)):
        c2_seq1_neg_index.append(i)
    for i in range(len(c3_seq1_neg)):
        c3_seq1_neg_index.append(i)     
    while (len(train_seq1_neg_test) < 9*len(train_seq1)):
        if(len(train_seq1_neg_index)>0):
            m= np.random.randint(0, len(train_seq1_neg_index))
        n=train_seq1_neg_index[m]
        train_seq1_neg_test.append(train_seq1_neg[n])
        train_seq2_neg_test.append(train_seq2_neg[n])
        train_y_neg.append(0)
        train_seq1_neg_index.remove(n)
    while (len(c1_seq1_neg_test) < 9*len(c1_seq1)):
        if(len(c1_seq1_neg_index)>0):
            m= np.random.randint(0, len(c1_seq1_neg_index))
        n=c1_seq1_neg_index[m]
        c1_seq1_neg_test.append(c1_seq1_neg[n])
        c1_seq2_neg_test.append(c1_seq2_neg[n])
        c1_y_neg.append(0)
        c1_seq1_neg_index.remove(n)
    while (len(c2_seq1_neg_test) < 9*len(c2_seq1)):
        if(len(c2_seq1_neg_index)>0):
            m= np.random.randint(0, len(c2_seq1_neg_index))
        n=c2_seq1_neg_index[m]
        c2_seq1_neg_test.append(c2_seq1_neg[n])
        c2_seq2_neg_test.append(c2_seq2_neg[n])
        c2_y_neg.append(0)
        c2_seq1_neg_index.remove(n)     
    while (len(c3_seq1_neg_test) < 9*len(c3_seq1)):
        if(len(c3_seq1_neg_index)>0):
            m= np.random.randint(0, len(c3_seq1_neg_index))
        n=c3_seq1_neg_index[m]
        c3_seq1_neg_test.append(c3_seq1_neg[n])
        c3_seq2_neg_test.append(c3_seq2_neg[n])
        c3_y_neg.append(0)
        c3_seq1_neg_index.remove(n)
        #n= np.random.randint(0, len(c3_seq1_neg))
        #c3_seq1_neg_test.append(c3_seq1_neg[n])
        #c3_seq2_neg_test.append(c3_seq2_neg[n]) 
    print("train_seq1_neg:",len(train_seq1_neg))
    print("c1_seq1_neg:",len(c1_seq1_neg)) 
    print("c2_seq1_neg:",len(c2_seq1_neg)) 
    print("c3_seq1_neg:",len(c3_seq1_neg))
    print("train_seq1_neg-test:",len(train_seq1_neg_test)) 
    print("c1_seq1_neg-test:",len(c1_seq1_neg_test)) 
    print("c2_seq1_neg-test:",len(c2_seq1_neg_test)) 
    print("c3_seq1_neg-test:",len(c3_seq1_neg_test))
    fold_no=fold_no+1
    if(fold_no>1):
        break
train_set_seq1=np.concatenate([train_seq1, train_seq1_neg_test])
train_set_seq2=np.concatenate([train_seq2, train_seq2_neg_test])
c1_set_seq1=np.concatenate([c1_seq1, c1_seq1_neg_test])
c1_set_seq2=np.concatenate([c1_seq2, c1_seq2_neg_test])
c2_set_seq1=np.concatenate([c2_seq1, c2_seq1_neg_test])
c2_set_seq2=np.concatenate([c2_seq2, c2_seq2_neg_test])
c3_set_seq1=np.concatenate([c3_seq1, c3_seq1_neg_test])
c3_set_seq2=np.concatenate([c3_seq2, c3_seq2_neg_test])
train_set_y=np.concatenate([train_y, train_y_neg])
c1_set_y=np.concatenate([c1_y, c1_y_neg])
c2_set_y=np.concatenate([c2_y, c2_y_neg])
c3_set_y=np.concatenate([c3_y, c3_y_neg])
test_set_seq1=np.concatenate([c1_set_seq1,c2_set_seq1,c3_set_seq1])
test_set_seq2=np.concatenate([c1_set_seq2,c2_set_seq2,c3_set_seq2])
test_set_y=np.concatenate([c1_set_y,c2_set_y,c3_set_y])
Dict_properties={'train_seq1':train_set_seq1,'train_seq2':train_set_seq2, 'train_y':train_set_y}
data_f=pd.DataFrame.from_dict(Dict_properties).to_csv('C:\\Users\\Public\\HIV_PPI_train.csv',index=False) 

Dict_properties2={'test_seq1':test_set_seq1,'test_seq2':test_set_seq2, 'test_y':test_set_y}
data_f2=pd.DataFrame.from_dict(Dict_properties2).to_csv('C:\\Users\\Public\\HIV_PPI_test.csv',index=False)  

#print(c1_seq1[0]) 
#print(x_list3)

            
         
    
       
#trainDF = pandas.DataFrame()
#trainDF['text'] = x_list
#trainDF['label'] = y_list

#x_train, x_test, y_train, y_test=train_test_split(trainDF['text'],trainDF['label'],train_size=0.6)
#tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(3,3), max_features=1000)
#tfidf_vect_ngram_chars.fit_transform(trainDF['text'])
#xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(trainDF['text'])
#print(xtrain_tfidf_ngram_chars[0])


# In[ ]:




