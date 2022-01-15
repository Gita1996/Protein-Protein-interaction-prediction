#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from numpy import asarray
import numpy as np
from sklearn.model_selection import KFold
import pandas
import re
file1 =open("C:\\Users\\Public\\SARS2_protein_cluster.txt", 'r')
file2=open("C:\\Users\\Public\\SARS2_protein_pair_label.txt", 'r')
Lines = file1.readlines()
Lines2=file2.readlines()
def get_key(val):
    for key, value in di.items():
         if(val == value):
            return key
l=list()
di=dict()
di2=dict()

for line in Lines:
    x=line.split(",")
    l2=list()
    for i in range(1,len(x)):
        # l2.append(x[i][1:7])
        idx1 = x[i].index('>')
        idx2 = x[i].index('.')
#        print(idx2)
#        result = re.search('>(.*)...', x[i])
#        nu=result.group(1)
        idx1=idx1 +1
        l2.append(x[i][idx1:idx2])
    l.append(l2)
#print(l)

for i in range(len(l)):
    for j in range(len(l[i])):
#        print(l[i][j])
        di2[i]=l[i][j]
i=0 
ll=len(Lines2)
   

i=0
sp=list()
for line in Lines2:
    x=line.split()
    w1=x[0]
    w2=x[1]
    output=x[2]
    output=int(output)
    for line2 in Lines2[i+1:ll]:
        x2=line2.split()
        w21=x2[0]
        w22=x2[1]
        p=[line,line2]
        r1=0
        r2=0
        n1=0
        n2=0
        for j in range(len(l)):
            if((w1 in l[j]) and (w21 in l[j])):   
                r1=r1+1
            if((w2 in l[j]) and (w22 in l[j])):
                r2=r2+1
            if((w1 in l[j]) and (w22 in l[j])):   
                n1=n1+1
            if((w2 in l[j]) and (w21 in l[j])):
                n2=n2+1
        if((r1>=1 and r2>=1) or (n1>=1 and n2>=1)):
            if(line not in sp):
                sp.append(line) 
                print("redundant pair")
                print(x)
        if(line in sp):
            break
#            Lines2.remove(line2)
            
    i=i+1
        
#                for k in range(len(l)):
#                    if((w2 in l[k]) and (w22 in l[k])):
#                        sp.append(line)
                        
                        
#                        Lines2.remove(line)                        
#                        
for p in sp:
    Lines2.remove(p)
print(Lines2)        


# In[18]:


x_train=[]
y_train=[]
x_test=[]
y_test=[]
x_list=[]
x_list1=[]
x_list2=[]
y_list=[]

for line in Lines2:
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
    

x_list1=asarray(x_list1)
x_list2=asarray(x_list2)
y_list=asarray(y_list)
num_folds=5
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(x_list1, x_list2, y_list):
    x_list3=[]
    test_seq1=[]
    test_seq2=[]
    test_seq1_neg=[]
    test_seq2_neg=[]
    test_seq1_neg_test=[]
    test_seq2_neg_test=[]
    test_y=[]
    test_y_neg=[]   
    train_seq1=[]
    train_seq2=[]
    train_seq1_neg=[]
    train_seq2_neg=[]
    train_seq1_neg_selected=[]
    train_seq2_neg_selected=[]
    train_y=[]
    train_y_neg=[]
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
        if(y_list[test][i]==1):
            test_seq1.append(x_list1[test][i])
            test_seq2.append(x_list2[test][i])
            test_y.append(1)
        else:
            test_seq1_neg.append(x_list1[test][i])
            test_seq2_neg.append(x_list2[test][i])  
#            c1_y.append(y_list[test][i])
    print("train_seq1:",len(train_seq1)) 
    print("test_seq1:",len(test_seq1)) 
    print("train_seq1_neg:",len(train_seq1_neg))
    print("test_seq1_neg:",len(test_seq1_neg)) 
          
    train_seq1_neg_index=[]
    test_seq1_neg_index=[]
    
    for i in range(len(train_seq1_neg)):
        train_seq1_neg_index.append(i)
        
    for i in range(len(test_seq1_neg)):
        test_seq1_neg_index.append(i)
        
    while (len(train_seq1_neg_selected)< 10*len(train_seq1)):
        if(len(train_seq1_neg_index)>0):
            m= np.random.randint(0, len(train_seq1_neg_index))
        n=train_seq1_neg_index[m]
        train_seq1_neg_selected.append(train_seq1_neg[n])
        train_seq2_neg_selected.append(train_seq2_neg[n])
        train_y_neg.append(0)
        train_seq1_neg_index.remove(n)
    while (len(test_seq1_neg_test) < 10*len(test_seq1)):
        if(len(test_seq1_neg_index)>0):
            m= np.random.randint(0, len(test_seq1_neg_index))
        n=test_seq1_neg_index[m]
        test_seq1_neg_test.append(test_seq1_neg[n])
        test_seq2_neg_test.append(test_seq2_neg[n])
        test_y_neg.append(0)
        test_seq1_neg_index.remove(n)
    if(fold_no==1):
        break
        
train_set_seq1=np.concatenate([train_seq1, train_seq1_neg_selected])
train_set_seq2=np.concatenate([train_seq2, train_seq2_neg_selected])
test_set_seq1=np.concatenate([test_seq1, test_seq1_neg_test])
test_set_seq2=np.concatenate([test_seq2, test_seq2_neg_test])
train_set_y=np.concatenate([train_y, train_y_neg])
test_set_y=np.concatenate([test_y, test_y_neg])
Dict_properties={'train_seq1':train_set_seq1,'train_seq2':train_set_seq2, 'train_y':train_set_y}
data_f=pd.DataFrame.from_dict(Dict_properties).to_csv('C:\\Users\\Public\\SARS2_PPI_train.csv',index=False) 

Dict_properties2={'test_seq1':test_set_seq1,'test_seq2':test_set_seq2, 'test_y':test_set_y}
data_f2=pd.DataFrame.from_dict(Dict_properties2).to_csv('C:\\Users\\Public\\SARS2_PPI_test.csv',index=False)  


# In[ ]:




