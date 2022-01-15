#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from numpy import asarray
import numpy as np
from sklearn.model_selection import KFold
file1 = open("C:\\Users\\Public\\SARS2_protein_pair_label.txt", 'r')
file2 = open("C:\\Users\\Public\\SARS2_protein_sequences.txt", 'r')
file3 =open("C:\\Users\\Public\\SARS2_protein_cluster.txt", 'r')
Lines = file1.readlines()
Lines2 = file2.readlines()
Lines3 = file3.readlines()
Dict1=dict()
l=list()
for line in Lines3:
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
num_folds=10
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
        r=0
        n=0
        for k in range(len(x_list3)):
            for j in range(len(l)):
                if((x_list1[test][i] in l[j]) and (x_list3[k] in l[j])):
                    r=r+1
                if((x_list2[test][i] in l[j]) and (x_list3[k] in l[j])):
                    n=n+1
        if(r>=1 and n>=1):                                        
            if(y_list[test][i]==1):
                c1_seq1.append(x_list1[test][i])
                c1_seq2.append(x_list2[test][i])
                c1_y.append(1)
            else:
                c1_seq1_neg.append(x_list1[test][i])
                c1_seq2_neg.append(x_list2[test][i])  
#            c1_y.append(y_list[test][i])
        if(r==0 and n==0):
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
    r=0
    sp_seq1_train=[]
    sp_seq2_train=[]
    v=len(train_seq1)
    for a in range(v):
        seq1=train_seq1[a]
        seq2=train_seq2[a]
        for b in range(len(train_seq1[r+1:v])):
            seq21=train_seq1[r+1:v][b]
            seq22=train_seq2[r+1:v][b]
#                    r1=0
#                    r2=0
#                    n1=0
#                    n2=0
            for j in range(len(l)):
#                        if((seq1 in l[j]) and (seq21 in l[j])):   
#                            r1=r1+1
#                        if((w2 in l[j]) and (seq22 in l[j])):
#                            r2=r2+1
#                        if((seq1 in l[j]) and (seq22 in l[j])):   
#                            n1=n1+1
#                        if((seq2 in l[j]) and (seq21 in l[j])):
#                            n2=n2+1
#                        if((r1>=1 and r2>=1) or (n1>=1 and n2>=1)):
                if(((seq1 in l[j]) and (seq21 in l[j])) or ((seq2 in l[j]) and (seq22 in l[j])) or ((seq1 in l[j]) and (seq22 in l[j])) or ((seq2 in l[j]) and (seq21 in l[j]))):
                    if((seq1 not in sp_seq1_train) and (seq2 not in sp_seq2_train)):
                        sp_seq1_train.append(seq1)
                        sp_seq2_train.append(seq2)
                        print("redundant pair in train_set")
                        print(seq1)
                        print(seq2)
            if((seq1 in sp_seq1_train) and (seq2 in sp_seq2_train)):
                break
            r=r+1

    for seq in sp_seq1_train:
        train_seq1.remove(seq)
        
    for seq in sp_seq2_train:
        train_seq2.remove(seq)
    #####

    r=0
    sp_seq1_train_neg=[]
    sp_seq2_train_neg=[]
    v=len(train_seq1_neg)
    for a in range(v):
        seq1=train_seq1_neg[a]
        seq2=train_seq2_neg[a]
        for b in range(len(train_seq1_neg[r+1:v])):
            seq21=train_seq1_neg[r+1:v][b]
            seq22=train_seq2_neg[r+1:v][b]     
#                    r1=0
#                    r2=0
#                    n1=0
#                    n2=0
            for j in range(len(l)):
#                        if((seq1 in l[j]) and (seq21 in l[j])):   
#                            r1=r1+1
#                        if((w2 in l[j]) and (seq22 in l[j])):
#                            r2=r2+1
#                        if((seq1 in l[j]) and (seq22 in l[j])):   
#                            n1=n1+1
#                        if((seq2 in l[j]) and (seq21 in l[j])):
#                            n2=n2+1
#                        if((r1>=1 and r2>=1) or (n1>=1 and n2>=1)):
                if(((seq1 in l[j]) and (seq21 in l[j])) or ((seq2 in l[j]) and (seq22 in l[j])) or ((seq1 in l[j]) and (seq22 in l[j])) or ((seq2 in l[j]) and (seq21 in l[j]))):
                    if((seq1 not in sp_seq1_train_neg) and (seq2 not in sp_seq2_train_neg)):
                        sp_seq1_train_neg.append(seq1)
                        sp_seq2_train_neg.append(seq2)
                        print("redundant pair in train_set_neg")
                        print(seq1)
                        print(seq2)
            if((seq1 in sp_seq1_train_neg) and (seq2 in sp_seq2_train_neg)):
                break
        r=r+1

    for seq in sp_seq1_train_neg:
        train_seq1_neg.remove(seq)
        
    for seq in sp_seq2_train_neg:
        train_seq2_neg.remove(seq)        
    ######
    r=0
    sp_seq1_c1=[]
    sp_seq2_c1=[]
    v=len(c1_seq1)
    for a in range(len(c1_seq1)):
        seq1=c1_seq1[a]
        seq2=c1_seq2[a]
        for b in range(len(c1_seq1[r+1:v])):
            seq21=c1_seq1[r+1:v][b]
            seq22=c1_seq2[r+1:v][b]
#                    r1=0
#                    r2=0
#                    n1=0
#                    n2=0
            for j in range(len(l)):
#                        if((seq1 in l[j]) and (seq21 in l[j])):   
#                            r1=r1+1
#                        if((w2 in l[j]) and (seq22 in l[j])):
#                            r2=r2+1
#                        if((seq1 in l[j]) and (seq22 in l[j])):   
#                            n1=n1+1
#                        if((seq2 in l[j]) and (seq21 in l[j])):
#                            n2=n2+1
#                        if((r1>=1 and r2>=1) or (n1>=1 and n2>=1)):
                if(((seq1 in l[j]) and (seq21 in l[j])) or ((seq2 in l[j]) and (seq22 in l[j])) or ((seq1 in l[j]) and (seq22 in l[j])) or ((seq2 in l[j]) and (seq21 in l[j]))):
                    if((seq1 not in sp_seq1_c1) and (seq2 not in sp_seq2_c1)):
                        sp_seq1_c1.append(seq1)
                        sp_seq2_c1.append(seq2)
                        print("redundant pair in c1_set")
                        print(seq1)
                        print(seq2)
            if((seq1 in sp_seq1_c1) and (seq2 in sp_seq2_c1)):
                break
        r=r+1

    for seq in sp_seq1_c1:
        c1_seq1.remove(seq)
        
    for seq in sp_seq2_c1:
        c1_seq2.remove(seq)
        
        
##########
    r=0
    sp_seq1_c1_neg=[]
    sp_seq2_c1_neg=[]
    v=len(c1_seq1_neg)
    for a in range(v):
        seq1=c1_seq1_neg[a]
        seq2=c1_seq2_neg[a]
        for b in range(len(c1_seq1_neg[r+1:v])):
            seq21=c1_seq1_neg[r+1][b]
            seq22=c1_seq2_neg[r+1][b]
#                    r1=0
#                    r2=0
#                    n1=0
#                    n2=0
            for j in range(len(l)):
#                        if((seq1 in l[j]) and (seq21 in l[j])):   
#                            r1=r1+1
#                        if((w2 in l[j]) and (seq22 in l[j])):
#                            r2=r2+1
#                        if((seq1 in l[j]) and (seq22 in l[j])):   
#                            n1=n1+1
#                        if((seq2 in l[j]) and (seq21 in l[j])):
#                            n2=n2+1
#                        if((r1>=1 and r2>=1) or (n1>=1 and n2>=1)):
                if(((seq1 in l[j]) and (seq21 in l[j])) or ((seq2 in l[j]) and (seq22 in l[j])) or ((seq1 in l[j]) and (seq22 in l[j])) or ((seq2 in l[j]) and (seq21 in l[j]))):
                    if((seq1 not in sp_seq1_c1_neg) and (seq2 not in sp_seq2_c1_neg)):
                        sp_seq1_c1_neg.append(seq1)
                        sp_seq2_c1_neg.append(seq2)
                        print("redundant pair in c1_set_neg")
                        print(seq1)
                        print(seq2)
            if((seq1 in sp_seq1_c1_neg) and (seq2 in sp_seq2_c1_neg)):
                break
        r=r+1

    for seq in sp_seq1_c1_neg:
        c1_seq1.remove(seq)
        
    for seq in sp_seq2_c1_neg:
        c1_seq2.remove(seq) 
        
        
 ##########
    r=0
    sp_seq1_c2=[]
    sp_seq2_c2=[]
    v=len(c2_seq1)
    for a in range(v):
        seq1=c2_seq1[a]
        seq2=c2_seq2[a]
        for b in range(len(c2_seq1[r+1:v])):
            seq21=c2_seq1[r+1:v][b]
            seq22=c2_seq2[r+1:v][b]
#                    r1=0
#                    r2=0
#                    n1=0
#                    n2=0
            for j in range(len(l)):
#                        if((seq1 in l[j]) and (seq21 in l[j])):   
#                            r1=r1+1
#                        if((w2 in l[j]) and (seq22 in l[j])):
#                            r2=r2+1
#                        if((seq1 in l[j]) and (seq22 in l[j])):   
#                            n1=n1+1
#                        if((seq2 in l[j]) and (seq21 in l[j])):
#                            n2=n2+1
#                        if((r1>=1 and r2>=1) or (n1>=1 and n2>=1)):
                if(((seq1 in l[j]) and (seq21 in l[j])) or ((seq2 in l[j]) and (seq22 in l[j])) or ((seq1 in l[j]) and (seq22 in l[j])) or ((seq2 in l[j]) and (seq21 in l[j]))):
                    if((seq1 not in sp_seq1_c2) and (seq2 not in sp_seq2_c2)):
                        sp_seq1_c2.append(seq1)
                        sp_seq2_c2.append(seq2)
                        print("redundant pair in c2_seq")
                        print(seq1)
                        print(seq2)
            if((seq1 in sp_seq1_c2) and (seq2 in sp_seq2_c2)):
                break
        r=r+1

    for seq in sp_seq1_c2:
        c2_seq1.remove(seq)
        
    for seq in sp_seq2_c2:
        c2_seq2.remove(seq)

        
###############        
    r=0
    sp_seq1_c2_neg=[]
    sp_seq2_c2_neg=[]
    v=len(c2_seq1_neg)
    for a in range(v):
        seq1=c2_seq1_neg[a]
        seq2=c2_seq2_neg[a]
        for b in range(c2_seq1_neg[r+1:v]):
            seq21=c2_seq1_neg[r+1:v][b]
            seq22=c2_seq1_neg[r+1:v][b]
#                    r1=0
#                    r2=0
#                    n1=0
#                    n2=0
            for j in range(len(l)):
#                        if((seq1 in l[j]) and (seq21 in l[j])):   
#                            r1=r1+1
#                        if((w2 in l[j]) and (seq22 in l[j])):
#                            r2=r2+1
#                        if((seq1 in l[j]) and (seq22 in l[j])):   
#                            n1=n1+1
#                        if((seq2 in l[j]) and (seq21 in l[j])):
#                            n2=n2+1
#                        if((r1>=1 and r2>=1) or (n1>=1 and n2>=1)):
                if(((seq1 in l[j]) and (seq21 in l[j])) or ((seq2 in l[j]) and (seq22 in l[j])) or ((seq1 in l[j]) and (seq22 in l[j])) or ((seq2 in l[j]) and (seq21 in l[j]))):
                    if((seq1 not in sp_seq1_c2_neg) and (seq2 not in sp_seq2_c2_neg)):
                        sp_seq1_c2_neg.append(seq1)
                        sp_seq2_c2_neg.append(seq2)
                        print("redundant pair in c2_set_neg")
                        print(seq1)
                        print(seq2)
            if((seq1 in sp_seq1_c2_neg) and (seq2 in sp_seq2_c2_neg)):
                break
        r=r+1

    for seq in sp_seq1_c2_neg:
        c2_seq1.remove(seq)
        
    for seq in sp_seq2_c2_neg:
        c2_seq2.remove(seq) 

    
###############
    r=0
    sp_seq1_c3=[]
    sp_seq2_c3=[]
    v=len(c3_seq1)
    for a in range(v):
        seq1=c3_seq1[a]
        seq2=c3_seq2[a]           
#                    r1=0
#                    r2=0
#                    n1=0
#                    n2=0
        for b in range(len(c3_seq1[r+1:v])):
            seq21=c3_seq1[r+1:v][b]
            seq22=c3_seq2[r+1:v][b]
            
            for j in range(len(l)):
#                        if((seq1 in l[j]) and (seq21 in l[j])):   
#                            r1=r1+1
#                        if((w2 in l[j]) and (seq22 in l[j])):
#                            r2=r2+1
#                        if((seq1 in l[j]) and (seq22 in l[j])):   
#                            n1=n1+1
#                        if((seq2 in l[j]) and (seq21 in l[j])):
#                            n2=n2+1
#                        if((r1>=1 and r2>=1) or (n1>=1 and n2>=1)):
                if(((seq1 in l[j]) and (seq21 in l[j])) or ((seq2 in l[j]) and (seq22 in l[j])) or ((seq1 in l[j]) and (seq22 in l[j])) or ((seq2 in l[j]) and (seq21 in l[j]))):
                    if((seq1 not in sp_seq1_c3) and (seq2 not in sp_seq2_c3)):
                        sp_seq1_c3.append(seq1)
                        sp_seq2_c3.append(seq2)
                        print("redundant pair in c3_set")
                        print(seq1)
                        print(seq2)
            if((seq1 in sp_seq1_c3) and (seq2 in sp_seq2_c3)):
                break
        r=r+1

    for seq in sp_seq1_c3:
        c3_seq1.remove(seq)
        
    for seq in sp_seq2_c3:
        c3_seq2.remove(seq)
        
#############

    r=0
    sp_seq1_c3_neg=[]
    sp_seq2_c3_neg=[]
    v=len(c3_seq1_neg)
    for a in range(v):
        seq1=c3_seq1_neg[a]
        seq2=c3_seq2_neg[a]
        for b in range(len(c3_seq1_neg[r+1:v])):
            seq21=c3_seq1_neg[r+1:v][b]
            seq22=c3_seq2_neg[r+1:v][b]
#                    r1=0
#                    r2=0
#                    n1=0
#                    n2=0
            for j in range(len(l)):
#                        if((seq1 in l[j]) and (seq21 in l[j])):   
#                            r1=r1+1
#                        if((w2 in l[j]) and (seq22 in l[j])):
#                            r2=r2+1
#                        if((seq1 in l[j]) and (seq22 in l[j])):   
#                            n1=n1+1
#                        if((seq2 in l[j]) and (seq21 in l[j])):
#                            n2=n2+1
#                        if((r1>=1 and r2>=1) or (n1>=1 and n2>=1)):
                if(((seq1 in l[j]) and (seq21 in l[j])) or ((seq2 in l[j]) and (seq22 in l[j])) or ((seq1 in l[j]) and (seq22 in l[j])) or ((seq2 in l[j]) and (seq21 in l[j]))):
                    if((seq1 not in sp_seq1_c3_neg) and (seq2 not in sp_seq2_c3_neg)):
                        sp_seq1_c3_neg.append(seq1)
                        sp_seq2_c3_neg.append(seq2)
                        print("redundant pair in c3_set_neg")
                        print(seq1)
                        print(seq2)
            if((seq1 in sp_seq1_c3_neg) and (seq2 in sp_seq2_c3_neg)):
                break
        r=r+1

    for seq in sp_seq1_c3_neg:
        c3_seq1_neg.remove(seq)
        
    for seq in sp_seq2_c3_neg:
        c3_seq2_neg.remove(seq) 

#################

##########
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
    while (len(train_seq1_neg_test) < 10*len(train_seq1)):
        if(len(train_seq1_neg_index)>0):
            m= np.random.randint(0, len(train_seq1_neg_index))
        n=train_seq1_neg_index[m]
        train_seq1_neg_test.append(train_seq1_neg[n])
        train_seq2_neg_test.append(train_seq2_neg[n])
        train_y_neg.append(0)
        train_seq1_neg_index.remove(n)
    while (len(c1_seq1_neg_test) < 10*len(c1_seq1)):
        if(len(c1_seq1_neg_index)>0):
            m= np.random.randint(0, len(c1_seq1_neg_index))
        n=c1_seq1_neg_index[m]
        c1_seq1_neg_test.append(c1_seq1_neg[n])
        c1_seq2_neg_test.append(c1_seq2_neg[n])
        c1_y_neg.append(0)
        c1_seq1_neg_index.remove(n)
    while (len(c2_seq1_neg_test) < 10*len(c2_seq1)):
        if(len(c2_seq1_neg_index)>0):
            m= np.random.randint(0, len(c2_seq1_neg_index))
        n=c2_seq1_neg_index[m]
        c2_seq1_neg_test.append(c2_seq1_neg[n])
        c2_seq2_neg_test.append(c2_seq2_neg[n])
        c2_y_neg.append(0)
        c2_seq1_neg_index.remove(n)     
    while (len(c3_seq1_neg_test) < 10*len(c3_seq1)):
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
    if(fold_no==1):
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
data_f=pd.DataFrame.from_dict(Dict_properties).to_csv('C:\\Users\\Public\\SARS2_train2.csv',index=False) 

Dict_properties2={'test_seq1':test_set_seq1,'test_seq2':test_set_seq2, 'test_y':test_set_y}
data_f2=pd.DataFrame.from_dict(Dict_properties2).to_csv('C:\\Users\\Public\\SARS2_PPI_test2.csv',index=False)  


# In[ ]:




