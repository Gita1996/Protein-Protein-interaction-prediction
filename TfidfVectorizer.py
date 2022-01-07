#!/usr/bin/env python
# coding: utf-8

# In[49]:


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
#    seq1=w1
#    seq2=w2
    seq1=Dict1[w1]
    seq2=Dict1[w2]
#    w3=seq1+seq2   
#    x_list.append(str(w3))
    y_list.append(output)
    x_list1.append(str(seq1))
    x_list2.append(str(seq2))
    
#print(x_list2[0:10])

#trainDF = pandas.DataFrame()
#trainDF['text'] = x_list

#x_list1=asarray(x_list1)
#x_list2=asarray(x_list2)

#print(x_list1[2])
#print(x_list2[2])

y_list=asarray(y_list)
x_list3=np.concatenate([x_list1, x_list2])

#x_list3=x_list1+x_list2

#print(x_list3)

#trainDF = pandas.DataFrame()
#trainDF['text'] = x_list3

#import nltk
#nltk.download()
#import textcleaner as tc
#nltk.download('stopwords')
#data = tc.document(x_list3)
#data=data.remove_stpwrds()

#trainDF['label'] = y_list
#x_list3=asarray(x_list3)

def listToString(s): 
    
    # initialize an empty string
    str1 = " " 
    
    # return string  
    return (str1.join(s))

listToStr = ' '.join([str(elem) for elem in x_list3])
listToStr2 = ' '.join(map(str, x_list3))

#listToStr2=[listToStr2]

#print(listToStr2)

texts=[]
listToStr3=listToStr2.split(" ")
#print(listToStr3)
for i in listToStr3:
    texts.append(" ".join(str(i)))
#print(texts[0])


#for i in listToStr3:
#    texts.append(" ".join(i))
#print(texts)





#data=listToString(x_list3)
#print(x_list3[0])
#print(listToStr2)

#trainDF = pandas.DataFrame()
#trainDF['text'] = listToStr2
#print(trainDF['text'])
#, ngram_range=(3,3)

tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=17000)
tfidf_vect_ngram_chars.fit(texts)
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(texts)
d=xtrain_tfidf_ngram_chars[0]
print(d)
#print(type(d))
#d1=np.asarray(xtrain_tfidf_ngram_chars[0])
#d1.reshape(1,20)
#print(d1.shape)
#d2=np.asarray(xtrain_tfidf_ngram_chars[1])
#print(d1-d2)

#l=list()
#for i, line in xtrain_tfidf_ngram_chars[0].items():
#    values = line.split()
#    l.append(line)
#    print(l[i])
#    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

#print(l[0])


#xtrain_tfidf_ngram_chars=list[xtrain_tfidf_ngram_chars]

#print(xtrain_tfidf_ngram_chars[0].shape)
#xtrain_tfidf_ngram_chars[0]=list(xtrain_tfidf_ngram_chars[0])
#xtrain_tfidf_ngram_chars[0]=np.asarray(xtrain_tfidf_ngram_chars[0])
#for i in range(xtrain_tfidf_ngram_chars.shape[0]):
#    xtrain_tfidf_ngram_chars[i]=np.array(xtrain_tfidf_ngram_chars[0][i])

#d1=xtrain_tfidf_ngram_chars[1]
#print(d)

#a=np.array(xtrain_tfidf_ngram_chars[0])
#b=np.array(xtrain_tfidf_ngram_chars[1])

#a=list[xtrain_tfidf_ngram_chars[0]]
#print(a)
#d= np.sum(np.power((a-b),2))

#print(list[xtrain_tfidf_ngram_chars])

#print(d)
        
#print(xtrain_tfidf_ngram_chars[0])


# take square of differences and sum them
#l2 = np.sum(np.power((d2-d1),2))

#l=(d2-d1)*(d2-d1)
#print(l)


# In[ ]:





# In[ ]:




