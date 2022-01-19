#!/usr/bin/env python
# coding: utf-8

# In[44]:


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
from numpy import asarray
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import MaxPool1D
from tensorflow import keras
from tensorflow.python.keras import optimizers
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.utils import to_categorical

file1 = pd.read_csv("C:\\Users\\Public\\SARS2_PPI_train.csv")
file2 = pd.read_csv("C:\\Users\\Public\\SARS2_PPI_test.csv")
file3 = open("C:\\Users\\Public\\SARS2_protein_sequences.txt",'r')

#Lines = file1.readlines()
#Lines2 = file2.readlines()
Lines3 = file3.readlines()

Dict1=dict()

for line in Lines3:
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

x_list_train=[]
x_list_test=[]

x_list1_train=[]
x_list2_train=[]

x_list1_test=[]
x_list2_test=[]

y_list_train=[]
y_list_test=[]


#for line in Lines:
#    x=line.split()
#    w1=x[0]
#    w2=x[1]
#    output=x[2]
#    output=int(output)
#    seq1=Dict1[w1]
#    seq2=Dict1[w2]
#    y_list_train.append(output)
#    x_list1_train.append(str(seq1))
#    x_list2_train.append(str(seq2))
    
#y_list_train=asarray(y_list_train)
#######

for i in range(len(file1['train_seq1'])):
    w1=file1['train_seq1'][i]
    w2=file1['train_seq2'][i]
    seq1=Dict1[w1]
    seq2=Dict1[w2]
    output=file1['train_y'][i]
    output=int(output)
    y_list_train.append(output)
    x_list1_train.append(str(seq1))
    x_list2_train.append(str(seq2))
#    x_list2_train.append(str(seq2))
y_list_train=asarray(y_list_train)    

#######
for i in range(len(file2['test_seq1'])):
    w1=file2['test_seq1'][i]
    w2=file2['test_seq2'][i]
    seq1=Dict1[w1]
    seq2=Dict1[w2]
    output=file2['test_y'][i]
    output=int(output)
    y_list_test.append(output)
    x_list1_test.append(str(seq1))
    x_list2_test.append(str(seq2))
y_list_test=asarray(y_list_test)



#######



#x_list3=np.concatenate([x_list1, x_list2])

def listToString(s): 
    
    # initialize an empty string
    str1 = " " 
    
    # return string  
    return (str1.join(s))


listToStr = ' '.join([str(elem) for elem in x_list1_train])
listToStr2 = ' '.join(map(str, x_list1_train))
texts_seq1_train=[]
listToStr3=listToStr2.split(" ")

for i in listToStr3:
    texts_seq1_train.append(" ".join(str(i)))

    

listToStr = ' '.join([str(elem) for elem in x_list2_train])
listToStr2 = ' '.join(map(str, x_list2_train))
#listToStr2=[listToStr2]
#print(listToStr2)
texts_seq2_train=[]
listToStr3=listToStr2.split(" ")
#print(listToStr3)
for i in listToStr3:
    texts_seq2_train.append(" ".join(str(i)))


    
istToStr = ' '.join([str(elem) for elem in x_list1_test])
listToStr2 = ' '.join(map(str, x_list1_test))
texts_seq1_test=[]
listToStr3=listToStr2.split(" ")

for i in listToStr3:
    texts_seq1_test.append(" ".join(str(i)))    

    
istToStr = ' '.join([str(elem) for elem in x_list2_test])
listToStr2 = ' '.join(map(str, x_list2_test))
texts_seq2_test=[]
listToStr3=listToStr2.split(" ")

for i in listToStr3:
    texts_seq2_test.append(" ".join(str(i)))     
    

#####################

tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=17000)
texts_train=np.concatenate([texts_seq1_train, texts_seq2_train])
tfidf_vect_ngram_chars.fit(texts_train)
xtrain_tfidf_ngram_chars1 =  tfidf_vect_ngram_chars.transform(texts_seq1_train)
xtrain_tfidf_ngram_chars2 =  tfidf_vect_ngram_chars.transform(texts_seq2_train)
xtest_tfidf_ngram_chars1 =  tfidf_vect_ngram_chars.transform(texts_seq1_test)
xtest_tfidf_ngram_chars2 =  tfidf_vect_ngram_chars.transform(texts_seq2_test)



xtrain_tfidf_ngram_chars1=xtrain_tfidf_ngram_chars1.toarray()
d=len(xtrain_tfidf_ngram_chars1)
xtrain_tfidf_ngram_chars2=xtrain_tfidf_ngram_chars2.toarray()

xtest_tfidf_ngram_chars1=xtest_tfidf_ngram_chars1.toarray()
d2=len(xtest_tfidf_ngram_chars1)
xtest_tfidf_ngram_chars2=xtest_tfidf_ngram_chars2.toarray()

#xtrain_tfidf_ngram_chars2=asarray(xtrain_tfidf_ngram_chars2)
#x_list_train=np.array(x_list_train)

for i in range(d):
    v=np.concatenate([xtrain_tfidf_ngram_chars1[i], xtrain_tfidf_ngram_chars2[i]])
#    print(v)
    x_list_train.append(v)
    


for i in range(d2):
    v=np.concatenate([xtest_tfidf_ngram_chars1[i], xtest_tfidf_ngram_chars2[i]])
    x_list_test.append(v)
    
      
######################


#x_train, x_test, y_train, y_test=train_test_split(x_list,y_list,train_size=0.6) 


x_list_train=asarray(x_list_train)
x_list_test=asarray(x_list_test)

#y_train=asarray(y_train)

#y_train=asarray(y_train)
#y_test=asarray(y_test)

#y_list_train

num_folds=5
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1

#inputs=asarray(x_list)
#targets=asarray(y_list)
#inputs=inputs.reshape(108680,1,12,1)
#targets=targets.reshape(108680,1)
#inputs =inputs.astype(int)
#targets=targets.astype(int)
#targets= to_categorical(targets)

x_list_train=asarray(x_list_train)
y_list_train=asarray(y_list_train)

x_list_train=x_list_train.reshape(4774,1,40,1)
#y_list_train=y_list_train.reshape(4774,1)

x_list_test=x_list_test.reshape(1078,1,40,1)
#y_list_test=y_list_test.reshape(1078,1)

y_list_train=to_categorical(y_list_train)
y_list_test=to_categorical(y_list_test)


model = keras.Sequential()
model.add(Conv2D(128,(1,20),activation='relu', input_shape=(1,40,1)))
model.add(Conv2D(64,(1,20), activation='relu'))
model.add(Flatten()) 
model.add(keras.layers.Dense(2, activation='softmax'))
    
#model.add(Activation("softmax"))
#model.compile(loss=keras.losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(x_list_train, y_list_train, epochs=5)
r=model.predict(x_list_test)
print(r)
    
# score=model.evaluate(inputs[test],targets[test])

r=r.flatten()
d=y_list_test.flatten()
score=np.corrcoef(d,r)
print(score)
    
#    correlation_coefficient(targets[test],r) 
    
#    print("Corellation efficient:", score)
#    print(r)

#    acc_per_fold.append(scores[1] * 100)
#    loss_per_fold.append(scores[0])

  # Increase fold number
#    fold_no = fold_no + 1


# In[ ]:





# In[ ]:




