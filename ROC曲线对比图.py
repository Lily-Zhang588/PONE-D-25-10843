# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#----------------------Byes-TF-IDF--------------------------------------------------------
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer     
from sklearn.naive_bayes import MultinomialNB    
from sklearn.metrics import roc_curve,roc_auc_score
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签  
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号  
plt.figure()  

Data = pd.read_csv('data.csv',encoding ='gbk')
Data = Data[(Data['labels'] == 1) | (Data['labels'] == 2)]
Data.loc[Data['labels'] == 2,'labels'] = 0
Data['labels'] = Data['labels'].astype(int)
Data['text'] = Data['text'].astype(str)

train, test, validate = np.split(Data.sample(frac=1), [int(.6*len(Data)), int(.8*len(Data))]) 
count_v0= CountVectorizer();  
counts_all = count_v0.fit_transform(Data['text']);
count_v1= CountVectorizer(vocabulary=count_v0.vocabulary_);  
counts_train = count_v1.fit_transform(train['text']);   
count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_);  
counts_test = count_v2.fit_transform(test['text']);  
tfidftransformer = TfidfTransformer();    
train_data = tfidftransformer.fit(counts_train).transform(counts_train);
test_data = tfidftransformer.fit(counts_test).transform(counts_test); 

x_train = train_data
x_test = test_data
y_train = (train['labels']).reset_index(drop = True)
y_test = (test['labels']).reset_index(drop = True) 


clf = MultinomialNB(alpha = 0.01)   
clf.fit(x_train, y_train);  
preds = clf.predict_proba(x_test);

p = []
gg = y_test.values
preds = preds.tolist()
for i in range(len(preds)):
    a= gg[i]
    b= preds[i]
    p.append(b[a])
 
y_true = np.array(y_test)
y_score = np.array(p)
fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
AUC_ROC = roc_auc_score(y_true, y_score)

#----------------------SVM-TF-IDF-------------------------------------------------------

train, test, validate = np.split(Data.sample(frac=1), [int(.6*len(Data)), int(.8*len(Data))]) 
count_v0= CountVectorizer();  
counts_all = count_v0.fit_transform(Data['text']);
count_v1= CountVectorizer(vocabulary=count_v0.vocabulary_);  
counts_train = count_v1.fit_transform(train['text']);   
count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_);  
counts_test = count_v2.fit_transform(test['text']);  
tfidftransformer = TfidfTransformer();    
train_data = tfidftransformer.fit(counts_train).transform(counts_train);
test_data = tfidftransformer.fit(counts_test).transform(counts_test); 

x_train = train_data
x_test = test_data
y_train = (train['labels']).reset_index(drop = True)
y_test = (test['labels']).reset_index(drop = True) 

from sklearn.svm import SVC   
svclf = SVC(kernel = 'linear') 
svclf.fit(x_train,y_train)  
preds = clf.predict_proba(x_test);

p = []
gg = y_test.values
preds = preds.tolist()
for i in range(len(preds)):
    a= gg[i]
    b= preds[i]
    p.append(b[a])
 
y_true = np.array(y_test)
y_score = np.array(p)
fpr3, tpr3, thresholds = roc_curve(y_true, y_score, pos_label=1)
AUC_ROC3 = roc_auc_score(y_true, y_score)


#----------------------LSTM+word2vec--------------------------------------------------------
import sys
import keras
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

VECTOR_DIR = 'vectors.bin'
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 128
VALIDATION_SPLIT = 0.16
TEST_SPLIT = 0.2

tokenizer = Tokenizer()
tokenizer.fit_on_texts(Data['text'])
sequences = tokenizer.texts_to_sequences(Data['text'])
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(Data['labels']))

p1 = int(len(data)*(1-VALIDATION_SPLIT-TEST_SPLIT))
p2 = int(len(data)*(1-TEST_SPLIT))
x_train = data[:p1]
y_train = labels[:p1]
x_val = data[p1:p2]
y_val = labels[p1:p2]
x_test = data[p2:]
y_test = labels[p2:]

import gensim
from keras.utils import plot_model
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=True)
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
not_in_model = 0
in_model = 0
for word, i in word_index.items(): 
    if word in w2v_model:
        in_model += 1
        embedding_matrix[i] = np.asarray(w2v_model[word], dtype='float32')
    else:
        not_in_model += 1

from keras.layers import Embedding
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import LSTM, Embedding
from keras.models import Sequential

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.2))
model.add(Dense(labels.shape[1], activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, batch_size=128)

preds = model.predict(x_test)
p = []
yyy=[lis[1] for lis in y_test]
preds = preds.tolist()
for i in range(len(preds)):
    a= int(yyy[i])
    b= preds[i]
    p.append(b[a])
 
y_true = np.array(yyy).astype(int)
y_score = np.array(p)
fpr4, tpr4, thresholds = roc_curve(y_true, y_score)
AUC_ROC4 = roc_auc_score(y_true, y_score)


#----------------------CNN+word2vec--------------------------------------------------------

from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D
from keras.models import Sequential

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.2))
model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(EMBEDDING_DIM, activation='relu'))
model.add(Dense(labels.shape[1], activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128)
preds = model.predict(x_test)
p = []
preds = preds.tolist()
for i in range(len(preds)):
    a= int(yyy[i])
    b= preds[i]
    p.append(b[a])
 
y_score = np.array(p)
fpr5, tpr5, thresholds = roc_curve(y_true, y_score)
AUC_ROC5 = roc_auc_score(y_true, y_score)

#----------------------polt--------------------------------------------------------
plt.figure(dpi=400,facecolor='#FFFFFF',edgecolor='#FF0000')
plt.plot(fpr,tpr,'-',color='m',linewidth=1,label='TF-IDF,朴素贝叶斯 ROC曲线下面的AUC面积 (AUC = %0.4f)' % AUC_ROC)
plt.plot(fpr3,tpr3, "--",color='k',label='TF-IDF,SVM ROC曲线下面的AUC面积 (AUC = %0.4f)' % AUC_ROC3)
plt.plot(fpr4,tpr4, "-.",color='y',label='word2vec,LSTM ROC曲线下面的AUC面积 (AUC = %0.4f)' % AUC_ROC4)
plt.plot(fpr5,tpr5, ":",color='c',label='word2vec,CNN ROC曲线下面的AUC面积 (AUC = %0.4f)' % AUC_ROC5)
plt.title('ROC曲线')
plt.xlabel("召回率")
plt.ylabel("准确率")
plt.legend(loc="lower right")
l = plt.legend()
l.set_zorder(4)
plt.savefig("data.png")

plt.show()






