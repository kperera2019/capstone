#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from keras import backend as K
import configparser

# In[2]:


import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

from keras import backend
from tensorflow.keras import backend
from tensorflow.python.keras import backend


# In[3]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
## Plotly
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
# Others
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
import keras


# In[4]:


print(os.getcwd())  
#os.chdir(r'C:\Users\lsahi\Documents\Lakamana_GMU_Sem3\DAEN690\mimic-iii-clinical-database-1.4')
#s.chdir(r'C:\Users\lsahi\Documents\Lakamana_GMU_Sem3\DAEN690\mimic-iii-clinical-database-1.4')


# In[5]:
config = configparser.ConfigParser()
config.read('D:\capstone\example.ini')
print('Section:', 'TestTwo')
print(' Options:', config.options('TestTwo'))
for name, value in config.items('TestTwo'):
	print(' {} = {}'.format(name,value))
print()

inputsone = config.items('TestTwo')[2][1]
inputstwo = config.items('TestTwo')[3][1]
inputsthree = config.items('TestTwo')[4][1]

class preprocess():
    
    def load_data():
        print(inputsthree)
        NOTEEVENTS=pd.read_csv('D:/capstone/mimic/INPUTS/NOTEEVENTS.csv',dtype={'ROW_ID':np.int32, 'SUBJECT_ID': np.int32,'HADM_ID': np.float64, 
                                                       'CHARTDATE':str,'STORETIME':str,'CHARTTIME':str,   
                                                       'STORETIME': str,'CATEGORY': str,'DESCRIPTION':str,'CGID':str,'ISERROR':str,
                                                       'TEXT':str}, parse_dates=['CHARTDATE'])
        DIAGNOSES_ICD=pd.read_csv('D:/capstone/mimic/INPUTS/DIAGNOSES_ICD.csv',dtype={'ROW_ID':np.int32, 'SUBJECT_ID': np.int32,'HADM_ID': np.int32,
                                                             'SEQ_NUM':  np.float64,'ICD9_CODE':str})
        DIAGNOSES_ICD['ICD9_CODE']=DIAGNOSES_ICD['ICD9_CODE'].str.pad(4,'left','0')
        DIAGNOSES_ICD['ICD9_CHAP']=DIAGNOSES_ICD['ICD9_CODE'].str.slice(0,3)
        DIAGNOSES_ICD=DIAGNOSES_ICD[~DIAGNOSES_ICD['ICD9_CODE'].str.slice(0,1).isin(['V','E','U','8','9'])]
        print(DIAGNOSES_ICD)
        return DIAGNOSES_ICD, NOTEEVENTS
    
    def diag_icd(DIAGNOSES_ICD):
        DIAGNOSES_ICD = pd.concat([DIAGNOSES_ICD,pd.get_dummies(DIAGNOSES_ICD['ICD9_CHAP'], prefix='')],axis=1)
        DIAGNOSES_ICD = DIAGNOSES_ICD.drop(["ROW_ID", "SUBJECT_ID", "SEQ_NUM", "ICD9_CODE", "ICD9_CHAP"], axis = 1)
        b = DIAGNOSES_ICD.groupby('HADM_ID').sum()
        b = b.replace([2,3,4,5,6,7,8,9], 1)
        DIAGNOSES_ICD_freq=pd.DataFrame(b)
        DIAGNOSES_ICD_freq = DIAGNOSES_ICD_freq.reset_index()
        return DIAGNOSES_ICD_freq  
    
    def noteevents(NOTEEVENTS):
        selected_doc=['Nutrition']
        df=NOTEEVENTS[NOTEEVENTS['CATEGORY'].isin(selected_doc)].groupby('HADM_ID')['TEXT'].apply(lambda x: "{%s}" % ', '.join(x))
        df2=pd.DataFrame(df)
        df2 = df2.reset_index()
        return df2
    
    def join_data(df2, DIAGNOSES_ICD_freq):
        embed_size = 300 # how big is each word vector
        max_features = 64763  # how many unique words to use (i.e num rows in embedding vector)
        maxlen = 300 # max number of words in a question to use
        NOTE_DIAGNOSES = pd.merge(df2, DIAGNOSES_ICD_freq, on = 'HADM_ID')
        train, test = model_selection.train_test_split(NOTE_DIAGNOSES,test_size=0.2)
        print('Size of train: '+str(train.shape[0])+' \nSize of test: '+str(test.shape[0]) )
        train_df, val_df = train_test_split(train, test_size=0.1, random_state=2018)
        train_X = train_df["TEXT"].fillna("_na_").values
        val_X = val_df["TEXT"].fillna("_na_").values
        test_X = test["TEXT"].fillna("_na_").values

        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(train_X)
        train_X = tokenizer.texts_to_sequences(train_X)
        val_X = tokenizer.texts_to_sequences(val_X)
        test_X = tokenizer.texts_to_sequences(test_X)

        train_X = pad_sequences(train_X, maxlen=maxlen)
        val_X = pad_sequences(val_X, maxlen=maxlen)
        test_X = pad_sequences(test_X, maxlen=maxlen)

        train_y = train_df.drop(['HADM_ID', 'TEXT'], axis = 1)
        val_y = val_df.drop(['HADM_ID', 'TEXT'], axis = 1)
        test_y = test.drop(['HADM_ID', 'TEXT'], axis = 1)
        return NOTE_DIAGNOSES, train_X, val_X, test_X, train_y, val_y, test_y, tokenizer
    
    
    
    


# In[8]:


class process():

    def embedding(x, tokenizer):
        embeddings_index = {}

        f = open(os.path.join(r'C:\Users\lsahi\Downloads', x), encoding = "utf-8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:])
            coefs = np.array(coefs, dtype=float)
            embeddings_index[word] = coefs
        f.close()
        embedding_matrix = np.zeros((20000, 100))
        for word, index in tokenizer.word_index.items():
            if index > 20000 - 1:
                break
            else:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector
        return embedding_matrix
    
    def model(embedding_matrix):
        
        def recall_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision_m(y_true, y_pred):
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
                precision = true_positives / (predicted_positives + K.epsilon())
                return precision

        def f1_m(y_true, y_pred):
            precision = precision_m(y_true, y_pred)
            recall = recall_m(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))
        
        model_glove = Sequential()
        model_glove.add(Embedding(20000, 100, input_length=300, weights=[embedding_matrix], trainable=False))
        model_glove.add(Dropout(0.2))
        model_glove.add(Conv1D(64, 5, activation='relu'))
        model_glove.add(MaxPooling1D(pool_size=4))
        model_glove.add(LSTM(100))
        model_glove.add(Dense(625, activation='sigmoid'))
        model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m,precision_m, recall_m])

        model_glove.fit(train_X, train_y, epochs = 50, verbose=True)
        return model_glove
    


# In[9]:

inputstwo, inputsthree = preprocess.load_data() 
print("DATA PREPROCESSED")
DIAGNOSES_ICD_freq = preprocess.diag_icd(inputstwo)
df2 = preprocess.noteevents(inputsthree)
NOTE_DIAGNOSES, train_X, val_X, test_X, train_y, val_y, test_y, tokenizer = preprocess.join_data(df2, DIAGNOSES_ICD_freq)






tsv_file = inputsone

embedding_matrix = process.embedding(tsv_file, tokenizer)
model_glove = process.model(embedding_matrix)

print(model_glove)
# In[ ]:




