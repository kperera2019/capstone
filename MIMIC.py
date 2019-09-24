# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:14:27 2019

@author: Jeyan
"""
import os
import pandas as pd
import matplotlib as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt


print(os.getcwd())  
os.chdir(r'C:\\Users\\murug\\Desktop\\Jey\\GMU\\Semester 3\\DAEN 690\\mimic-iii-clinical-database-1.4')

data1 = pd.read_csv('DIAGNOSES_ICD.csv.gz', nrows=100, compression='gzip', error_bad_lines=False)
print(data1)

data2 = pd.read_csv('NOTEEVENTS.csv.gz', nrows=100, compression='gzip', error_bad_lines=False)
print(data2)


data3 = pd.read_csv('MICROBIOLOGYEVENTS.csv.gz', nrows=100, compression='gzip', error_bad_lines=False)
data3.info()


data1.info()
data1.isnull().sum()
data2.info()
data2.isnull().sum()
np.unique(data1.ICD9_CODE)

pd.set_option('display.max_colwidth',-1)
data2['TEXT']

sns.set(rc={'figure.figsize':(20,20)})
chart = sns.countplot(data1['ICD9_CODE'], palette='Set1')
chart.set_xticklabels(chart.get_xticklabels(), rotation=50)

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
word_string=" ".join(data2['TEXT'].str.lower())
wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white', 
                      max_words=300
                         ).generate(word_string)
plt.clf()
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


