import configparser
import os.path
from os import path
import os
import subprocess

Config = configparser.ConfigParser()
Config['TestOne'] = {'TestVariety': 'UMLS', 'Test': 'Synonym','InputType': 'D:/capstone/embeddings/wemb_sample.tsv','WordLength': '20','VectorLength':'100'}

Config['TestTwo'] = {'TestVariety':'MIMIC', 'Test':'NeuralNetwork', 'InputOne':'D:/capstone/embeddings/wemb_sample.tsv','InputTwo':'D:/capstone/mimic/INPUTS/DIAGNOSES_ICD.csv','InputThree':'D:/capstone/mimic/INPUTS/NOTEEVENTS.csv'}

Config['TestThree'] = {'TestVariety':'UMLS','Test':'Semantics', 'InputType': 'D:/capstone/embeddings/wemb_sample.tsv','WordLength': '20','VectorLength':'100'}
with open('example.ini', 'w') as configfile:
	Config.write(configfile)

print(os.path.exists('example.ini'))

