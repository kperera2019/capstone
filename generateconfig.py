import configparser
import os.path
from os import path
import os
import subprocess

Config = configparser.ConfigParser()
Config['TestOne'] = {'TestVariety': 'UMLS', 'Test': 'Synonym','InputType': 'D:/capstone local/UMLS/umls_flowchart_1/wemb_sample.tsv','WordLength': '20','VectorLength':'100'}

Config['TestTwo'] = {'TestVariety':'MIMIC', 'Test': 'Cosine'}

Config['TestThree'] = {'TestVariety':'Something'}

with open('example.ini', 'w') as configfile:
	Config.write(configfile)

print(os.path.exists('example.ini'))

