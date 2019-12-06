import configparser
import os.path
from os import path
import os
import subprocess

config = configparser.ConfigParser()
config.read('example.ini')

for section_name in config.sections():
	print('Section:', section_name)
	for name, value in config.items(section_name):
		print(' {} = {}'.format(name,value))
	print()

inputsone = config.items('TestOne')[2][1]
inputstwo = config.items('TestOne')[3][1]
inputsthree = config.items('TestOne')[4][1]


