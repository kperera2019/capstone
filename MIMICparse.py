import pandas as pd
data = pd.read_csv('D:\DAEN 690\mimic-iii-clinical-database-1.4\mimic-iii-clinical-database-1.4\DIAGNOSES_ICD.csv.gz', nrows=100, compression='gzip', error_bad_lines=False)
print(data)