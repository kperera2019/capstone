# UMLS Schema Counter
The code here, takes a TSV file of word embedding as input and provides the distribution of embedding vocab in various UMLS semantics

  - The incoming TSV file should be a simple TSV with 1st column as word and other columns as dimensions of the vector
  - The total length of vocab and the dimensions of the vector must be provided as command line arguments to `classifier.py`
 The sample command to run the classifier
```
python classifier.py ./wemb_sample.tsv 8000 100
```

# Important Features!

The `classifier.py` requires UMLS data extracted inform of umls.db (sqlite db). In order to create the database, run the `umls.sh` as per the instructions below:

  - Because the `umls.sh` is a shell script, it must be ran in a linux shell. If you are using linux based system, simply run the script in the terminal window. If you are using Windows system, run the script in `git bash shell`
  - Extract the `umls-2019AA-metathesaurus.zip` to a suitable location
  - Go to databases folder within this folder where `umls.sh` is located
  - Run the following command

> ./umls.sh <Provide the path to the UMLS extracted directory, which is named something like "2019AA" and contains a "META" directory, as first argument when invoking this script.> 
e.g. Suppose the extracted directory is umls-2019AA-metathesaurus in C drive then command is:
```
./umls.sh /c/umls-2019AA-metathesaurus/2019AA
```
