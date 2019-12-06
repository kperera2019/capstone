import logging
import os
import sys

from sqlite import SQLite
from collections import OrderedDict
from collections import Counter
from gensim.models import KeyedVectors
import pandas as pd
import openpyxl

logging.basicConfig(format='%(asctime)s %(process)d %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)


class DBCheck(object):
    """ A utility class for checking connection to SQLITE UMLS DB."""

    @classmethod
    def check_database(cls):
        """ Check if our database is in place and if not, prompts to import it. Will raise on errors!

        UMLS: (umls.db)
        If missing prompt to use the `umlsdbload` script
        """

        umls_db = os.path.join('../../databases', 'umls.db')
        if not os.path.exists(umls_db):
            raise Exception("The UMLS database at {} does not exist. Run the import script `umlsdbloadscript`.".format(os.path.abspath(umls_db)))


class UMLSLookup(object):
    sqlite = None
    did_check_dbs = False
    preferred_sources = ['"SNOMEDCT"', '"MTH"']
    semantics_dict = {}

    def __init__(self):
        absolute = os.path.dirname(os.path.realpath(__file__))
        self.sqlite = SQLite.get("../../databases/umls.db")

    def lookup_word(self, word, preferred=True):
        if word is None or len(word) < 1:
            return []

        if not self.did_check_dbs:
            DBCheck.check_database()
            self.did_check_dbs = True

        if preferred:
            sql = 'SELECT STY, lower(STR) FROM descriptions WHERE CUI = (SELECT CUI FROM descriptions WHERE STR = ? COLLATE NOCASE) AND SAB IN ({})'.format(", ".join(UMLSLookup.preferred_sources))
        else:
            sql = 'SELECT STY, lower(STR) FROM descriptions WHERE CUI = (SELECT CUI FROM descriptions WHERE STR = ? COLLATE NOCASE)'

        # return as list
        semantics_for_word = []
        synonyms_for_word = []
        worde = []
        # Separates 2 lists, one for semantics and one for synonyms
        for res in self.sqlite.execute(sql, (word,)):
            semantics_for_word += ([x for x in res[0].split('|')])
            synonyms_for_word.append(res[1])
            worde.append(word)

        semantics_for_word = list(OrderedDict.fromkeys(semantics_for_word))
        synonyms_for_word = list(OrderedDict.fromkeys(synonyms_for_word))
        if len(semantics_for_word) == 0:
            semantics_for_word.append('Out-Of-UMLS')

        logging.debug("semantics for word {} are {}".format(word, semantics_for_word))
        logging.debug("synonyms for word {} are {}".format(word, synonyms_for_word))

        # Grouping synonyms by semantics
        semantics_and_synonyms = []
        for semantic in semantics_for_word:
            semantics_and_synonyms.append((semantic, "||".join(synonyms_for_word)))
        return semantics_and_synonyms, worde

class EmbeddingReader(object):
    model = None

    def __init__(self, tsv, count, dimension):
        file_name = os.path.basename(tsv).split(".")[0]
        dest_file = str(file_name + '.txt')
        if not os.path.exists(os.path.curdir + "/" + dest_file):
            logging.info("{} doesn't exist".format(dest_file))
            with open(tsv, 'r') as inp, open(dest_file, 'w') as outp:
                line_count = count  # line count of the tsv file (as string)
                dimensions = dimension  # vector size (as string)
                outp.write(' '.join([line_count, dimensions]) + '\n')
                for line in inp:
                    words = line.strip().split()
                    outp.write(' '.join(words) + '\n')
        else:
            logging.info("{} already exists".format(dest_file))
        self.model = KeyedVectors.load_word2vec_format(dest_file, binary=False)
        print(self.model.vocab)

    def get_word_list(self):
        return list(self.model.vocab)

    def get_cosine_similar_words(self, word_list, word_emb):
        similar_word_list = []
        count = 0 # Number of words for which cosine similarity can be found
        similarity = 0 # Cosine similarity for each word
        total_similarity = 0 # Cosine similarity for each semantic
        no_count = 0
        try:
            for n in range(len(word_list)):
                try:
                    similar_word_list = self.model.similarity(word_list[n], word_emb[n])
                    total_similarity = total_similarity + similar_word_list
                    count = count + 1

                    print("synonyms: " + word_list[n] + " , word: " + word_emb[n] + " , similarity: " + str(
                        similar_word_list))
                except:
                    similar_word_list1 = "Not Available in vocabulary"
                    no_count = no_count + 1
                    print("synonyms: " + word_list[n] + " , word: " + word_emb[n] + " , similarity: " + str(
                        similar_word_list1))
            similarity = total_similarity
        except:
            pass

        return similarity, count, no_count


if '__main__' == __name__:

<<<<<<< HEAD:UMLS/umls_flowchart_2/classifier_task2.py
    config = configparser.ConfigParser()
    config.read('D:\capstone\example.ini')
    print('Section:', 'TestThree')
    print(' Options:', config.options('TestThree'))
    for name, value in config.items('TestThree'):
        print(' {} = {}'.format(name,value))
    print()
    inputsone = config.items('TestThree')[2][1]
    inputstwo = config.items('TestThree')[3][1]
    inputsthree = config.items('TestThree')[4][1]

    tsv_file = inputsone
    word_count = inputstwo
    vector_dimension = inputsthree
    # if not tsv_file:
    #     raise Exception("Please provide path of the TSV word embedding file")
=======
    if len(sys.argv) != 4:
        raise Exception("Please provide valid parameter as <Path of word embedding TSV> <Vocab length> <Dimension of vectors for words>")

    tsv_file = sys.argv[1]
    word_count = sys.argv[2]
    vector_dimension = sys.argv[3]
>>>>>>> 22f79f0a85095c0a579b6a05b53fa0b9f746aef3:UMLS/umls_flowchart_2/vicinity_analysis.py

    if not os.path.exists(tsv_file):
        raise Exception("Please provide valid path of the TSV word embedding file")

    embeddingReader = EmbeddingReader(tsv_file, word_count, vector_dimension)
    # Check DB availability
    DBCheck.check_database()

    # Create lookup obj -> This creates connection to DB
    look = UMLSLookup()

    semantics = []
    repeat = 0 # To avoid words that already been seen as synonym
    counter = 0
    list_of_words = embeddingReader.get_word_list()
    while counter < int(word_count):
        # For the first word there will be no words in "semantics" list to compare the words with
        if len(semantics) == 0:
            semantics_synonyms_word, word_emb = look.lookup_word(list_of_words[counter], False)
            for record in semantics_synonyms_word:
                similar_words, words_present, words_not_present = embeddingReader.get_cosine_similar_words(record[1].split("||"), word_emb)
                logging.debug("{} for word {}".format(similar_words, record[1]))
                semantics.append((record[0], words_present + words_not_present, words_present, similar_words))
            counter += 1
        # If any word has alredy seen as synonym then, declare j = 1
        else:
            for term in semantics:
                if list_of_words[counter] == term:
                    repeat = 1

        if repeat == 1:
            logging.debug("Synonyms for word {} has been already stated".format(record[1]))
            counter += 1
        else:
            semantics_synonyms_word, word_emb = look.lookup_word(list_of_words[counter], False)
            for record in semantics_synonyms_word:
                similar_words, words_present, words_not_present = embeddingReader.get_cosine_similar_words(
                    record[1].split("||"), word_emb)
                logging.debug("{} for word {}".format(similar_words, record[1]))
                semantics.append((record[0], words_present + words_not_present, words_present, similar_words))
            counter += 1


    semantics_df = pd.DataFrame(semantics, columns=['semantics', 'total', 'present', 'avg'])
    semantics_grp = semantics_df.groupby(['semantics']).agg({'total': 'sum', 'present': 'sum', 'avg': 'sum'})
    semantics_grp = semantics_grp.reset_index()
    semantics_grp['avg'] = semantics_grp['avg'] / semantics_grp['present']
    #Final Output
    print(semantics_grp)
    # Output saved as excel
    semantics_grp.to_excel(r'output.xlsx')

