import logging
import os
import sys
import matplotlib.pyplot as plt

from sqlite import SQLite
from collections import OrderedDict
from collections import Counter
from gensim.models import KeyedVectors

logging.basicConfig(format='%(asctime)s %(process)d %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


class DBCheck(object):
    """ A utility class for checking connection to SQLITE UMLS DB."""

    @classmethod
    def check_database(cls):
        """ Check if our database is in place and if not, prompts to import it. Will raise on errors!

        UMLS: (umls.db)
        If missing prompt to use the `umlsdbload` script
        """

        umls_db = os.path.join('databases', 'umls.db')
        if not os.path.exists(umls_db):
            raise Exception("The UMLS database at {} does not exist. Run the import script `umlsdbloadscript`.".format(os.path.abspath(umls_db)))


class UMLSLookup(object):
    sqlite = None
    did_check_dbs = False
    preferred_sources = ['"SNOMEDCT"', '"MTH"']
    semantics_dict = {}

    def __init__(self):
        absoulte = os.path.dirname(os.path.realpath(__file__))
        self.sqlite = SQLite.get(os.path.join(absoulte, "databases/umls.db"))

    def lookup_word(self, word, preferred=True):
        if word is None or len(word) < 1:
            return []

        if not self.did_check_dbs:
            DBCheck.check_database()
            self.did_check_dbs = True

        if preferred:
            sql = 'SELECT STY FROM descriptions WHERE STR LIKE ? AND SAB IN ({})'.format(
                ", ".join(UMLSLookup.preferred_sources))
        else:
            sql = 'SELECT STY FROM descriptions WHERE STR LIKE ?'

        # return as list
        arr = []
        for res in self.sqlite.execute(sql, ('%' + word + '%',)):
            arr += (res[0].split('|'))

        arr = list(OrderedDict.fromkeys(arr))
        return arr

    def get_semantics(self):
        if not self.did_check_dbs:
            DBCheck.check_database()
            self.did_check_dbs = True

        sql = 'SELECT DISTINCT TUI, STY FROM MRSTY'

        sem_dict = {}
        for res in self.sqlite.execute(sql):
            sem_dict[res[0]] = res[1]
        return sem_dict

    def get_semantics_for_word(self, word):
        if word is None or len(word) < 1:
            return []

        if not self.did_check_dbs:
            DBCheck.check_database()
            self.did_check_dbs = True

        if len(self.semantics_dict) == 0:
            logging.info("Loading semantics from UMLS DB")
            self.semantics_dict = self.get_semantics()

        logging.info("Fetching semantics for word: {}".format(word))
        semantics_id_list_for_word = self.lookup_word(word)
        logging.debug("Semantics ID:\n{}".format(semantics_id_list_for_word))

        word_semantics = []
        for tui in semantics_id_list_for_word:
            word_semantics.append(self.semantics_dict[tui])

        return word_semantics


class EmbeddingReader(object):

    @staticmethod
    def convert_to_word2vec_embedding(tsv, count, dimension):
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
        return KeyedVectors.load_word2vec_format(dest_file, binary=False)


if '__main__' == __name__:

    if len(sys.argv) != 4:
        raise Exception("Please provide valid parameter as <Path of word embedding TSV> <Vocab length> <Dimension of vectors for words>")

    tsv_file = sys.argv[1]
    word_count = sys.argv[2]
    vector_dimension = sys.argv[3]

    # if not tsv_file:
    #     raise Exception("Please provide path of the TSV word embedding file")

    if not os.path.exists(tsv_file):
        raise Exception("Please provide valid path of the TSV word embedding file")

    embeddingReader = EmbeddingReader()
    model = embeddingReader.convert_to_word2vec_embedding(tsv_file, word_count, vector_dimension)

    # Check DB availability
    DBCheck.check_database()

    # Create lookup obj -> This creates connection to DB
    look = UMLSLookup()

    semantics = []
    f = open("Output.txt", "w+")

    counter = 0
    listofvocab = list(model.vocab)
    while(counter < int(word_count)):
        semantics_word = look.get_semantics_for_word(listofvocab[counter])
        if len(semantics_word) == 0:
            semantics_word.append("Out-of-UMLS")
        semantics += semantics_word
        counter += 1

    umls_semantics_dict = Counter(semantics).items()
    counts = [xx[1] for xx in list(umls_semantics_dict)]
    sum = sum(counts)
    for key, value in umls_semantics_dict:
        logging.info("{} :: {}".format(key, value))
        f.write("{} :: {} {}%\n".format(key, value, round(int(value) / sum * 100, 2)))
    f.close()
    # final_dict = Counter(semantics_for_allergy + semantics_for_fever)
    # print(list(final_dict.keys()))
    # print(list(final_dict.values()))
    # plt.bar(list(final_dict.keys()), list(final_dict.values()))
    # plt.show()
