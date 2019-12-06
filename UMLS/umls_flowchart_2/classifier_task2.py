import logging
import os
import sys
import configparser
from sqlite import SQLite
from collections import OrderedDict
from collections import Counter
from gensim.models import KeyedVectors

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
        semantics_and_synonyms = []
        for semantic in semantics_for_word:
            semantics_and_synonyms.append((semantic, "||".join(synonyms_for_word)))
        print("smantics Length: " + str(len(semantics_and_synonyms)) + " Word Length: " + str(len(worde)))
        return semantics_and_synonyms, worde

    # def get_semantics(self):
    #     if not self.did_check_dbs:
    #         DBCheck.check_database()
    #         self.did_check_dbs = True
    #
    #     sql = 'SELECT DISTINCT TUI, STY FROM MRSTY'
    #
    #     sem_dict = {}
    #     for res in self.sqlite.execute(sql):
    #         sem_dict[res[0]] = res[1]
    #     return sem_dict

    # def get_semantics_for_word(self, word):
    #     if word is None or len(word) < 1:
    #         return []
    #
    #     if not self.did_check_dbs:
    #         DBCheck.check_database()
    #         self.did_check_dbs = True
    #
    #     logging.info("Fetching semantics for word: {}".format(word))
    #     semantics_id_list_for_word = self.lookup_word(word, False)
    #     logging.debug("Semantics ID:\n{}".format(semantics_id_list_for_word))
    #
    #     word_semantics = []
    #     for tui in semantics_id_list_for_word:
    #         word_semantics.append(self.semantics_dict[tui])
    #
    #     return word_semantics


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
        print(len(word_list), len(word_emb))
        # for word in word_list[0]:
        #     for worde in word_emb[0]:
        #         try:
        #             # similar_word_list += self.model.most_similar(positive=[word, worde])
        #             similar_word_list = self.model.similarity(word, worde)
        #             print("synonyms: " + word + " , word: " + worde + " , similarity: " + str(similar_word_list))
        #         except:
        #             similar_word_list = "Not Available in vocabulary"
        #             print("synonyms: " + word + " , word: " + worde + " , similarity: " + str(similar_word_list))
        #             # pass
        try:
            for n in range(len(word_list)):
                # print(word_list[n])
                try:
                    # similar_word_list += self.model.most_similar(positive=[word, worde])
                    similar_word_list = self.model.similarity(word_list[n], word_emb[n])
                    print("synonyms: " + word_list[n] + " , word: " + word_emb[n] + " , similarity: " + str(
                        similar_word_list))
                except:
                    similar_word_list = "Not Available in vocabulary"
                    print("synonyms: " + word_list[n] + " , word: " + word_emb[n] + " , similarity: " + str(
                        similar_word_list))
                    # pass
        except:
            pass

        # while word in word_list and worde
        print(similar_word_list)
        return similar_word_list


if '__main__' == __name__:

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

    if not os.path.exists(tsv_file):
        raise Exception("Please provide valid path of the TSV word embedding file")

    embeddingReader = EmbeddingReader(tsv_file, word_count, vector_dimension)
    # Check DB availability
    DBCheck.check_database()

    # Create lookup obj -> This creates connection to DB
    look = UMLSLookup()

    semantics = []
    f = open("Output.txt", "w+")

    counter = 0
    list_of_words = embeddingReader.get_word_list()
    while counter < int(word_count):
        semantics_synonyms_word, word_emb = look.lookup_word(list_of_words[counter], False)
        for record in semantics_synonyms_word:
            similar_words = embeddingReader.get_cosine_similar_words(record[1].split("||"), word_emb)
            logging.debug("{} for word {}".format(similar_words, record[1]))
            semantics.append((record[0], similar_words))
        counter += 1

    print(semantics)
    print(type(semantics))
    # umls_semantics_dict = Counter(semantics).items()
    # counts = [xx[1] for xx in list(umls_semantics_dict)]
    # sum = sum(counts)
    # for key, value in umls_semantics_dict:
    #     logging.info("{} :: {}".format(key, value))
    #     f.write("{} :: {} {}%\n".format(key, value, round(int(value) / sum * 100, 2)))
    # f.close()
