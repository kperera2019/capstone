#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
from owlready2.pymedtermino2.umls import *
from sqlite import SQLite  # for py-umls standalone


class UMLSDB(object):

    @classmethod
    def check_umls_database(cls):
        umls_db = os.path.join('databases', 'umls.db')
        if not os.path.exists(umls_db):
            raise Exception("The UMLS database at {} does not exist.".format(os.path.abspath(umls_db)))

    @classmethod
    def check_pym_database(cls):
        pym_db = os.path.join('databases', 'pym.sqlite3')
        return os.path.exists(pym_db)


class UMLSLookup(object):
    sqlite = None
    did_check_dbs = False

    def __init__(self):
        absolute = os.path.dirname(os.path.realpath(__file__))
        self.sqlite = SQLite.get(os.path.join(absolute, 'databases/umls.db'))

    def lookup_cui(self):
        if not UMLSLookup.did_check_dbs:
            UMLSDB.check_umls_database()
            UMLSLookup.did_check_dbs = True

        sql = 'SELECT DISTINCT CUI FROM descriptions'
        # return as list
        return self.sqlite.execute(sql).fetchall()


class OwlReady2Lookup(object):
    CUI = None

    def __init__(self):
        default_world.set_backend(filename="databases/pym.sqlite3")
        if not UMLSDB.check_pym_database():
            import_umls("umls-2019AA-metathesaurus.zip", terminologies=["ICD10", "SNOMEDCT_US", "CUI"])
            default_world.save()
        PYM = get_ontology("http://PYM/").load()
        self.CUI = PYM["CUI"]

    def lookup_cui(self, cui):
        concept = self.CUI[cui]
        if concept:
            return [concept.name, concept.label[0]]


# running this as a script does the database setup/check
if '__main__' == __name__:
    UMLSDB.check_umls_database()

    look = UMLSLookup()
    CUI_List = (look.lookup_cui())
    owl = OwlReady2Lookup()

    filename = "CUI_SYN.csv"
    f = open(filename, 'w', encoding='utf-8')
    headers = "cui_name,cui_label\n"
    f.write(headers)

    for x in CUI_List:
        concept_name_cui = owl.lookup_cui(x[0])
        if concept_name_cui:
            f.write(','.join(concept_name_cui))
            f.write("\n")
    f.close()
