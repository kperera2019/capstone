from owlready2 import *
from owlready2.pymedtermino2 import *
from owlready2.pymedtermino2.umls import *
default_world.set_backend(filename = "pym.sqlite3")
import_umls("umls-2019AA-metathesaurus.zip", terminologies = ["ICD10", "SNOMEDCT_US", "CUI"])
default_world.save()