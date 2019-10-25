import sqlite3
from sqlite3 import Error
from glob import glob
import os
import io


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


# ......................................................
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn


# ...................................................
def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


# ....................................................
def main():
    database = r"C:\Users\nikita\Desktop\data_analytics_material\DAEN_690\capstone_local\pythonsqlite.db"
    directory = r"C:\Users\nikita\Desktop\data_analytics_material\DAEN_690\capstone_local\UMLS_metathesaurus\2019AA\META"

    sql_create_MRDEF_table = """ CREATE TABLE IF NOT EXISTS MRDEF(
                                        CUI varchar,
		                                AUI varchar,
		                                ATUI varchar,
		                                SATUI varchar,
		                                SAB varchar,
		                                DEF text,
		                                SUPPRESS varchar,
		                                CVF varchar
                                    ); """

    sql_create_MRCONSO_table = """CREATE TABLE IF NOT EXISTS MRCONSO (
                                   CUI varchar,
		                           LAT varchar,
		                           TS varchar,
		                           LUI varchar,
		                           STT varchar,
		                           SUI varchar,
		                           ISPREF varchar,
		                           AUI varchar,
		                           SAUI varchar,
		                           SCUI varchar,
		                           SDUI varchar,
		                           SAB varchar,
		                           TTY varchar,
		                           CODE varchar,
		                           STR text,
		                           SRL varchar,
		                           SUPPRESS varchar,
		                           CVF varchar
                                );"""
    sql_create_MRSTY_table = """CREATE TABLE IF NOT EXISTS MRSTY (
                                       CUI varchar,
		                               TUI varchar,
		                               STN varchar,
		                               STY text,
		                               ATUI varchar,
		                               CVF varchar
                                    );"""

    # create a database connection
    conn = create_connection(database)

    # create tables
    if conn is not None:
        # create MRDEF table
        create_table(conn, sql_create_MRDEF_table)

        # create MRCONSO table
        create_table(conn, sql_create_MRCONSO_table)

        # create MRSTY table
        create_table(conn, sql_create_MRSTY_table)
    else:
        print("Error! cannot create the database connection.")

    conn.execute("CREATE INDEX X_CUI_MRDEF ON MRDEF (CUI);")
    conn.execute("CREATE INDEX X_SAB_MRDEF ON MRDEF (SAB);")
    conn.execute("CREATE INDEX X_CUI_MRCONSO ON MRCONSO (CUI);")
    conn.execute("CREATE INDEX X_LAT_MRCONSO ON MRCONSO (LAT);")
    conn.execute("CREATE INDEX X_TS_MRCONSO ON MRCONSO (TS);")
    conn.execute("CREATE INDEX X_CUI_MRSTY ON MRSTY (CUI);")
    conn.execute("CREATE INDEX X_TUI_MRSTY ON MRSTY (TUI);")

    list_files3(directory, 'pipe', conn)
    # list_files4(directory, 'pipe', conn)
    # list_files5(directory, 'pipe', conn)

    # return cur.lastrowid

    # conn.execute("CREATE TABLE descriptions AS SELECT CUI, LAT, SAB, TTY, STR FROM MRCONSO WHERE LAT = 'ENG' AND TS = 'P' AND ISPREF = 'Y'")
    # conn.execute("ALTER TABLE descriptions ADD COLUMN STY TEXT")
    # conn.execute("CREATE INDEX X_CUI_desc ON descriptions (CUI)")
    # conn.execute("UPDATE descriptions SET STY = (SELECT GROUP_CONCAT(MRSTY.TUI, '|') FROM MRSTY WHERE MRSTY.CUI = descriptions.CUI GROUP BY MRSTY.CUI")


def list_files3(directory, pipe, conn):
    # it = glob(directory + '\*.' + pipe)
    # print(it)
    # for x in range(len(it)):
    for f in io.open(
            r"C:\Users\nikita\Desktop\data_analytics_material\DAEN_690\capstone_local\UMLS_metathesaurus\2019AA\META\MRCONSO.pipe",
            'r', encoding='utf8'):
        f.replace('\n', '')
        print(f)
        print(f.split('|'))
        values = f.split('|')
        print(values)
        print(len(values))
        sql = "INSERT INTO MRCONSO VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        conn.execute(sql, values)
        conn.commit()

def list_files4(directory, pipe, conn):
    # it = glob(directory + '\*.' + pipe)
    # print(it)
    # for x in range(len(it)):
    for f in io.open(r"C:\Users\nikita\Desktop\data_analytics_material\DAEN_690\capstone_local\UMLS_metathesaurus\2019AA\META\MRDEF.pipe",
            'r', encoding='utf8'):
        f.replace('\n', '')
        print(f)
        print(f.split('|'))
        values = f.split('|')
        print(values)
        print(len(values))
        sql = "INSERT INTO MRDEF VALUES(?,?,?,?,?,?)"
        conn.execute(sql, values)
        conn.commit()

def list_files4(directory, pipe, conn):
    # it = glob(directory + '\*.' + pipe)
    # print(it)
    # for x in range(len(it)):
    for f in io.open(r"C:\Users\nikita\Desktop\data_analytics_material\DAEN_690\capstone_local\UMLS_metathesaurus\2019AA\META\MRSTY.pipe",
            'r', encoding='utf8'):
        f.replace('\n', '')
        print(f)
        print(f.split('|'))
        values = f.split('|')
        print(values)
        print(len(values))
        sql = "INSERT INTO MRSTY VALUES(?,?,?,?,?,?)"
        conn.execute(sql, values)
        conn.commit()

if __name__ == '__main__':
    main()
    create_connection(r"C:\Users\nikita\Desktop\data_analytics_material\DAEN_690\capstone_local\pythonsqlite.db")
