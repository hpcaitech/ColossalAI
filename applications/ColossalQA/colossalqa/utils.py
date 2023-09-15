import re
from sqlalchemy import create_engine
from sqlalchemy import Engine
from typing import Tuple, Union
from sqlalchemy import MetaData
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import SQLAlchemyError

def drop_table(engine: Engine) -> None:
    '''
    drop all existing table
    '''
    Base = declarative_base()
    metadata = MetaData()
    metadata.reflect(bind=engine)
    for key in metadata.tables:
        table = metadata.tables[key]
        if table is not None:
            Base.metadata.drop_all(engine, [table], checkfirst=True)

# def create_empty_sql_database(sql_path: str)-> Tuple[Engine, str]:
#     '''
#     create an empty sql database
#     Args:
#         sql_path: the path to the sql database, please use absolute path only
#             so that you can easily find the database file and remove it later
#     '''
#     print(sql_path)
#     sql_engine = create_engine(sql_path)
#     drop_table(sql_engine)
#     sql_engine.dispose()
    
#     sql_engine = create_engine(sql_path)
#     return sql_engine, sql_path

def create_empty_sql_database(database_uri):
    try:
        # Create an SQLAlchemy engine to connect to the database
        engine = create_engine(database_uri)

        # Create the database
        engine.connect()

        print(f"Database created at {database_uri}")
    except SQLAlchemyError as e:
        print(f"Error creating database: {str(e)}")
    return engine, database_uri

def destroy_sql_database(sql_engine: Union[Engine, str]) -> None:
    '''
    destroy an sql database
    '''
    if isinstance(sql_engine, str):
        sql_engine = create_engine(sql_engine)
    drop_table(sql_engine)
    sql_engine.dispose()
    sql_engine = None


def detect_lang_naive(s):
    '''
    naive function for language detection, should be replaced by an independant layer
    '''
    remove_nota = u'[’·°–!"#$%&\'()*+,-./:;<=>?@，。?★、…【】（）《》？“”‘’！[\\]^_`{|}~]+'
    s = re.sub(remove_nota, '', s)
    s = re.sub('[0-9]', '', s).strip()
    res = re.sub('[a-zA-Z]', '', s).strip()
    if len(res)<=0:
        return 'en'
    else:
        return 'zh'
