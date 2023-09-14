import re
from sqlalchemy import create_engine
from sqlalchemy import Engine
from typing import Tuple, Union
from sqlalchemy import MetaData
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base


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

def create_empty_sql_database(sql_path: str)-> Tuple[Engine, str]:
    '''
    create an empty sql database
    '''
    sql_engine = create_engine(sql_path)
    drop_table(sql_engine)
    
    sql_engine = create_engine(sql_path)
    return sql_engine, sql_path

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
