from colossalqa.data_loader.table_dataloader import drop_table
from sqlalchemy import create_engine
from sqlalchemy import Engine
from typing import Tuple

def create_empty_sql_database(sql_path: str)-> Tuple[Engine, str]:
    '''
    create an empty sql database
    '''
    sql_engine = create_engine(sql_path)
    drop_table('users', sql_engine)
    
    sql_engine = create_engine(sql_path)
    return sql_engine, sql_path

def destroy_sql_database(sql_engine: Engine) -> None:
    '''
    destroy an sql database
    '''
    if sql_engine:
        drop_table('users', sql_engine)
        sql_engine.dispose()
        sql_engine = None

