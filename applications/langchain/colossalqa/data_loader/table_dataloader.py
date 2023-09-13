import pandas as pd
import os
from sqlalchemy import MetaData
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base
from typing import Any

SUPPORTED_DATA_FORMAT = ['.csv','.xlsx', '.xls','.json','.html','.h5', '.hdf5','.parquet','.feather','.dta']

def drop_table(engine: Any) -> None:
    # drop all existing table
    Base = declarative_base()
    metadata = MetaData()
    metadata.reflect(bind=engine)
    for key in metadata.tables:
        table = metadata.tables[key]
        if table is not None:
            Base.metadata.drop_all(engine, [table], checkfirst=True)

class TableLoader:
    def __init__(self, files: str, sql_path:str='sqlite:///mydatabase.db', **kwargs) -> None:
        '''
        Args:
            files: list of files (list[file path, name])
            sql_path: how to serve the sql database
            **kwargs: keyword type arguments, useful for certain document types 
        '''
        self.data = {}
        self.sql_path = sql_path
        self.kwargs = kwargs
        self.sql_engine = create_engine(self.sql_path)
        drop_table(self.sql_engine)
        
        self.sql_engine = create_engine(self.sql_path)
        for item in files:
            path = item[0]
            dataset_name = item[1]
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} doesn't exists")
            if not any([path.endswith(i) for i in SUPPORTED_DATA_FORMAT]):
                raise TypeError(f"{path} not supported. Supported type {SUPPORTED_DATA_FORMAT}")
            
            print("loading data")
            self.load_data(path)
            print("data loaded")
            self.to_sql(path, dataset_name)

    def load_data(self, path):
        '''
        load data and serve the data as sql database.
        data must be in pandas format
        '''
        if path.endswith('.csv'):
            # load csv
            self.data[path] = (pd.read_csv(path))
        elif path.endswith('.xlsx') or path.endswith('.xls'):
            # load excel
            self.data[path] = (pd.read_excel(path))  # You can adjust the sheet_name as needed
        elif path.endswith('.json'):
            # load json
            self.data[path] = (pd.read_json(path))
        elif path.endswith('.html'):
            # load html
            html_tables = pd.read_html(path)
            # Choose the desired table from the list of DataFrame objects
            self.data[path] = (html_tables[0])  # You may need to adjust this index
        elif path.endswith('.h5') or path.endswith('.hdf5'):
            # load h5
            self.data[path] = (pd.read_hdf(path, key=self.kwargs.get('key', 'data')))  # You can adjust the key as needed
        elif path.endswith('.parquet'):
            # load parquet
            self.data[path] = (pd.read_parquet(path, engine='fastparquet'))
        elif path.endswith('.feather'):
            # load feather
            self.data[path] = (pd.read_feather(path))
        elif path.endswith('.dta'):
            # load dta
            self.data[path] = (pd.read_stata(path))
        else:
            raise ValueError("Unsupported file format")
        
    def to_sql(self, path, table_name):
        '''
        serve the data as sql database.
        '''
        self.data[path].to_sql(table_name, con=self.sql_engine, if_exists='replace', index=False)
        print(f"loaded to Sqlite3\nPath: {path}")
        return self.sql_path
    
    def get_sql_path(self):
        return self.sql_path

    def __del__(self):
        if self.sql_engine:
            drop_table(self.sql_engine)
            self.sql_engine.dispose()
            del self.data
            del self.sql_engine




