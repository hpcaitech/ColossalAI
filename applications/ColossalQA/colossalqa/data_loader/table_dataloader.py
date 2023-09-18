'''
Class for loading table type data. please refer to Pandas-Input/Output for file format details.
'''


import os
import glob
import pandas as pd
from sqlalchemy import create_engine
from colossalqa.utils import drop_table
from colossalqa.mylogging import get_logger

logger = get_logger()

SUPPORTED_DATA_FORMAT = ['.csv','.xlsx', '.xls','.json','.html','.h5', '.hdf5','.parquet','.feather','.dta']

class TableLoader:
    '''
    Load tables from different files and serve a sql database for database operations
    '''
    def __init__(self, files: str, 
                 sql_path:str='sqlite:///mydatabase.db', 
                 verbose=False, **kwargs) -> None:
        '''
        Args:
            files: list of files (list[file path, name])
            sql_path: how to serve the sql database
            **kwargs: keyword type arguments, useful for certain document types 
        '''
        self.data = {}
        self.verbose = verbose
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
            
            logger.info("loading data", verbose=self.verbose)
            self.load_data(path)
            logger.info("data loaded", verbose=self.verbose)
            self.to_sql(path, dataset_name)

    def load_data(self, path):
        '''
        Load data and serve the data as sql database.
        Data must be in pandas format
        '''
        files = []
        # Handle glob expression
        try:
            files = glob.glob(path)
        except Exception as e:
            logger.error(e)
        if len(files)==0:
            raise ValueError("Unsupported file/directory format. For directories, please use glob expression")
        elif len(files)==1:
            path = files[0]
        else:
            for file in files:
                self.load_data(file)

        if path.endswith('.csv'):
            # Load csv
            self.data[path] = pd.read_csv(path)
        elif path.endswith('.xlsx') or path.endswith('.xls'):
            # Load excel
            self.data[path] = pd.read_excel(path)  # You can adjust the sheet_name as needed
        elif path.endswith('.json'):
            # Load json
            self.data[path] = pd.read_json(path)
        elif path.endswith('.html'):
            # Load html
            html_tables = pd.read_html(path)
            # Choose the desired table from the list of DataFrame objects
            self.data[path] = html_tables[0]  # You may need to adjust this index
        elif path.endswith('.h5') or path.endswith('.hdf5'):
            # Load h5
            self.data[path] = pd.read_hdf(path, key=self.kwargs.get('key', 'data'))  # You can adjust the key as needed
        elif path.endswith('.parquet'):
            # Load parquet
            self.data[path] = pd.read_parquet(path, engine='fastparquet')
        elif path.endswith('.feather'):
            # Load feather
            self.data[path] = pd.read_feather(path)
        elif path.endswith('.dta'):
            # Load dta
            self.data[path] = pd.read_stata(path)
        else:
            raise ValueError("Unsupported file format")
        
    def to_sql(self, path, table_name):
        '''
        Serve the data as sql database.
        '''
        self.data[path].to_sql(table_name, con=self.sql_engine, if_exists='replace', index=False)
        logger.info(f"Loaded to Sqlite3\nPath: {path}", verbose=self.verbose)
        return self.sql_path
    
    def get_sql_path(self):
        return self.sql_path

    def __del__(self):
        if self.sql_engine:
            drop_table(self.sql_engine)
            self.sql_engine.dispose()
            del self.data
            del self.sql_engine




