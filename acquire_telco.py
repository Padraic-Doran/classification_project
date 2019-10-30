
import env
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas_profiling
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from os import path
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import math
import split_scale


def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

    
def get_telco_data_from_mysql():
   
    # Use a double "%" in order to escape %'s default string formatting behavior.
    query = ''' Select * from customers
JOIN contract_types USING (contract_type_id)
Join internet_service_types USING (internet_service_type_id)
JOIN payment_types USING (payment_type_id);

    '''

    url = get_db_url("telco_churn") 
    df = pd.read_sql(query, url)
    return df


