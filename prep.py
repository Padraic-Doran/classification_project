#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, classification_report
from graphviz import Source
from scipy import stats


# In[2]:


def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'


# In[3]:


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


# In[5]:


df = get_telco_data_from_mysql()
new_df = df[['customer_id', 'churn', 'contract_type_id', 'contract_type', 'tenure', 'monthly_charges', 'total_charges', 'gender', 'partner', 'dependents', 'senior_citizen', 'phone_service', 'multiple_lines', 'internet_service_type', 'internet_service_type_id' ,'tech_support', 'streaming_tv', 'streaming_movies', 'online_security', 'online_backup', 'device_protection', 'payment_type', 'payment_type_id', 'paperless_billing']]


# In[7]:


def prep(df):
    df = get_telco_data_from_mysql()
    new_df = df[['customer_id', 'churn', 'contract_type_id', 'contract_type', 'tenure', 'monthly_charges', 'total_charges', 'gender', 'partner', 'dependents', 'senior_citizen', 'phone_service', 'multiple_lines', 'internet_service_type', 'internet_service_type_id' ,'tech_support', 'streaming_tv', 'streaming_movies', 'online_security', 'online_backup', 'device_protection', 'payment_type', 'payment_type_id', 'paperless_billing']]
    new_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df = new_df.dropna()
    df.total_charges = df.total_charges.astype(float, inplace = True)
    df.tech_support.replace('No internet service', 'No', inplace = True)
    df.replace('No internet service', 'No', inplace = True)
    df.replace('No phone service', 'No', inplace=True)
    return df


# In[8]:


def encode(df):
#     Family describes the family type based on partners and children

    conditions_1 =[
        (df['partner']=='Yes')& (df['dependents']=='Yes'),
        (df['partner']=='Yes')& (df['dependents']=='No'),
        (df['partner']=='No')& (df['dependents']=='Yes'),
        (df['partner']=='No')& (df['dependents']=='No')]
    choices_1 = [0,1,2,3]
    df['family'] = np.select(conditions_1, choices_1)

#Phone services describes whether someone has a phone plan and whether or not they have multiple lines

    conditions_2 =[
        (df['phone_service']=='Yes')& (df['multiple_lines']=='Yes'),
        (df['phone_service']=='Yes')& (df['multiple_lines']=='No'),
        (df['phone_service']=='No')& (df['multiple_lines']== 'No')]
    choices_2 = [0,1,2]
    df['phone_services'] = np.select(conditions_2, choices_2)

#Streaming services denotes what streaming services someone has.

    conditions_3 =[
        (df['streaming_tv']=='Yes')& (df['streaming_movies']=='Yes'),
        (df['streaming_tv']=='Yes')& (df['streaming_movies']=='No'),
        (df['streaming_tv']=='No')& (df['streaming_movies']=='Yes'),
        (df['streaming_tv']=='No')& (df['streaming_movies']=='No')]
    choices_3 = [0,1,2,3]
    df['streaming_services'] = np.select(conditions_3, choices_3)

#Online_services describes what types of online services someone has.

    conditions_4=[
        (df['online_security']=='Yes')& (df['online_backup']=='Yes'),
        (df['online_security']=='Yes')& (df['online_backup']=='No'),
        (df['online_security']=='No')& (df['online_backup']=='Yes'),
        (df['online_security']=='No')& (df['online_backup']=='No')]
    choices_4 = [0,1,2,3]
    df['online_services'] = np.select(conditions_4, choices_4)

#Tech_support as to whether someone has tech support

    df['e_tech_support'] = df['tech_support'].apply({"Yes":1,'No':0}.get)

# I'm confident you're getting the sentiment of these

    df['e_device_protection'] = df['device_protection'].apply({"Yes":1,'No':0}.get)

# Looking for eco allies with this function

    df['e_paperless_billing'] = df['paperless_billing'].apply({"Yes":1,'No':0}.get)

    df['e_churn'] = df['churn'].apply({"Yes":1,'No':0}.get)

    encoded_df = df[['customer_id', 'e_churn','e_gender' ,'contract_type_id', 'contract_type','tenure',
       'monthly_charges', 'total_charges','senior_citizen','internet_service_type_id','family', 'phone_services',
       'streaming_services', 'online_services', 'e_tech_support',
       'e_device_protection', 'e_paperless_billing', 'payment_type_id' ]]
    return encoded_df
# df.drop(columns= 'e_senior_citizen', inplace = True)


# In[13]:


def split_scale(df, test_size, random_state, stratify):
    train, test = train_test_split(encoded_df, test_size=.3, random_state=123, stratify=encoded_df.e_churn)
    X_train = train.drop(columns = ['customer_id', 'contract_type', 'e_churn'])
    X_test = test.drop(columns = ['customer_id', 'contract_type', 'e_churn'])
    y_train = train['e_churn']
    y_test = test['e_churn']
    numeric_X_train = X_train.drop(columns = ['e_gender', 'contract_type_id', 'streaming_services', 'online_services', 'e_tech_support', 'e_device_protection', 'e_paperless_billing','senior_citizen',
       'internet_service_type_id', 'family', 'phone_services', 'payment_type_id'])
    numeric_X_test = X_test.drop(columns = ['e_gender', 'contract_type_id', 'streaming_services', 'online_services', 'e_tech_support', 'e_device_protection', 'e_paperless_billing','senior_citizen',
       'internet_service_type_id', 'family', 'phone_services', 'payment_type_id'])
    return X_train, X_test, y_train, y_test, numeric_X_train, numeric_X_test


# In[ ]:




