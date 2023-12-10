# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 00:41:52 2023

@author: Kostas
"""

import numpy as np
import pandas as pd

btc_data = pd.read_csv('btc_data.csv')
usd_data = pd.read_csv('usd.csv')
oil_data = pd.read_csv('brent_oil.csv')
gold_data = pd.read_csv('gold.csv')
vix_data = pd.read_csv('vix.csv')
sahm_data = pd.read_csv('sahmrealtime.csv')

# Drop unexpected Dates
df_weka = btc_data.merge(usd_data, on='Date').merge(oil_data, on='Date').merge(gold_data, 
                                                                               on='Date').merge(vix_data, 
                                                                                                on='Date')


# Drop a hole null column
df_weka.drop(inplace=True, columns='VIX_Vol.')
   

# Drop all null cells
df_weka.dropna(axis=0, inplace=True)


# Reformat Dates
df = pd.DataFrame(columns=['Date'])
df_weka['Date'] = pd.to_datetime(df_weka['Date'], format='%m/%d/%y')

for column in df_weka.columns:
    if column != 'Date':
        df_weka[column] = df_weka[column].replace('[^\d.]', '', regex=True).astype(float)

# Sorting dataframe on Date 
df_weka = df_weka.sort_values('Date')

# Save to csv
df_weka.to_csv('cryptocurrency.csv', index=False)
