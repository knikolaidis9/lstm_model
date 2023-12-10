# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 23:41:08 2023

@author: Kostas
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor



# function to print the number of lines for each dataset
def printLength(btc, usd, oil, gold, vix, sahm):
    print("BTC Data Length:", len(btc))
    print("USD Data Length:", len(usd))
    print("Oil Data Length:", len(oil))
    print("Gold Data Length:", len(gold))
    print("VIX Data Length:", len(vix))
    print("SAHM Data Length:", len(sahm))


# function to print the number of lines for each dataset
def printDatesNumber(btc, usd, oil, gold, vix, sahm):
    print("BTC Data Unique Dates:", btc['Date'].nunique())
    print("USD Data Unique Dates:", usd['Date'].nunique())
    print("Oil Data Unique Dates:", oil['Date'].nunique())
    print("Gold Data Unique Dates:", gold['Date'].nunique())
    print("VIX Data Unique Dates:", vix['Date'].nunique())
    print("SAHM Data Unique Dates:", sahm['Date'].nunique())

def regressor_selection(X, y, metric='r2'):
    pipe = Pipeline([('scaler', StandardScaler()), ('regressor', RandomForestRegressor())])
    param = [
        {
            'regressor': [HistGradientBoostingRegressor()],
            'regressor__max_iter': [100, 200, 500],  # Example hyperparameter
            'regressor__learning_rate': [0.01, 0.1, 0.5],  # Example hyperparameter
            # Add other hyperparameters specific to HistGradientBoostingRegressor
        },
        {
            'regressor': [KNeighborsRegressor()],
            'regressor__n_neighbors': [5, 10, 20, 30],
            'regressor__p': [1, 2]
        },
        {
            'regressor': [Lasso(max_iter=500)],
            'regressor__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
    ]

    clf = GridSearchCV(pipe, param_grid=param, cv=5, n_jobs=-1, scoring=metric)
    best_clf = clf.fit(X, y)
    return best_clf.best_estimator_

if __name__ == '__main__':
    btc_data = pd.read_csv('btc_data.csv')
    usd_data = pd.read_csv('usd.csv')
    oil_data = pd.read_csv('brent_oil.csv')
    gold_data = pd.read_csv('gold.csv')
    vix_data = pd.read_csv('vix.csv')
    sahm_data = pd.read_csv('sahmrealtime.csv')

    df_weka = btc_data.merge(usd_data, on='Date').merge(oil_data, on='Date').merge(gold_data, on='Date').merge(vix_data,
                                                                                                               on='Date')
    # df = pd.merge(df1, df2, on='date')
    # rename all column names
    #     df = df.rename(columns={
    #     'old_name1': 'new_name1',
    #     'old_name2': 'new_name2'
    # })

    # Drop a hole null column
    df_weka.drop(inplace=True, columns='VIX_Vol.')
    # print(df_weka.isnull().sum())
    # print(df_weka.count())

    # Drop all null cells
    df_weka.dropna(axis=0, inplace=True)

    # Date column does not needed so we have to drop it
    df_weka.drop(columns='Date', inplace=True)

    for column in df_weka.columns:
        if column != 'Date':
            df_weka[column] = df_weka[column].replace('[^\d.]', '', regex=True).astype(float)

        
    df_shifted = df_weka['BTC_Price'].shift(1)
    df_shifted = pd.concat([df_weka[2:], df_shifted], axis=1)
    # correlation between BTC_Price , feature selection
    correlation = df_shifted.corr()
    print(f"CORRELATION =====> {correlation}")
    # BTCPrice_corr = correlation[1][2:]

    plt.figure(figsize=(29, 29))
    sb.heatmap(correlation > 0.5, annot=True, cbar=False)
    plt.show()

    correlation = StandardScaler().fit_transform(correlation)

    # Save the new dataset to a CSV file
    # df_weka.to_csv('df_weka.csv', index=False)

    
   # Extracting the target variable and features for correlation
    target_index = 0  # Assuming the first column is the target (BTC_Price)
    target_corr = df_shifted.iloc[:, target_index].values.reshape(-1, 1)  # Target column as 2D array
    
    features_corr = df_shifted.iloc[:, 1:].values  # Features excluding the target
    
    # Calculating the correlation matrix (excluding the target column)
    correlation_matrix = np.corrcoef(features_corr.T)
    corr_with_target = np.abs(correlation_matrix[target_index])  # Correlation without the target column
    
    # Selecting top n features correlated with the target (BTC_Price)
    n = 5  # Adjust the number of features you want to select
    top_feature_indices = np.argsort(corr_with_target)[::-1][:n]  # Indices of top correlated features
    
    # Extracting the selected features
    X_selected_corr = features_corr[:, top_feature_indices]
    
    # Fit the selected model on the selected features
    selected_regressor_corr = regressor_selection(X_selected_corr, target_corr.flatten(), metric='r2')
    model_corr = selected_regressor_corr.fit(X_selected_corr, target_corr.flatten())
    print(f"Selected Model for Correlation: {model_corr.named_steps['regressor']}")
       
   # Visualization for Correlation Table
    # Scatter plot of actual vs. predicted values
    preds_corr = model_corr.predict(X_selected_corr)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(correlation['BTC_Price'], preds_corr, color='blue', alpha=0.5)
    plt.title('Actual vs Predicted Values (Correlation)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.show()
    
    # Residuals plot
    residuals_corr = correlation['BTC_Price'] - preds_corr
    plt.figure(figsize=(8, 6))
    plt.scatter(preds_corr, residuals_corr, color='green', alpha=0.5)
    plt.title('Residuals Plot (Correlation)')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.grid(True)
    plt.show()
    
    # Feature Importance for Correlation Table
    top_features_corr = df_shifted.columns[1:][top_feature_indices]  # Retrieve feature names
    feature_importance_corr = model_corr.named_steps['regressor'].feature_importances_
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features_corr)), feature_importance_corr, align='center')
    plt.yticks(range(len(top_features_corr)), top_features_corr)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance of Variables (Correlation)')
    plt.show()