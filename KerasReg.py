# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 23:25:18 2023

@author: Kostas
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# Prepare data for LSTM

def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=28, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Load pre-processed data
df_weka = pd.read_csv('cryptocurrency.csv')

# Visualize the current BTC Price
plt.figure(figsize=(18, 9))
plt.plot(range(df_weka.shape[0]), df_weka['BTC_Price'])
plt.xticks(range(0, df_weka.shape[0], 100), df_weka['Date'].loc[::100], rotation=45)
plt.xlabel('Date', fontsize=18)
plt.ylabel('BTC Price', fontsize=18)
plt.show()

# Normalize data and Scaling all features except Date
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_weka.drop(columns=['Date']))  

# Convert the scaled data back to a DataFrame
df_scaled = pd.DataFrame(scaled_data, columns=df_weka.drop(columns=['Date']).columns)

features = df_scaled.drop(columns=['BTC_Price']).values  # Use all features except Date
target = df_scaled['BTC_Price'].values  # Target variable (BTC_Price)

# # Split the data into training and testing sets
# # Split x to y for train and test (shift y by one day) predict next day's price
X_train = features[:1000,:]
X_test = features[1000:-1,:]
y_train = target[1:1001]
y_test = target[1001:]

#  OR

# Split the data into training and testing sets
# Split x to y for train and test (do not shift y by one day, i.e. predict same day price)
# X_train = features[:1000,:]
# X_test = features[1000:,:]
# y_train = target[:1000]
# y_test = target[1000:]



# Build LSTM model
# Model setup
model = KerasRegressor(build_fn=create_model, verbose=0)

# Define hyperparameters for tuning
param_grid = {'epochs': [10, 50, 100],
              'batch_size': [10, 20, 40, 60, 80, 100]}

# GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)


# Display GridSearchCV results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("Mean: %f, Std: %f with: %r" % (mean, stdev, param))

   
# Access the best model from the grid search
best_model = grid_result.best_estimator_

# Make predictions using the best model
predicted_prices = best_model.predict(X_test)

results_df = pd.DataFrame({'Actual_Prices': y_test, 'Predicted_Prices': predicted_prices})
results_df.to_csv('btc_predictions.csv', index=False)


# Plot actual vs predicted BTC prices
plt.figure(figsize=(18, 9))
df_last = df_weka.tail(len(y_test))['Date']
plt.plot(df_last, y_test, label='Actual BTC Prices')
plt.plot(predicted_prices, label='Predicted BTC Prices (using all features)')
plt.gcf().autofmt_xdate()
locs, labels = plt.xticks()
locs = locs[::10]
labels = labels[::10]
plt.xticks(locs, labels)
plt.legend()
plt.xlabel('Time')
plt.ylabel('BTC Price')
plt.title('Actual vs Predicted BTC Prices')
plt.show()

# save predicted price and actual price in csv, so we could visualize them in Tableau
# df = pd.DataFrame({'Date': df_last, 'Actual_Prices': y_test, 'Predicted_Prices': predicted_prices})
# df.to_csv('btc_predicted.csv', index=False)

# Add more plots using your function plot_loss or other visualization techniques
# For example, visualize a scatter plot between actual and predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predicted_prices, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Scatter plot of Actual vs Predicted BTC Prices (using all features)')
plt.show()

# You can create additional plots or visualizations to further analyze the results
# For instance, visualize the error distribution
error = y_test - predicted_prices
plt.figure(figsize=(10, 6))
plt.hist(error, bins=20)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Histogram of Prediction Errors (using all features)')
plt.show()
    
    