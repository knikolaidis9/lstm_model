# LSTM Model for Time Series Analysis & BTC Price Prediction 📈🕒 #
This repository contains a Long Short-Term Memory (LSTM) model implemented in Python for time series analysis, specifically for predicting Bitcoin (BTC) prices. The IDE that I am using is Spyder. If you don't have Spyder installed you can download it directly from this website https://sourceforge.net/projects/winpython/. If you prefer other IDEs like Pycharm, VS Code etc. feel free to use them.

## Overview ℹ️ ##

The LSTM model is a type of recurrent neural network (RNN) that is well-suited for sequence prediction tasks like time series analysis. It can capture patterns and dependencies within time series data, making it particularly useful for forecasting future trends in cryptocurrency prices like Bitcoin. When we run the model we get 3 cvs files cryptocurrency, btc_predictions and btc_predicted. Cryptocurrency.csv contains all data cleaned and categorised, btc_predictions.csv contains our predictions from the model and btc_predicted.csv contains predictions with Date so that we will use it on visualization tools like Tableau/PowerBI. 

## How to Use 🚀 ##
## Dependencies 🛠️ ##
* Python (>=3.6)
* TensorFlow (>=2.0)
* Pandas
* NumPy
* Matplotlib

## Steps to Use the Model: ##
1. git https://github.com/knikolaidis9/lstm_model.git
   * cd lstm_model
2. Install Dependancies:
   * pip install tensorflow
   * pip unistall tensorflow
   * pip install tensorflow==2.12.0 if you have issues go directly with the version, but first unistall the tensorflow
   * pip install keras==2.12.0 you can also try this if you still have issues.
3. Prepare Data
   * Obtain historical BTC price data.
   * Preprocess the data (cleaning, normalization, splitting into training and testing sets) using the **prep_data.py** script.
4. Train the LSTM Model:
   * Run the training script **KerasReg.py**, providing the prepared dataset.
   * Tweak hyperparameters as needed for optimal performance.
5. Make Predictions:
   * Use the trained model to predict future BTC prices.
   * Evaluate the model's performance and visualize predictions against actual prices.
6. Fine-tuning and Improvement:
   * Experiment with different architectures, hyperparameters, or additional features to improve prediction accuracy.

## Why LSTM for BTC Price Prediction? 📊 ## 

* Sequence Learning: LSTM can capture long-term dependencies and patterns in time series data, which is crucial for predicting BTC prices affected by various market factors.
* Memory Retention: Its ability to retain information over extended periods enables better understanding of complex market behaviors.
* Prediction Accuracy: LSTM's architecture often outperforms traditional models for time series forecasting tasks.

## Contribution Guidelines 🤝 ##

Contributions to enhance the model's performance or expand its capabilities are welcome! Feel free to open issues, submit pull requests, or share suggestions.

