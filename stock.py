import os
import importlib.util

import pandas as pd
import numpy as np
import os
from   sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
'''
-------------------------------------------------------------------------------
'''

# from config import BASE_DIR
# Define the path to config.py
# config_path = os.path.abspath(os.path.join(os.path.dirname(r'C:\Users\nilay\OneDrive - Cal State Fullerton (1)\Desktop\NILAY-TO-JOB-DATA\SPRING 2024\CPSC 597 Project\Stock-Price-Prediction'), 'Stock-Price-Prediction', 'config.py'))


# config_path = os.path.abspath(os.path.join(os.path.dirname(r'/Users/pmarathay/code/Stock-Price-Prediction/config.py')))
# print("Path: ", config_path)
# print("In config.py: ", BASE_DIR)

# # Load the module from the specified path
# spec = importlib.util.spec_from_file_location("config", config_path)
# print(spec)
# config = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(config)

print("In Stock Prediction File")

BASE_DIR = r'C:\Users\nilay\OneDrive - Cal State Fullerton (1)\Desktop\NILAY-TO-JOB-DATA\SPRING 2024\CPSC 597 Project\Stock-Price-Prediction'

print("Base_DIR: ", BASE_DIR)

file_path = os.path.join(BASE_DIR, 'dataset', 'preprocessed_data', 'AAPL.csv')
df = pd.read_csv(file_path)
df = df[['Date', 'Open', 'High', 'Low', 'Close']]

# Convert date from string to datetime
import datetime

def str_to_datetime(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)

df['Date'] = df['Date'].apply(str_to_datetime)
print(df['Date'])
df.index = df.pop('Date')
df.dropna(axis=0, inplace=True)


# This Scales the Data
# Initialize scalers for each column
scalers = {
    'Open': MinMaxScaler(),
    'High': MinMaxScaler(),
    'Low': MinMaxScaler(),
    'Close': MinMaxScaler()
}

print("Before for loop")
# Scale each column separately
for column in ['Open', 'High', 'Low', 'Close']:
    df[column] = scalers[column].fit_transform(df[[column]])
    print(df[column])

'''
-------------------------------------------------------------------------------
'''

# Function to create a windowed dataframe
def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    first_date = str_to_datetime(first_date_str)
    last_date  = str_to_datetime(last_date_str)

    target_date = first_date

    dates = []
    X_open, X_high, X_low, X_close, Y_open, Y_high, Y_low, Y_close = [], [], [], [], [], [], [], []

    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n+1)

        if len(df_subset) != n+1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return

        open_values = df_subset['Open'].to_numpy()
        high_values = df_subset['High'].to_numpy()
        low_values = df_subset['Low'].to_numpy()
        close_values = df_subset['Close'].to_numpy()

        X_open.append(open_values[:-1])
        Y_open.append(open_values[-1])
        
        X_high.append(high_values[:-1])
        Y_high.append(high_values[-1])
        
        X_low.append(low_values[:-1])
        Y_low.append(low_values[-1])
        
        X_close.append(close_values[:-1])
        Y_close.append(close_values[-1])

        dates.append(target_date)

        next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year, month, day = map(int, next_date_str.split('-'))
        next_date = datetime.datetime(year=year, month=month, day=day)

        if last_time:
            break

        target_date = next_date

        if target_date == last_date:
            last_time = True

    data = {
        'Target Date': dates
    }

    X_open = np.array(X_open)
    X_high = np.array(X_high)
    X_low = np.array(X_low)
    X_close = np.array(X_close)

    for i in range(n):
        data[f'Open-{n-i}'] = X_open[:, i]
        data[f'High-{n-i}'] = X_high[:, i]
        data[f'Low-{n-i}'] = X_low[:, i]
        data[f'Close-{n-i}'] = X_close[:, i]

    data['Open-Target'] = Y_open
    data['High-Target'] = Y_high
    data['Low-Target'] = Y_low
    data['Close-Target'] = Y_close

    ret_df = pd.concat({key: pd.Series(value) for key, value in data.items()}, axis=1)

    return ret_df

# Window settings:  Date range and n = number of past dates used to predict next day
windowed_df = df_to_windowed_df(df, 
                                '2016-06-10', 
                                '2024-06-27',
                                n=5)
print(windowed_df)

'''
-------------------------------------------------------------------------------
'''

#  Separating Window DF into 2 parts:  Past Dates (Training and Validation) and Prediction Dates
def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:, 0]


    #  Setting the Past Dates (Training and Validation) into X
    # Extract the features (Open, High, Low, Close) from the columns
    middle_matrix = df_as_np[:, 1:-4]
    # Reshape the features matrix to include the multiple feature sets
    # Each row has n Open, High, Low, and Close values
    num_features = 4  # We have Open, High, Low, Close
    n = (middle_matrix.shape[1] // num_features)  # Number of time steps (n=3 in this case)
    X = middle_matrix.reshape((len(dates), n, num_features))

    # Extract the target values (Open-Target, High-Target, Low-Target, Close-Target)
    Y_open = df_as_np[:, -4]
    Y_high = df_as_np[:, -3]
    Y_low = df_as_np[:, -2]
    Y_close = df_as_np[:, -1]


    #  Setting thePrediction Dates (target value) into y
    # Combine all target values into a single array
    Y = np.stack((Y_open, Y_high, Y_low, Y_close), axis=-1)

    return dates, X.astype(np.float32), Y.astype(np.float32)

dates, X, y = windowed_df_to_date_X_y(windowed_df)

# dates.shape, X.shape, y.shape
print("Dates: ", dates.shape)
print("Dates: ", dates)
print("Specific Date: ", dates[2024])

# X includes the n day window OHLC that we use to predict the next days OHLC
print("Shape of Training Data (shows n): ", X.shape)
print("Training Data (shows n): ", X)

# y represents the target value for the next days OHLC
print("Shape of Target Data: ", y.shape)
print("Prediction Data (actual value): ", y)

'''
-------------------------------------------------------------------------------
'''

q_80 = int(len(dates) * .80)  # Training Data 80%
q_90 = int(len(dates) * .90)  # Validation Data 10%

# Splitting the data
dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

'''
-------------------------------------------------------------------------------
'''
# Plot Chart
plt.figure(figsize=(20, 10))

# Plotting target values for each feature set
plt.plot(dates_train, y_train[:, 0], label='Train - Open', color='blue')
plt.plot(dates_val, y_val[:, 0], label='Validation - Open', color='orange')
plt.plot(dates_test, y_test[:, 0], label='Test - Open', color='green')

plt.legend()
plt.title('Target Values Over Time')
plt.xlabel('Date')
plt.ylabel('Value')

plt.show()
'''
-------------------------------------------------------------------------------
'''
# Model Architecture
# Define the model
model = Sequential([
    layers.Input((3, 4)),                       # Input Layer
    layers.LSTM(256),                           # Long Short Term Memory.  Neural Network with data storage.  Used vs GRU Gated Recurrent Unit
    layers.Dense(64, activation='relu'),        # Number of mathematical functions used to caluculate next dense layer
    layers.Dense(32, activation='relu'),        # Next Dense Layer
    layers.Dense(4)                             # Output Value
])

# Compile the model
model.compile(loss='mse', 
              optimizer=Adam(learning_rate=0.0001),
              metrics=['mean_absolute_error'])

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=5, 
                               restore_best_weights=True)


# Define the print callback
print_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(f"Epoch: {epoch+1}, Loss: {logs['loss']}, MAE: {logs['mean_absolute_error']}, Val Loss: {logs['val_loss']}, Val MAE: {logs['val_mean_absolute_error']}"))

# Train the model with both callbacks
model.fit(X_train, y_train, 
          validation_data=(X_val, y_val), 
          epochs=100, 
          callbacks=[early_stopping, print_callback])
'''
-------------------------------------------------------------------------------
'''
# # # Print statement after each
# # class TrainingCallback(Callback):
# #     def on_epoch_end(self, epoch, logs=None):
# #         print(f"Epoch {epoch + 1}: loss = {logs['loss']}, accuracy = {logs['accuracy']}")

# print_status = print("hello world")

# # Train the model with early stopping
# model_2.fit(X_train2, y_train2, 
#           validation_data=(X_val2, y_val2), 
#           epochs=100, 
#           callbacks=[early_stopping, print_status])

# Now the Model is Trained

predictions = model.predict(X_test)
# predictions

print("Prediction Dates: ", dates[q_90:])
print("Predictions: ", predictions)

'''
-------------------------------------------------------------------------------
'''

# Inverse scaling the predictions
inv_predictions = {
    'Open': scalers['Open'].inverse_transform(predictions[:, 0].reshape(-1, 1)).flatten(),
    'High': scalers['High'].inverse_transform(predictions[:, 1].reshape(-1, 1)).flatten(),
    'Low': scalers['Low'].inverse_transform(predictions[:, 2].reshape(-1, 1)).flatten(),
    'Close': scalers['Close'].inverse_transform(predictions[:, 3].reshape(-1, 1)).flatten()
}

# Inverse scaling the actual values
inv_y_test = {
    'Open': scalers['Open'].inverse_transform(y_test[:, 0].reshape(-1, 1)).flatten(),
    'High': scalers['High'].inverse_transform(y_test[:, 1].reshape(-1, 1)).flatten(),
    'Low': scalers['Low'].inverse_transform(y_test[:, 2].reshape(-1, 1)).flatten(),
    'Close': scalers['Close'].inverse_transform(y_test[:, 3].reshape(-1, 1)).flatten()
}

# Filter the dates and values for the required date range
start_date = datetime.datetime(2024, 6, 3)
end_date = datetime.datetime(2024, 6, 27)
mask = (dates_test >= start_date) & (dates_test <= end_date)

filtered_dates = dates_test[mask]
filtered_inv_y_test = {key: value[mask] for key, value in inv_y_test.items()}
filtered_inv_predictions = {key: value[mask] for key, value in inv_predictions.items()}


# Plotting
plt.figure(figsize=(20, 10))

# plt.plot(filtered_dates, filtered_inv_y_test['Open'], label='Actual Open', color='blue')
# plt.plot(filtered_dates, filtered_inv_y_test['High'], label='Actual High', color='green')
# plt.plot(filtered_dates, filtered_inv_y_test['Low'], label='Actual Low', color='red')
plt.plot(filtered_dates, filtered_inv_y_test['Close'], label='Actual Close', color='orange')

# plt.plot(filtered_dates, filtered_inv_predictions['Open'], label='Predicted Open', linestyle='--', color='blue')
# plt.plot(filtered_dates, filtered_inv_predictions['High'], label='Predicted High', linestyle='--', color='green')
# plt.plot(filtered_dates, filtered_inv_predictions['Low'], label='Predicted Low', linestyle='--', color='red')
plt.plot(filtered_dates, filtered_inv_predictions['Close'], label='Predicted Close', linestyle='--', color='orange')

plt.legend()
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')

# Adding each date as a tick on the x-axis
plt.xticks(filtered_dates, rotation=45)
plt.show()

'''
-------------------------------------------------------------------------------
'''


import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

# Function to predict the values for the next X business days
def predict_next_X_days(model, last_input, start_date):
    predicted_dates = []
    predicted_values = []

    current_input = last_input

    for _ in range(9):  # Predicting the next X business days
        # Predict the next day
        prediction = model.predict(current_input)
        
        # Store the prediction
        predicted_values.append(prediction.flatten())

        # Prepare the next input
        next_input = current_input[0, 1:, :].tolist()  # Drop the first day
        next_input.append(prediction.flatten().tolist())  # Add the new prediction
        current_input = np.array([next_input])

        # Get the next business day
        last_date = start_date if len(predicted_dates) == 0 else predicted_dates[-1]
        next_date = pd.Timestamp(last_date) + pd.offsets.BDay()  # Use BDay to skip weekends
        predicted_dates.append(next_date)

    # Convert predicted values to numpy array for inverse scaling
    predicted_values = np.array(predicted_values)

    return predicted_dates, predicted_values

# Assuming X_test and scalers are already defined as in the previous code
last_input = X_test[-1:]  # Example input data
start_date = pd.Timestamp('2024-06-27')  # Start date for predictions

# Predict the next 30 business days
predicted_dates, predicted_values = predict_next_X_days(model, last_input, start_date)

# Inverse scaling the predictions (assuming scalers and model_2 are defined appropriately)
inv_predictions = {
    'Open': scalers['Open'].inverse_transform(predicted_values[:, 0].reshape(-1, 1)).flatten(),
    'High': scalers['High'].inverse_transform(predicted_values[:, 1].reshape(-1, 1)).flatten(),
    'Low': scalers['Low'].inverse_transform(predicted_values[:, 2].reshape(-1, 1)).flatten(),
    'Close': scalers['Close'].inverse_transform(predicted_values[:, 3].reshape(-1, 1)).flatten()
}

# Convert predictions to DataFrame for better visualization
predicted_df = pd.DataFrame(inv_predictions, index=predicted_dates)

# Plotting the predictions
plt.figure(figsize=(20, 10))

# Plotting 'Close' prices
plt.plot(predicted_df.index, predicted_df['Open'], label='Predicted Open (Next X Business Days)', linestyle='--', color='blue')
plt.plot(predicted_df.index, predicted_df['High'], label='Predicted High (Next X Business Days)', linestyle='--', color='green')
plt.plot(predicted_df.index, predicted_df['Low'], label='Predicted Low (Next X Business Days)', linestyle='--', color='red')
plt.plot(predicted_df.index, predicted_df['Close'], label='Predicted Close (Next X Business Days)', linestyle='--', color='orange')

plt.legend()
plt.title('Predicted Stock Prices for the Next X Business Days')
plt.xlabel('Date')
plt.ylabel('Price')

plt.xticks(predicted_df.index, rotation=45)
plt.show()
