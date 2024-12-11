from tqdm import tqdm
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Dropout, RepeatVector, TimeDistributed
from keras.losses import CategoricalCrossentropy

# Sometimes the importing is slow, so this is just a message to confirm the completion of it.
print("")
print("Packages imported.")
print("")

# DATA PROCESSING

# Setting the random seed with proof in both tensorflow and numpy
seed = 0
tf.random.set_seed(seed)
np.random.seed(seed)
print("tensorflow:",tf.random.uniform([1]))
print("numpy:",np.random.rand(1))
print("")

# Setting the scaling method to be used later
scaler = MinMaxScaler()

# Function to split and scale the data, changing its form from a DataFrame to a Numpy ndarray
def data_split(data):
    x = data.iloc[:, :-1]
    x_scaled_data = scaler.fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled_data, columns=x.columns).to_numpy()
    y = data.iloc[:, -1].to_numpy()
    return (x_scaled, y)

# Reading the Car Hacking Dataset and storing it as a DataFrame
print("Reading Dataset...")
df = pd.read_csv("car_hacking_dataset.csv")

# Creating a normal data only dataset for training an unsupervised LSTM autoencoder
df = df[df.iloc[:, -1]<1]
print("")

# Shuffling the Dataset using the .sample() method
print("Full dataset (normal data only):",df.shape)
print("")
df = df.sample(frac=1)

# Setting up for LSTM
time_steps = 1

# Splitting the Dataset into 3 ratios, currently set to 70%, 5% and 25% for training, validating and testing data respectively
train_data = df.iloc[:int(0.7*len(df.index)), :]
val_data = df.iloc[int(0.7*len(df.index)):int((0.75)*len(df.index)), :]
test_data = df.iloc[int((0.75)*len(df.index)):, :]

# Splitting the subsets into x and y, scaling the independent variables (x) using the MinMaxScaler() method
train_x, train_y = data_split(train_data)
train_x = train_x.reshape((len(train_x), 10, 1))

val_x, val_y = data_split(val_data)
val_x = val_x.reshape((len(val_x), 10, 1))

test_x, test_y = data_split(test_data)
test_x = test_x.reshape((len(test_x), 10, 1))

print("Training data:",train_x.shape,train_y.shape)
print("Validation data:",val_x.shape,val_y.shape)
print("Testing data:",test_x.shape,test_y.shape)
print("")

# Defining LSTM autoencoder
lstm = Sequential()
lstm.add(Input(shape=(len(train_x),10)))
lstm.add(LSTM(128, activation="relu", return_sequences=True))
lstm.add(Dropout(0.2))
lstm.add(LSTM(64, activation="relu", return_sequences=False))
lstm.add(Dropout(0.2))
lstm.add(RepeatVector(len(train_x)))

lstm.add(LSTM(64, activation="relu", return_sequences=True))
lstm.add(Dropout(0.2))
lstm.add(LSTM(128, activation="relu", return_sequences=True))
lstm.add(Dropout(0.2))
lstm.add(TimeDistributed(Dense(10)))
lstm.summary()

# Compile and train model
lstm.compile(optimizer="adam", loss="mse")
model_hist = lstm.fit(train_x, train_x, batch_size=64, epochs=100, verbose=0, validation_data=(val_x,val_x))
score,acc = lstm.evaluate(train_x,train_x, batch_size=64)
print("Training score:",score)
print("Training accuracy:",acc)
