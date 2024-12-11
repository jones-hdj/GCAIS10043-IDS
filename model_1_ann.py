from tqdm import tqdm
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Input
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
print("")

# Shuffling the Dataset using the .sample() method
print("Full dataset:",df.shape)
print("")
df = df.sample(frac=1)

# Splitting the Dataset into 3 ratios, currently set to 70%, 5% and 25% for training, validating and testing data respectively
train_data = df.iloc[:int(0.7*len(df.index)), :]
val_data = df.iloc[int(0.7*len(df.index)):int((0.75)*len(df.index)), :]
test_data = df.iloc[int((0.75)*len(df.index)):, :]

# Splitting the subsets into x and y, scaling the independent variables (x) using the MinMaxScaler() method
train_x, train_y = data_split(train_data)
val_x, val_y = data_split(val_data)
test_x, test_y = data_split(test_data)
print("Training data:",train_x.shape,train_y.shape)
print("Validation data:",val_x.shape,val_y.shape)
print("Testing data:",test_x.shape,test_y.shape)
print("")

# MODEL

# notes here
classifier = Sequential()
classifier.add(Input(shape=(10,)))
classifier.add(Dense(units=16, kernel_initializer="uniform", activation="relu"))
classifier.add(Dense(units=16, kernel_initializer="uniform", activation="relu"))
classifier.add(Dense(units=5, kernel_initializer="uniform", activation="softmax"))
classifier.summary()
classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model_hist = classifier.fit(train_x, train_y, batch_size=256, epochs=10, verbose=0, validation_data=(val_x,val_y))
score,acc = classifier.evaluate(train_x,train_y, batch_size=256)
print("Training score:",score)
print("Training accuracy:",acc)

test_yh = classifier.predict(test_x, batch_size=256)
test_loss = tf.keras.losses.SparseCategoricalCrossentropy(test_y, test_yh)
test_accuracy = accuracy_score(test_y, np.argmax(test_yh, axis=-1)) # fix this
test_cfm = confusion_matrix(test_y, np.argmax(test_yh,axis=-1), labels=range(5))

print("Test loss:",test_loss)
print("Test accuracy:",test_accuracy)
print(test_cfm)
