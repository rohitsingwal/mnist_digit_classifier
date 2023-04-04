import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#What are the steps to be taken for model building?
#0. Importing data
train_data = pd.read_csv('/Users/neelumeena/Documents/Machine Learning/20230404_MNIST_digit_classifier/data/train.csv.zip',compression='zip')
test_data = pd.read_csv('/Users/neelumeena/Documents/Machine Learning/20230404_MNIST_digit_classifier/data/test.csv.zip',compression='zip')
X = train_data.loc[:,train_data.columns != 'label']
y = train_data.label

#1. Split train CV
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.33, random_state=23)

#2. Build a neural network for classification
model = Sequential([
    Dense(units = 128, activation = 'relu'),
    Dense(units = 10, activation = 'linear')
])
model.compile(Adam(learning_rate = 0.001),loss = SparseCategoricalCrossentropy(from_logits = True))
model.fit(X_train, y_train, epochs = 10)

#yhat_train = tf.nn.softmax(model.predict(X_train))
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

yhat_train = probability_model.predict(X_train)
yhat_train[0]
np.argmax(yhat_train[0])
y_train[0]
accuracy_score(y_train,np.argmax(yhat_train,axis=1))

#3. Fine-tune the model based on CV performance

#4. Predict the answer for test dataset

#5. Generate reports for the model output
