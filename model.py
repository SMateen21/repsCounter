import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('landmarks.csv', delimiter=',')

y_train = df.status

x_train = df.drop(columns=['status'])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=63, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=24, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=15)


model.save('repsCounter.keras')
