#importing libraries
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#loading dataset
(x_train, y_train) , (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train/255
x_test = x_test/255

#building the model
model = keras.Sequential([
              keras.layers.Flatten(input_shape = (28,28)),
              keras.layers.Dense(100, activation='relu'),
              keras.layers.Dense(10, activation = 'softmax')
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

#fitting data
model.fit(x_train, y_train, epochs = 10)

#evaluation
model.evaluate(x_test, y_test)

#making predictions
y_predicted = model.predict(x_test)
