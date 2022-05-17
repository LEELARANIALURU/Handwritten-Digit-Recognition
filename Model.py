from tensorflow import keras
import numpy as np

#loading dataset
(x_train, y_train) , (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train/255
x_test = x_test/255

#building the model
model = keras.Sequential([
              keras.layers.Flatten(input_shape = (28,28)),
              keras.layers.Dense(100, activation='relu'),
              keras.layers.Dense(50, activation='relu'),
              keras.layers.Dense(20, activation='relu'),
              keras.layers.Dense(15, activation='relu'),
              keras.layers.Dense(10, activation = 'softmax')
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

#fitting data
model.fit(x_train, y_train, epochs = 20)

#evaluation
model.evaluate(x_test, y_test)

#making predictions
y_predicted = model.predict(x_test)

model.save("hand_writ_digi.h5")

#testing on custom data
import cv2

img = cv2.imread(r"C:\Users\HP\Desktop\test_img8.jpg")

img = cv2.resize(img,(28,28))

img = np.array(img)

img = img[:,:,0]

img = img.reshape(1,28,28,1)
img = img/255.0
res = model.predict([img])[0]
print(np.argmax(res))

#reading out the output
import pyttsx3
pyobj = pyttsx3.init()
pyobj.say(np.argmax(res))
pyobj.runAndWait()
