import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Loading Data

df = tf.keras.datasets.mnist.load_data()

(X_train,y_train),(X_test,y_test)=df

# Normalization

X_train=(X_train.astype('float32').reshape(X_train.shape[0],28,28,1))/255.0
X_test=(X_test.astype('float32').reshape(X_test.shape[0],28,28,1))/255.0


# Building the model

from tensorflow.keras.layers import BatchNormalization
model=tf.keras.models.Sequential([
  
  # Layer1
  tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
  tf.keras.layers.MaxPooling2D(2,2),
  BatchNormalization(axis=-1),
  
  # Layer2
  tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
  tf.keras.layers.MaxPooling2D(2,2),
  BatchNormalization(axis=-1),
  
  #Flattening
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Flatten(),
  
  #Dense Layer
  tf.keras.layers.Dense(128,activation='relu'),
  BatchNormalization(axis=-1),
  
  #Output Layer
  tf.keras.layers.Dense(10,activation='softmax')
])

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics= ['accuracy'])

#Fitting Data into the model
h = model.fit (X_train, y_train , epochs= 10,validation_split=0.2)

#saving the model
model.save("handwritten_digit_recognition_model.h5")

#Training Accuracy
loss,acc=model.evaluate(X_train,y_train, verbose =0)
print("The train accuracy of model is :-",acc*100)

#Testing Accuracy
test_loss,test_acc=model.evaluate(X_test,y_test,verbose =0)
print("The test accuracy of model is :-",test_acc*100)

# Loading saved model
#model = tf.keras.models.load_model(r'path\handwritten_digit_recognition_model.h5')

# Making prediction
import cv2
import numpy as np

img = cv2.imread(r"path\test_img12.jpg")
img = cv2.resize(img,(28,28))
img = np.array(img)
img = img[:,:,0]
img = img.reshape(1,28,28,1)
img = img/255.0
res = model.predict([img])[0]
print(np.argmax(res))

# Reading out the output
import pyttsx3
pyobj = pyttsx3.init()
pyobj.say(np.argmax(res))
pyobj.runAndWait()

# Plotting Accuracy and Loss
df0=pd.DataFrame(x.history)
plt.figure(figsize = (7,7))
sns.lineplot(data=df0[['accuracy','val_accuracy']],palette=['r', 'g'],linewidth=4)
plt.show()

plt.figure(figsize = (7,7))
sns.lineplot(data=df0[['loss','val_loss']],palette=['b', 'k'],linewidth=4)
plt.show()
