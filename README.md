# Handwritten-Digit-Recognition

## Project

This project aims to recognise handwritten digits and read them loud. It uses the image of a digit written by the user, recognizes the digit and reads it loud.
To build the model, we use **Convolutional Neural Network**.

## Dataset

The Dataset used is the **MNIST Dataset**. It can be imported using:
```
import tensorflow
tensorflow.keras.datasets.mnist.load_data()
```
This contains 70,000 images of handwritten digits 0-9. It has 60,000 images in training set and 10,000 images in testing set. The images in the dataset look like:

![Images from Dataset](https://user-images.githubusercontent.com/76864260/173120940-df9067dc-d391-4c22-aaad-c39654a93a18.png)


## Libraries used
- TensorFlow
- NumPy
- CV2
- pyttsx3
- matplotlib
- seaborn

## Results

The accuracy obtained is:
- On training set: 99.75%
- On testing set: 99.22%
### Model Accuracy
![Accuracy Plot](https://user-images.githubusercontent.com/76864260/173128486-071648b8-9380-4e5e-abf9-404914338579.png)

### Model Loss
![Loss Plot](https://user-images.githubusercontent.com/76864260/173129128-3ecaf43a-9c08-4b90-83fc-a54b7224d5a7.png)



## Done by
[Leela Rani Aluru](https://github.com/LEELARANIALURU)
