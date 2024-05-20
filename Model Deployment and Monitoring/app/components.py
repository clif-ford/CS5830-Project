from PIL import Image
from io import BytesIO
import tensorflow as tf
import numpy as np
import pickle

def load_model(path:str) -> tf.keras.Sequential:
    """
    Function to load saved keras model

    Args:
        path (str): path to the keras model including the file name and extension (.keras)

    Returns:
        tf.keras.Sequential: The loaded model
    """
    model = tf.keras.models.load_model(path)
    return model


def predict_digit(model:tf.keras.Sequential, data_point:np.array) -> str:
    """
    Takes a model and an input sample and returns the model's prediction.

    Args:
        model (tf.keras.Sequential): The model being used to classify MNIST
        data_point (np.array): The input the model must evaluate

    Returns:
        str: A single digit representing the model's prediction
    """

    probs = model.predict(data_point)
    return probs


def read_imagefile(file:str) -> Image.Image:
    """
    Function to read an image file such as jpg, jpeg or png 

    Args:
        file (str): path to the image including the file name and extension

    Returns:
        Image.Image: a python PIL image object
    """
    image = Image.open(BytesIO(file))
    return image



def format_image(image:Image.Image) -> np.array:
    """
    Formats the image into the required 28x28 grayscale format required by the MNIST classifier and returns the formatted image.

    Args:
        image (Image.Image): The image to be formatted

    Returns:
        np.array: a 1x784 numpy array corresponding to the intensity values on a scale of 0 to 1.
    """
    img_size = image.height * image.width
    image = image.resize((28,28)).convert('L')
    arr = np.array(image)
    arr = arr/255
    arr = arr.reshape(-1, 28, 28, 1)
    return arr, img_size