import numpy as np
# from flask import Flask, render_template, request
from keras.models import load_model
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

def predict(image1):
    model = load_model('models\chest_xray_resnet.h5')
    image = load_img(image1, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # Expand dimentions only for vgg
    # image = np.expand_dims(image, axis=0)
    # prepare the image for the VGG model only
    # image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # st.write(yhat)
    # convert the probabilities to class labels
    # label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = int(yhat[0][1])
    return label


import streamlit as st
from PIL import Image
import requests


def get_image(url):
    img = requests.get(url)
    file = open("sample_image.jpg", "wb")
    file.write(img.content)
    file.close()
    img_file_name = 'sample_image.jpg'
    return img_file_name


# Main driver
st.title("Peneumonia Detection")
st.write("Using Deep learning Model to classify the image")

url = st.text_input("Enter Image Url:")

if not url:
    st.write("Paste Image URL")
else:
    image = get_image(url)
    st.image(image)
    classify = st.button("classify image")
    if classify:
        st.write("")
        # st.write("Classifying...")
        label = predict(image)
        if label == 0:
            st.write("Person is Affected By PNEUMONIA")
        else:
            st.write("Person is Normal")
        # st.write(label)
