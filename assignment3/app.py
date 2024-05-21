import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st
from PIL import Image

def ImgClassification(image, model):
    image_load = tf.keras.utils.load_img(image, target_size=(150, 150))
    img_arr = tf.keras.utils.array_to_img(image_load)
    img_batch = tf.expand_dims(img_arr,0)
    image = Image.open(image)
    image = image.resize((200,200))

    predict = model.predict(img_batch)
    score = tf.nn.softmax(predict)
    category = ['boron-B', 'calcium-Ca', 'healthy', 'iron-Fe', 'magnesium-Mg', 'manganese-Mn', 'more-deficiencies', 'nitrogen-N', 'phosphorus-P', 'potasium-K']

    st.image(image)
    st.write('This is {} with accuracy of {:0.2f}'.format(category[np.argmax(score)], np.max(score)*100))

CNN = load_model(r"D:\uni\366\assignment3\CNN_CoLeaf.keras")
ResNet = load_model(r"D:\uni\366\assignment3\RN50_CoLeaf.keras")
Select = None

model_options = {'-Select-', 'CNN', 'ResNet'}



image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if image is not None:

    model = st.selectbox(
        "Select model to apply", 
        list(model_options)
        )
    
    if model is '-Select-':
        st.write("Please select an option.")
    elif model is 'CNN':
        st.write(f"You have selected {model}")
        ImgClassification(image, CNN)
    else:
        st.write(f"You have selected {model}")
        ImgClassification(image, ResNet)
        
    
