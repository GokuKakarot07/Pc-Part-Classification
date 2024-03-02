import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model('image_classification_model.h5')  

img_size = (224, 224)

class_names = {
    0: 'Cable',
    1: 'Case',
    2: 'CPU',
    3: 'GPU',
    4: 'Hardisk',
    5: 'Headset',
    6: 'Keyboard',
    7: 'Microphone',
    8: 'Monitor',
    9: 'Motherboard',
    10: 'Mouse',
    11: 'Ram',
    12: 'Speaker',
    13: 'Webcam',
}

st.title('Image Classification App')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.resize(img_size)  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.image(img, caption=f'Uploaded Image', use_column_width=True)

    if predicted_class in class_names:
        predicted_category = class_names[predicted_class]
        st.write(f'Prediction: It is {predicted_category}')
    else:
        st.write(f'Prediction: Class {predicted_class}')
