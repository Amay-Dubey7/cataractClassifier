import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load the trained model
# Load the cataract detection model. The model file should be in the same directory.
model = load_model('cataractEfficientNet.h5')

# Define the class labels
# 0 corresponds to 'Cataract' and 1 corresponds to 'Normal'
class_labels = {0: 'Cataract', 1: 'Normal'}

def prepare_image(img):
    """Preprocess the image for model prediction"""
    # Resize image to 224x224 pixels as required by the model
    img = img.resize((224, 224))
    # Convert the image to a numpy array
    img_array = tf_image.img_to_array(img)
    # Expand dimensions to match the model input shape
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess the image array for EfficientNet model
    img_array = preprocess_input(img_array)
    return img_array

# Streamlit app setup
# Set the title of the Streamlit app
st.title('Cataract Detection')

# Create a file uploader in Streamlit for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Open the uploaded image file
    image = Image.open(uploaded_file)
    # Display the uploaded image in the Streamlit app
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Prepare the image for prediction
    img_array = prepare_image(image)

    # Predict the class of the image using the loaded model
    prediction = model.predict(img_array)
    # Extract the confidence score for the 'Normal' class
    confidence = float(prediction[0][0])
    # Determine the predicted class based on the confidence score
    predicted_class = 1 if confidence > 0.5 else 0

    # Display the prediction and confidence in the Streamlit app
    st.write(f'Prediction: {class_labels[predicted_class]}')
    st.write(f'Confidence: {confidence if predicted_class == 1 else 1 - confidence}')
