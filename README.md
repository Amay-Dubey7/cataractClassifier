# Cataract Detection API

This project is a Streamlit-based web application for detecting cataracts in uploaded eye images using a pre-trained EfficientNet model.

## Features
- Upload an image of an eye in JPG, JPEG, or PNG format.
- The app classifies the image as either "Cataract" or "Normal".
- Displays the uploaded image and prediction confidence.

## Setup Instructions

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Amay-Dubey7/cataractClassifier.git
cd cataractClassifier
```

2. **Install the required packages:**
```bash
pip install -r requirements.txt
```
```bash
3. **Run the application:**
streamlit run app.py
```

```bash
4. **Open your web browser and go to:**
http://localhost:8501
```

## File Structure
- app.py: The main Streamlit application script.
- cataractEfficientNet.h5: The pre-trained EfficientNet model file.
- requirements.txt: The file containing all the dependencies.

## How to Use the API
- Upload an Image:

- Click on "Choose an image..." and select a JPG, JPEG, or PNG file from your device.

## View Prediction:

-Once the image is uploaded, the application will display the image and classify it as either "Cataract" or "Normal".
-The prediction confidence will also be displayed.

## Model Information
- Model Architecture: EfficientNet
- Input Image Size: 224x224 pixels
- Classes: Cataract, Normal

## Acknowledgements
- This application uses TensorFlow and Streamlit.
- The model is pre-trained on a dataset for cataract detection.

