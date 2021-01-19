# import libraries
import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.applications.vgg16 import VGG16 as PretrainedModel, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import joblib
from apps.car_purchase_prediction import CarPurchasePrediction

# function to load supporting files, e.g. model, scaler, encoder, etc.
# function to select which app
def load_app(app_name):
	app = None
	if app_name == 'car purchase prediction':
		app = CarPurchasePrediction(apps[app_name])
	return app

# input parameters
apps = {
	'car purchase prediction': 'car-purchase-prediction',
	'face mask detection': 'simple-face-mask-detection',
	'sign traffic classifier': 'sign-traffic-classifier',
	'spam detection': 'spam-detection'
}

# sidebar
st.sidebar.title('Hendra Ronaldi')
st.sidebar.write('Machine learning & Data science enthusiast')
st.sidebar.write("""
	[Github](https://github.com/hendraronaldi)
	[Linkedin](https://www.linkedin.com/in/hendra-ronaldi-4a7a1b121/)
""")
st.sidebar.write('***')
st.sidebar.header('Project List')
app_name = st.sidebar.selectbox('Select app', list(apps.keys()))

# main page
if app_name == None or app_name == '':
	st.title('Please select a project from sidebar')
else:
	st.title(app_name.title())

try:
	app = load_app(app_name)
	app.predict()
except:
	st.write('Please select a project from sidebar')