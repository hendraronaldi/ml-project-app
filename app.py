# import libraries
import streamlit as st
import numpy as np
from apps.car_purchase_prediction import CarPurchasePrediction
from apps.simple_face_mask_detection import FaceMaskDetection

# function to select which app
def load_app(app_name):
	app = None
	if app_name == 'car purchase prediction':
		app = CarPurchasePrediction(apps[app_name])
	elif app_name == 'face mask detection':
		app = FaceMaskDetection(apps[app_name])
	elif app_name == 'spam detection':
		pass
	elif app_name == 'sign traffic classifier':
		pass
	return app

# app list
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
	st.write('Project error, please select another project from sidebar')