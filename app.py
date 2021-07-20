# import libraries
import streamlit as st
import numpy as np
from apps.car_purchase_prediction import CarPurchasePrediction
from apps.simple_face_mask_detection import face_mask_detection
from apps.hotel_cancelation_prediction import HotelCancelation
from apps.spam_detection import SpamDetection
from apps.sign_traffic_classifier import SignTrafficClassifier

# function to select which app
def load_app(app_name):
	app = None
	if app_name == 'car purchase prediction':
		app = CarPurchasePrediction(apps[app_name])
	elif app_name == 'face mask detection':
		face_mask_detection()
	elif app_name == 'hotel cancelation prediction':
		app =  HotelCancelation(apps[app_name])
	elif app_name == 'spam detection':
		app = SpamDetection(apps[app_name])
	elif app_name == 'sign traffic classifier':
		app = SignTrafficClassifier(apps[app_name])
	return app

# app list
apps = {
	'car purchase prediction': 'car-purchase-prediction',
	'face mask detection': 'simple-face-mask-detection',
	'hotel cancelation prediction': 'hotel-demand-booking',
	'sign traffic classifier': 'sign-traffic-classifier',
	'spam detection': 'spam-detection',
	'wushu pose similarity': 'wushu-pose-similarity'
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

if app_name == 'wushu pose similarity':
	st.write("""
		Please use this [link](https://wushu-pose-estimation.herokuapp.com/poseSimilarity) to open the app, thanks
	""")
else:
	try:
		if app_name == 'face mask detection':
			load_app(app_name)
		else:
			app = load_app(app_name)
			app.predict()
	except Exception as e:
		print(f'error: {e}')
		st.write('Project error, please select another project from sidebar')