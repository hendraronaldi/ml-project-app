import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import joblib
import requests
from os import path

def set_target_labels(labels):
	return [x[1].decode('utf-8') for x in labels]

labels = [( 0, b'Speed limit (20km/h)'),
( 1, b'Speed limit (30km/h)'),
( 2, b'Speed limit (50km/h)'),
( 3, b'Speed limit (60km/h)'),
( 4, b'Speed limit (70km/h)'),
( 5, b'Speed limit (80km/h)'),
( 6, b'End of speed limit (80km/h)'),
( 7, b'Speed limit (100km/h)'),
( 8, b'Speed limit (120km/h)'),
( 9, b'No passing'),
(10, b'No passing for vehicles over 3.5 metric tons'),
(11, b'Right-of-way at the next intersection'),
(12, b'Priority road'),
(13, b'Yield'),
(14, b'Stop'),
(15, b'No vehicles'),
(16, b'Vehicles over 3.5 metric tons prohibited'),
(17, b'No entry'),
(18, b'General caution'),
(19, b'Dangerous curve to the left'),
(20, b'Dangerous curve to the right'),
(21, b'Double curve'),
(22, b'Bumpy road'),
(23, b'Slippery road'),
(24, b'Road narrows on the right'),
(25, b'Road work'),
(26, b'Traffic signals'),
(27, b'Pedestrians'),
(28, b'Children crossing'),
(29, b'Bicycles crossing'),
(30, b'Beware of ice/snow'),
(31, b'Wild animals crossing'),
(32, b'End of all speed and passing limits'),
(33, b'Turn right ahead'),
(34, b'Turn left ahead'),
(35, b'Ahead only'),
(36, b'Go straight or right'),
(37, b'Go straight or left'),
(38, b'Keep right'),
(39, b'Keep left'),
(40, b'Roundabout mandatory'),
(41, b'End of no passing'),
(42, b'End of no passing by vehicles over 3.5 metric tons')]
labels = set_target_labels(labels)

class SignTrafficClassifier:
	def __init__(self, repo):
		self.repo = repo
		self.load_model()

	def load_model(self):
		if not path.exists('apps/stc_model.h5'):
			m = requests.get(f'https://github.com/hendraronaldi/{self.repo}/blob/master/model.h5?raw=true')
			open('apps/stc_model.h5', 'wb').write(m.content)
		self.model = tf.keras.models.load_model('apps/stc_model.h5')

	def preprocess_image(self, img):
		pimg = img.copy()
		pimg = np.sum(img/3, axis=3, keepdims=True)
		pimg = (pimg - 128) / 128
		return pimg

	def info(self):
		st.write('Classify sign traffic from image')
		st.write("""This is part of project from Udemy course [Machine Learning Practical Workout 8 Real-World Projects](https://www.udemy.com/course/deep-learning-machine-learning-practical/)""")
		st.write("""Notebook can be found [here](https://github.com/hendraronaldi/machine_learning_projects/blob/main/Courses/Udemy%20Machine%20Learning%20Practical%20Workout%20%208%20Real-World%20Projects/MLProject%20LeNet%20Traffic%20Sign%20Classifier.ipynb)""")

	def input_features(self):
		st.subheader('Input Parameters')
		uploaded_img = st.file_uploader("Upload Image")
		if uploaded_img is not None:
			try:
				file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
				frame = cv2.imdecode(file_bytes, 1)

				if frame.shape[0] > 500:
					st.image(frame, channels="BGR", width=500)
				else:
					st.image(frame, channels="BGR")
			except:
				st.subheader("Please upload image")

		return frame

	def predict(self):
		self.info()
		frame = self.input_features()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.resize(frame, (32, 32))
		img = image.img_to_array(frame)
		img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
		img = self.preprocess_image(img)

		try:
			predictions = np.argmax(self.model.predict(img))
			self.show(predictions)
		except:
			self.show(None)

	def show(self, predictions):
		if predictions != None:
			st.subheader('Prediction Sign')
			st.write(labels[predictions])
		else:
			st.subheader("Please upload an image file")