import cv2
import av
import streamlit as st
import tensorflow as tf
from os import path
from streamlit_webrtc import (
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
from twilio.rest import Client

sid = st.secrets['secrets']['TURN_SID']
auth = st.secrets['secrets']['TURN_AUTH']

client = Client(sid, auth)
token = client.tokens.create()


def human_emotion_detection():
    WEBRTC_CLIENT_SETTINGS = ClientSettings(
        rtc_configuration={"iceServers": token.ice_servers},
        # rtc_configuration={"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]},
        media_stream_constraints={
            "video": True,
        },
    )

    class HumanEmotionDetection(VideoProcessorBase):
        def __init__(self):
            self.predictions = None
            self.frame = None
            self.load_model()

        def load_model(self):
            if not path.exists('apps/hed_model.h5'):
                raise Exception("No available model exists, please choose another project.")
            self.model =  tf.keras.models.load_model('apps/hed_model.h5')

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            self.frame = frame
            try:
                frame = frame.to_ndarray(format="bgr24")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                im = tf.constant(frame, dtype=tf.float32)
                im = tf.expand_dims(im, axis=0)
                self.predictions = ["angry", "happy", "nothing", "sad"][tf.argmax(self.model(im), axis=-1).numpy()[0]]
            except Exception as e:
                print(e)

    webrtc_ctx = webrtc_streamer(
        key="human-emotion-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=HumanEmotionDetection,
        async_processing=True,
    )

    capture_frame = None

    if webrtc_ctx.state.playing:
        if st.button("Click to Make Prediction") and webrtc_ctx.video_processor:
            capture_frame = webrtc_ctx.video_processor.frame

        st.header('Your Emotion')
        pred_text = st.empty()

        if capture_frame is not None:
            st.image(capture_frame.to_ndarray(format="rgb24"))
            pred = webrtc_ctx.video_processor.predictions
            pred_text.subheader(pred)
