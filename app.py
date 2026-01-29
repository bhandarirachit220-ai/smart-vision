import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2

st.title("👁️ Live Vision Assistant")

# This function draws the boxes and detects faces
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # AI Detection
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# THE BRIDGE (This is the part that usually breaks)
webrtc_streamer(
    key="vision",
    video_frame_callback=video_frame_callback,
    # This Google server helps the camera find your browser
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)
