import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av

st.title("👁️ AI Vision Assistant")

# AI Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def video_frame_callback(frame):
    # Convert camera frame to a format Python can read
    img = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Draw a green box around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# This creates the START button on your website
webrtc_streamer(
    key="vision-assistant",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)
