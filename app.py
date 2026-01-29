import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("👁️ Smart Vision Assistant")
st.write("Take a photo to check for people.")

# This opens the camera correctly on any phone or laptop
img_file = st.camera_input("Scan environment")

if img_file:
    # 1. Process the image
    img = Image.open(img_file)
    frame = np.array(img)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # 2. AI Detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 3. Voice & Vibration Output
    if len(faces) > 0:
        msg = "Person detected"
        st.success(msg)
        st.components.v1.html(f"""
            <script>
            window.navigator.vibrate(500);
            var msg = new SpeechSynthesisUtterance('{msg}');
            window.speechSynthesis.speak(msg);
            </script>
        """, height=0)
    else:
        st.info("No one found.")
