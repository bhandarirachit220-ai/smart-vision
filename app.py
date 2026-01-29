import streamlit as st
import cv2
import time

st.title("👁️  Smart vision")

# 1. Setup UI
run = st.checkbox('Start Assistant')
FRAME_WINDOW = st.image([])

# 2. Load the Face AI
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# This keeps track of time even when the app refreshes
if 'last_speech_time' not in st.session_state:
    st.session_state.last_speech_time = 0

# 3. Browser Voice Function
def speak(text):
    st.components.v1.html(f"""
        <script>
        var msg = new SpeechSynthesisUtterance('{text}');
        window.speechSynthesis.speak(msg);
        </script>
    """, height=0)

# 4. Main Loop
if run:
    camera = cv2.VideoCapture(0)
    
    while run:
        ret, frame = camera.read()
        if not ret: break

        # Mirror and Gray for better detection
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        current_time = time.time()

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Logic for Near vs Far (based on face width 'w')
                distance_msg = "Person is near be careful" if w > 180 else "Person is far"

                # --- 4 SECOND DELAY LOGIC ---
                if current_time - st.session_state.last_speech_time > 4:
                    speak(distance_msg)
                    st.session_state.last_speech_time = current_time
                    st.toast(distance_msg) # Show a small message on screen too

        # Update the video frame on the website
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

    camera.release()
else:
    st.info("Check the box to start. The assistant will speak every 4 seconds.")