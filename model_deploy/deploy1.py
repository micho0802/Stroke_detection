import streamlit as st
import cv2
import requests
import numpy as np
import base64

st.title("Real-Time Motion Detection and Event Prediction")

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame in Streamlit
    st.image(frame, channels="BGR")

    # Convert the frame to the correct format (encode to base64)
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')  # Encode as base64

    # Predict button with a unique key
    if st.button('Predict', key='predict_button'):
        # Send the frame to the Flask API for prediction
        headers = {"Content-Type": "application/json"}
        data = {'frame': frame_base64}
        response = requests.post('http://127.0.0.1:5000/predict', json=data, headers=headers)

        if response.status_code == 200:
            st.write(response.json())
        else:
            st.write('Error:', response.status_code)

    # Stop button with a unique key
    if st.button('Stop', key='stop_button'):
        break

cap.release()
