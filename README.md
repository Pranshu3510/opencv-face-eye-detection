# OpenCV Face Eye Detection
This project uses OpenCV in Python to detect faces, recognize users using the LBPH Face Recognizer, and monitor eye states (open/closed) in real time.
If the eyes remain closed for a continuous duration (e.g. 10 seconds), the system displays a visual alert.

# Features
Real-time face detection using Haar Cascades.
Face recognition using LBPH algorithm.
Eye detection and monitoring (open/closed).
Alert system when eyes remain closed beyond threshold.
Lightweight, fast, runs on webcam in real time.
Written fully in Python.

# Working
Camera feed is captured frame-by-frame.
Faces are detected using Haar Cascade.
Each detected face is recognized using a trained LBPH model.
Eye regions are extracted and checked for open/closed state.
Eye closure duration is observed.
System displays alert if eyes are closed too long (e.g. 10 seconds).

# Customization
Eye closure threshold.
Alert text.
Face recognition labels.
Display configurations.

# Notes
Ensure good lighting.
cascades work better when facing camera directly.
Use good webcam with more FPS for better accuracy.
Upload more Photos of a face for more accuracy.
