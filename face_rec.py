import cv2 as cv
import numpy as np
import os
img=cv.imread('D:\popololo\pics\psp.jpg')
face_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    canny=cv.Canny(gray,100,200)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.imshow("Face Detection", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.imshow('Detected Face',canny)
cv.waitKey(0)   

cap.release()
cv.destroyAllWindows()
