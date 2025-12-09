import cv2 as cv
import numpy as np
import time as tm

haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')  # Load eye cascade

# Try opening camera with a couple of backends (Windows users: try CAP_DSHOW)
cap = cv.VideoCapture(0)
if not cap.isOpened():
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened():
    # try other indices
    for i in (1,2,3):
        cap = cv.VideoCapture(i)
        if cap.isOpened():
            print("Opened camera index", i)
            break

if not cap.isOpened():
    print("ERROR: Could not open video capture. Check camera, permissions, or try different index/backend.")
    raise SystemExit

# Load heavy resources once (moved outside loop)
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
pics = ['psp','mom','pops']

eye_closed_start=None
eye_closed_dur=0
alert_played=False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed — stopping")
        break
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces_rect=haar_cascade.detectMultiScale(gray,1.1,6)
    for (x,y,w,h) in faces_rect:
        faces_roi=gray[y:y+h,x:x+w]
        label,confidence=face_recognizer.predict(faces_roi)
        print(f'Label={pics[label]} with a confidence of {confidence}')
        cv.putText(frame,str(pics[label]),(x,y-40),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=2)
        eyes=eye_cascade.detectMultiScale(faces_roi,1.1,3)
        if len(eyes)==0:
            if eye_closed_start is None:
                eye_closed_start=tm.time()
            else:
                eye_closed_dur=tm.time()-eye_closed_start
            if eye_closed_dur>=10:
                status="ALERT: Sleeping!!!!"
            else:
                status="Eyes Closed"
        else:
            eye_closed_start=None
            eye_closed_dur=0
            status="Eyes Open"
        cv.putText(frame,status,(x-20,y),cv.FONT_HERSHEY_COMPLEX,0.8,(0,0,255),2)
    cv.imshow('Detected Face', frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
