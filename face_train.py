import cv2 as cv
import numpy as np
import os
# img=cv.imread('D:\popololo\pics\psp.jpg')
pics=['mummy','peebs','pops']
dir=r'D:\popololo\pics'
haar_cascade=cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
features=[]
labels=[]
def create_train():
   for pic in pics:
      path=os.path.join(dir,pic+'.jpg')
      label=pics.index(pic)
      img_array=cv.imread(path)
      for img in os.listdir(os.path.join(dir,pic)):
         path=os.path.join(dir,pic,img)
         img_array=cv.imread(path)
      if img_array is None:
        print(f"Image not found at {path}")
        continue
      gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
      faces_rect=haar_cascade.detectMultiScale(gray,1.1,3)
      for (x,y,w,h) in faces_rect:
            faces_roi=gray[y:y+h,x:x+w]
            features.append(faces_roi)
            labels.append(label)
            print(f"Face ROI shape: {faces_roi.shape}")
create_train()
if len(features) == 0 or len(labels) == 0:
    print("No faces detected. Training cannot proceed.")
    exit()
features = [np.array(face, dtype=np.uint8) for face in features]
labels=np.array(labels)
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,np.array(labels))
np.save('features.npy', np.array(features, dtype=object))
np.save('labels.npy',labels)
face_recognizer.save('face_trained.yml')
print('Training done')