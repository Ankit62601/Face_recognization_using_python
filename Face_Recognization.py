#%%
''' Face Recognization using OpenCV '''
# Face Recognization based on Local Binary Patterns Histogram(LBPH) Algorithm developed in 1996.

import cv2
import os

faceCascade = cv2.CascadeClassifier('Cascade_Classifier/haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer/faces_trained.yml")

names = []   # name of persons
ids = []    # ID
for user in os.listdir("Dataset") :
    names.append(user.split("_")[0])
    ids.append(user.split("_")[1])

# names = [users for users in os.listdir("Dataset")]

capture = cv2.VideoCapture(0)    # start capturing
capture.set(3, 640)    # set Width
capture.set(4, 480)    # set Height

while True:
    isTrue, img = capture.read()
    img = cv2.flip(img, 1)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        id, confidence  = recognizer.predict(gray_img[y:y+h, x:x+w])    # to predict ID of faces

        # print name of identified person
        if confidence > 50 :
            cv2.putText(img, names[ids.index(str(id))], (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else :
            cv2.putText(img, "Unknown", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow('Live Face Recognization....', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:    # press 'ESC' to quit
        break

capture.release()
cv2.destroyAllWindows()

# %%
