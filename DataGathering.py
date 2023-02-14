#%%
''' Face Data Gathering to train model '''
import cv2
import os

face_name = input('\n Enter your name : ')
face_id = input('\n Enter ID : ')

# Create Directory of Person's Name 
if os.path.exists("Dataset/" + str(face_name) + "_" + str(face_id)):
    pass
else:
    os.mkdir("Dataset/" + str(face_name) + "_" + str(face_id))

print("\n Initializing face capture. Look the camera and wait.... ")
cam = cv2.VideoCapture(0)
cam.set(3, 640)     # set video width
cam.set(4, 480)     # set video height

face_detector = cv2.CascadeClassifier('Cascade_Classifier/haarcascade_frontalface_default.xml')

count = 0    # Initialize individual sampling face count

while(True):
    ret, img = cam.read()
    # img = cv2.flip(img, 1)      # flip video image horizontally
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_img, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        count += 1

        crop_img = gray_img[y:y+h, x:x+w]

        # Save the captured image into the Dataset folder
        cv2.imwrite("Dataset/" + str(face_name) + "_" + str(face_id) + "/" + str(face_id) + '_' + str(count) + ".jpg", crop_img)

        cv2.imshow('image', img)

    k = cv2.waitKey(1000) & 0xff    # it will capture 2 image per second
    if k == 27:
        break           # Press 'Esc' to exit
    elif count >= 50:    # Take 30 face sample and stop video
        break

print("\n Successfully Completed.... ")
cam.release()
cv2.destroyAllWindows()
# %%
