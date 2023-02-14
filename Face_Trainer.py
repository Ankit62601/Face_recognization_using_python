#%%
''' Training Face Dataset '''

import cv2
import os
import numpy as np
from PIL import Image

names = []   # name of persons
paths = []   # path of each image

# names = [users for users in os.listdir("Dataset")]
for users in os.listdir("Dataset") :
    names.append(users)

# print(os.getcwd())    # working directory
for name in names :     # select person's name from names list
    for image in os.listdir(f"Dataset/{name}") :    # select image from person's name folder
        img_path = os.path.join(f"Dataset/{name}",image)
        paths.append(img_path)    

# print(names,len(paths),paths)   

faces = [] 
labels = []

for img_path in paths :
    image = Image.open(img_path).convert("L")   # to convert image to B&W
    # image.show()      # to show image using Image module
    ImgNp = np.array(image,dtype='uint8')    # create numpy_array of images
    faces.append(ImgNp)
    id = int(img_path.split('\\')[1].split('_')[0])   # get id from image name
    labels.append(id)

faces = np.array(faces,dtype=object)
labels = np.array(labels)     # convert to numpy_array

# print(labels)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces,labels)    # LBPH recognizer trains the model with faces & labels

recognizer.write("Trainer/faces_trained.yml")   # create a .yml file after training the model

print("Trained Successfully...!!!")
# %%
