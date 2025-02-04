import numpy as np
import os
import cv2
from PIL import Image

def train_classifier(data_dir):
    faces, ids = [], []
    for image_path in [os.path.join(data_dir, f) for f in os.listdir(data_dir)]:
        img = np.array(Image.open(image_path).convert('L'), 'uint8')
        id = int(os.path.split(image_path)[1].split(".")[1])
        faces.append(img)
        ids.append(id)
    
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, np.array(ids))
    clf.write("classifier.xml")

train_classifier("data")
