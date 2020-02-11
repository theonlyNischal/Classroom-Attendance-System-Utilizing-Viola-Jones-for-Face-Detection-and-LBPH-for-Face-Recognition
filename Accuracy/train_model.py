import json
import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join
from helper.preprocess import face_detector, preprocess_image


# Get the training data path
data_path = "./Datasets/Training/"
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create list for training data and labels

training_data, labels = [], []

# Creating a numpy array for training data

labels_to_name = {}

total = 0
face_identified = 0
for i, files in enumerate(onlyfiles):
    total = total + 1
    image_path = data_path + onlyfiles[i]
    images = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    face, _ = face_detector(images)
    if face is not None:
        face = preprocess_image(face)
        face_identified = face_identified + 1
        training_data.append(np.asarray(face, dtype=np.uint8))
        labels.append(i)
        full_name, student_id, _, _, _, _, _, _, _, _, _, _, _ = files.split("_")
        labels_to_name[i] = {"full_name": full_name, "student_id": student_id}


print("Total: {} ---- Face Identified: {}".format(total, face_identified))
with open("labels_to_name.json", "w") as write_file:
    json.dump(labels_to_name, write_file)


# Creating a numpy array for both training 

labels = np.asarray(labels, dtype=np.int32)
training_data = np.asarray(training_data, dtype=np.uint8)
# Initialize the facial recognizer
model = cv.face.LBPHFaceRecognizer_create()

model.train(training_data, labels)
model.write("trainer/model.xml")

print("Models Trained Successfully.")