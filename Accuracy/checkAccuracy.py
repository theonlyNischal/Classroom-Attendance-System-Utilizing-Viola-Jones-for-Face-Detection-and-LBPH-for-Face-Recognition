

# Import the libraries
import json, glob, os
import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join
from helper.preprocess import preprocess_image, face_detector
font = cv.FONT_HERSHEY_COMPLEX

with open('labels_to_name.json') as json_file:
    labels_to_name = json.load(json_file)


# Load the model
model = cv.face.LBPHFaceRecognizer_create()
model.read("trainer/model.xml")


# Get the testing data path

total = 0
face_not_found = 0
correct = 0

for img_file in glob.glob("./Datasets/Testing/*.jpg"):
    total = total + 1
    comment = "Wrong"
    real_name, _, _, _, _, _, _, _, _, _, _, _, _ = img_file.split("_")
    correct_label = real_name.split("\\")[1]
    img = cv.imread(img_file)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face, _ = face_detector(gray_img)

    if face is not None:
            # Pass the face to model
            # "results" comprises of a tuple
            # containing label and confidence value
        face = preprocess_image(face)
        results = model.predict(face)
        confidence = int(100 * (1-(results[1]/300)))
        matched_id = str(results[0])
        predicted_label = labels_to_name[matched_id]["full_name"]
        

        if predicted_label == correct_label:
            correct = correct + 1
            comment = "Correct"
        else:
            print(img_file)
        print("Correct = {} Predicted = {} Confidence = {},  Comment = {}".format(correct_label, predicted_label, confidence, comment))
    else:
        os.remove(img_file)
        face_not_found = face_not_found + 1
        print("Face not Found.")
        print(img_file)

original_total = total
total = total - face_not_found

## calculates and print accuracy
accuracy = (correct/total)*100

print("--------------------------------------")

print("Total Testing Images: {}\nFace Not Found: {}\nRevised Total: {}\nCorrect: {}\nAccuracy: {}%".format(original_total, face_not_found, total, correct, accuracy))
