## Import necessary packages
import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join

# Load HAAR face classifier
face_cascade_path = "assets/face_detection/haarcascade_frontalface_default.xml"
left_eye_cascade_path = "assets/face_detection/haarcascade_left_eye.xml"
right_eye_cascade_path = "assets/face_detection/haarcascade_right_eye.xml"
face_cascade = cv.CascadeClassifier(face_cascade_path)
left_eye_cascade = cv.CascadeClassifier(left_eye_cascade_path)
right_eye_cascade = cv.CascadeClassifier(right_eye_cascade_path)

# defines code for face detector
def face_detector(gray_img):
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

    if faces is ():
        return None, (None, None, None,None)
    
    for (x,y,w,h) in faces:
        cv.rectangle(gray_img, (x,y), (x+w, y+h), (0, 255, 255), 2)
        cropped_face = gray_img[y:y+h, x:x+w]
        roi = cv.resize(cropped_face, (200, 200))
        coord = (x,y,w,h)

    return roi, coord


## divides image into half and apply bilateral filter
def preprocess_image(gray_img):
    height, width = gray_img.shape[:2]

    midX = int(width/2)
    leftSide = gray_img[int(0):height, int(0):midX]
    rightSide = gray_img[int(0):height, midX:width]

    equL = cv.equalizeHist(leftSide)
    equR = cv.equalizeHist(rightSide)

    img_rotated_equalized = np.concatenate((equL, equR), axis=1)
    img = cv.bilateralFilter(img_rotated_equalized, 15, 75, 75)
    return img
    


# def preprocess_image(gray_img):
#     height, width = gray_img.shape[:2]
#     left_eye_region = gray_img[int(0.2*height):int(0.5*height), int(0.1*width):int(0.5*width)]
#     left_eye = left_eye_cascade.detectMultiScale(left_eye_region, scaleFactor=1.1, minNeighbors=3, flags=cv.CASCADE_FIND_BIGGEST_OBJECT)
#     left_eye_center = None
#     for (xl, yl, wl, hl) in left_eye:
#     # find the center of the detected eye region
#         left_eye_center = np.array([0.1 * width + xl + wl / 2, 0.2 * height + yl + hl / 2])
#         break # need only look at first, largest eye

#     right_eye_region = gray_img[int(0.2*height):int(0.5*height), int(0.5*width):int(0.9*width)]
#     right_eye = right_eye_cascade.detectMultiScale(right_eye_region, scaleFactor=1.1, minNeighbors=3,flags=cv.CASCADE_FIND_BIGGEST_OBJECT)
#     right_eye_center = None
#     for (xr, yr, wr, hr) in right_eye:
#     # find the center of the detected eye region
#         right_eye_center = np.array([0.5 * width + xr + wr / 2, 0.2 * height + yr + hr / 2])
#         break  # need only look at first, largest eye

#     if left_eye_center is None or right_eye_center is None:
#         return None

#     desired_eye_x = 0.25
#     desired_eye_y = 0.2

#     desired_img_width = 200
#     desired_img_height = desired_img_width

#     eye_center = (left_eye_center + right_eye_center) / 2
#     eye_angle_deg = np.arctan2( right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0]) * 180.0 / np.pi
    
#     eye_size_scale = (1.0 - desired_eye_x * 2) * desired_img_width / np.linalg.norm(right_eye_center - left_eye_center)

#     rot_mat = cv.getRotationMatrix2D(tuple(eye_center), eye_angle_deg, eye_size_scale)

#     rot_mat[0,2] += desired_img_width*0.5 - eye_center[0]
#     rot_mat[1,2] += desired_eye_y*desired_img_height - eye_center[1]

#     img_rotated = cv.warpAffine(gray_img, rot_mat, (desired_img_width, desired_img_height))

#     # histogram equalization

#     midX = int(width/2)
#     leftSide = img_rotated[int(0):height, int(0):midX]
#     rightSide = img_rotated[int(0):height, midX:width]

#     equL = cv.equalizeHist(leftSide)
#     equR = cv.equalizeHist(rightSide)

#     img_rotated_equalized = np.concatenate((equL, equR), axis=1)
#     img = cv.bilateralFilter(img_rotated_equalized, 15, 75, 75)
#     return img
