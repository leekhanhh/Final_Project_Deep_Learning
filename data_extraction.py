#%%
import cv2
import os
import time
from os import listdir
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from PIL import Image

import argparse
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# function for face detection with mtcnn
def extract_face(pixels, required_size=(320, 320)):
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    #Check if any faces are detected
    if results:
        # Extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # Bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # Extract the face
        face = pixels[y1:y2, x1:x2]
        # Resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        return face_array
    else:
        # If no face is detected, return None
        return None

# Capture video from the webcam
video_capture = cv2.VideoCapture(0)
image_counter = 0
dataset_dir = 'dataset'
project_path = 'D:/Workspace/AI/Deep/final_project'
dataset_path = 'D:/Workspace/AI/Deep/final_project/dataset'
if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)

face_id = input('\n Make sure the first user entered is 0.enter user id end press <return> ==>  ')
face_name = input(f'\n Give the name for {face_id} <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
label_file_path = os.path.join(project_path, 'labels.txt')
# Capture frame-by-frame

with open(label_file_path, 'a') as file:
    if os.path.exists(label_file_path):
        file.write('\n')
    text_to_append = f'{face_id} {face_name}'
    file.write(text_to_append)

while True:
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect faces in the frame
    face_array = extract_face(rgb_frame)

    # Display the resulting frame
    if face_array is not None:
        cv2.imshow('Video', cv2.cvtColor(face_array, cv2.COLOR_RGB2BGR))
        # Save the captured image to the dataset directory
        image_path = os.path.join(dataset_dir, f'image_User_{face_id}_{image_counter}.jpg')
        cv2.imwrite(image_path, cv2.cvtColor(face_array, cv2.COLOR_RGB2BGR))

        # Increment the image counter
        image_counter += 1
        # Check if captured enough images
        if image_counter >= 20:
            break
    else:
        cv2.imshow('Video', frame)

    k = cv2.waitKey(50) & 0xFF
    if k == 27:  # press 'ESC' to quit
        break

print("\n [INFO] Exiting Program and cleanup stuff")
# Release the capture
video_capture.release()
cv2.destroyAllWindows()





# %%
