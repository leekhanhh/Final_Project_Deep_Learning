# Load the augmented face images and corresponding labels
import os
import cv2
import numpy as np

def load_data_func(dataset_dir):
    images = []
    labels = []
    label_names = {}

    # Đọc nhãn từ labels.txt
    with open('../final_project/labels.txt', 'r') as file:
        for line in file:
            parts = line.split()
            label_names[int(parts[0])] = parts[1]

    # Load hình ảnh và nhãn
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(dataset_dir, filename)
            image = cv2.imread(image_path)
            if image is not None:  # Check if the image is successfully loaded
                images.append(image)
                # Extract label from filename
                label = int(filename.split('image_User_')[1].split('_')[0])
                labels.append(label)
    
    return np.array(images), np.array(labels), label_names