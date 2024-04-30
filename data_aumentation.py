#%%
import cv2
import os
import numpy as np

# Function to perform data augmentation on face images
def augment_data(image):
    augmented_images = []
    
    # Flip the image horizontally
    flipped_image = cv2.flip(image, 1)
    augmented_images.append(flipped_image)
    
    # Rotate the image by 10 degrees clockwise
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    augmented_images.append(rotated_image)
    
    # Add Gaussian noise to the image
    mean = 0
    std_dev = 30
    noise = np.random.normal(mean, std_dev, image.shape)
    noisy_image = np.clip((image + noise).astype(np.uint8), 0, 255)
    augmented_images.append(noisy_image)
    
    return augmented_images

# Path to the directory containing face images

dataset_dir = '../final_project/dataset'
augmentated_dataset = '../final_project/dataset/augmented_dataset'
# Iterate through each image in the dataset directory
for filename in os.listdir(dataset_dir):
    if filename.endswith(".jpg"):
        # Read the image
        image_path = os.path.join(dataset_dir, filename)
        image = cv2.imread(image_path)
        
        # Perform data augmentation
        augmented_images = augment_data(image)
        
        # Save augmented images
        for idx, augmented_image in enumerate(augmented_images):
            augmented_image_path = os.path.join(augmentated_dataset, f"{filename.split('.')[0]}_aug_{idx}.jpg")
            cv2.imwrite(augmented_image_path, augmented_image)

# %%
