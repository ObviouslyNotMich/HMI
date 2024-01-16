import cv2
import os
import numpy as np

def get_images_and_labels(path):
    images = []
    labels = []
    label_dict = {}
    current_label = 0

    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(dirname, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    if dirname not in label_dict:
                        label_dict[dirname] = current_label
                        current_label += 1
                    labels.append(label_dict[dirname])

    return images, np.array(labels)

# Path to the dataset
dataset_path = r'F:\Code\HMIvoice\image\p'

# Prepare the dataset
images, labels = get_images_and_labels(dataset_path)

# Create an EigenFace Recognizer
model = cv2.face.EigenFaceRecognizer_create()

# Train the model
model.train(images, labels)

# Save the model
model.save("eigenface_model.yml")
