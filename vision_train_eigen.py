import cv2
import os
import numpy as np
import shutil

def read_annotations(annotations_path):
    annotations = {}
    with open(annotations_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            image_path = parts[0]
            # Assuming the format is: [image_path x y width height]
            x, y, width, height = map(int, parts[2:6])
            annotations[image_path] = (x, y, width, height)
    return annotations


def create_cropped_image_dirs(original_path, cropped_path):
    if os.path.exists(cropped_path):
        shutil.rmtree(cropped_path)  # Remove the directory if it already exists
    os.makedirs(cropped_path)  # Create the root directory for cropped images

    for root, dirs, _ in os.walk(original_path):
        for dirname in dirs:
            dirpath = os.path.join(root, dirname)
            relative_path = os.path.relpath(dirpath, original_path)
            new_dirpath = os.path.join(cropped_path, relative_path)
            os.makedirs(new_dirpath)

def get_images_and_labels(path, annotations, cropped_path, standard_size=(32, 32)):
    images = []
    labels = []
    label_dict = {}
    current_label = 0

    for dirname, _, filenames in os.walk(path):
        person_name = os.path.basename(dirname)
        if person_name not in label_dict:
            label_dict[person_name] = current_label
            current_label += 1

        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(dirname, filename)
                if img_path in annotations:
                    x, y, w, h = annotations[img_path]
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        face = img[y:y+h, x:x+w]  # Crop the face region
                        resized_face = cv2.resize(face, standard_size)  # Resize the face
                        images.append(resized_face)
                        labels.append(label_dict[person_name])

                        # Save the resized cropped image in the corresponding new directory
                        relative_dir = os.path.relpath(dirname, path)
                        cropped_img_path = os.path.join(cropped_path, relative_dir, filename)
                        cv2.imwrite(cropped_img_path, resized_face)

    return images, np.array(labels)



# Paths
dataset_path = r'F:\Code\HMIvoice\image\p'
cropped_dataset_path = r'F:\Code\HMIvoice\image\p_cropped'
annotations_path = r'F:\Code\HMIvoice\positives.txt'

# Create directories for cropped images
create_cropped_image_dirs(dataset_path, cropped_dataset_path)

# Read the annotations
annotations = read_annotations(annotations_path)

# Prepare the dataset with cropped images
images, labels = get_images_and_labels(dataset_path, annotations, cropped_dataset_path)

# Create an EigenFace Recognizer
model = cv2.face.EigenFaceRecognizer_create()

# Train the model
model.train(images, labels)

# Save the model
model.save("eigenface_model.yml")
