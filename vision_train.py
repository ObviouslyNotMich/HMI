import os
import glob
import subprocess
import cv2

# ONLY WORKS WITH OPENCV 3.4.3!

def resize_images_in_folder(folder_path, target_size=(48, 48)):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)

                # Only resize if the image is not already the target size
                if image.shape[:2] != target_size:
                    resized_image = cv2.resize(image, target_size)
                    cv2.imwrite(image_path, resized_image)
                    print(f"Resized and saved: {image_path}")
                else:
                    print(f"Skipped resizing (already target size): {image_path}")

def check_and_create_classifier_dir():
    classifier_dir = r'F:\Code\HMIvoice\classifier'  # Path to your classifier directory
    params_file = os.path.join(classifier_dir, 'params.xml')

    # Check if the classifier directory exists, if not, create it
    if not os.path.exists(classifier_dir):
        os.makedirs(classifier_dir)
        print(f"Created directory: {classifier_dir}")

def generate_negative_description_file(negative_images_folder, output_file='negatives.txt'):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(negative_images_folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    f.write(root + '\\' + file + '\n')

def create_samples():
    # Replace with the full path to opencv_createsamples if it's not in PATH
    command = "opencv_createsamples -info positives.txt -num 357 -w 48 -h 48 -vec positives.vec"
    subprocess.run(command, shell=True)

def train_classifier():
    # Replace with the full path to opencv_traincascade if it's not in PATH
    command = "opencv_traincascade -data classifier -vec positives.vec -bg negatives.txt -numPos 340 -numNeg 32 -numStages 15 -w 48 -h 48 -numThreads 16"

    subprocess.run(command, shell=True)


positive_images_folder = r'F:\Code\HMIvoice\image\p'
negative_images_folder = r'F:\Code\HMIvoice\image\n'

check_and_create_classifier_dir()
resize_images_in_folder(positive_images_folder)
resize_images_in_folder(negative_images_folder)
generate_negative_description_file(negative_images_folder)


# create_samples()
# train_classifier()
print("Done")
