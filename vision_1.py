import cv2
import os

def detect_faces(image_path, cascade_classifier):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image

def process_directory(directory, cascade_classifier):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                processed_image = detect_faces(image_path, cascade_classifier)
                cv2.imshow("Faces found", processed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

if __name__ == "__main__":
    cascade_path = "path_to_haar_cascade_xml"  # Replace with the path to your Haar cascade XML file
    cascade_classifier = cv2.CascadeClassifier(cascade_path)

    images_directory = r'F:\Code\HMIvoice\image'
    process_directory(images_directory, cascade_classifier)
