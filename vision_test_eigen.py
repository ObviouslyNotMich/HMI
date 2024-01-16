import cv2
import numpy as np

def predict_face(model, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize or preprocess the frame as needed
    label, confidence = model.predict(gray)
    return label, confidence

def recognize_faces(model, image_path=None):
    if image_path:
        # Recognize faces from an image file
        frame = cv2.imread(image_path)
        label, confidence = predict_face(model, frame)
        cv2.putText(frame, f"Label: {label}, Confidence: {confidence}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Recognition', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # Start the camera for live face recognition
        cap = cv2.VideoCapture(0)  # '0' for the primary camera

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            label, confidence = predict_face(model, frame)
            cv2.putText(frame, f"Label: {label}, Confidence: {confidence}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Load the trained model
model_path = "eigenface_model.yml"
model = cv2.face.EigenFaceRecognizer_create()
model.read(model_path)

# To use with an image
# recognize_faces(model, image_path='path/to/image.jpg')

# To use with live camera stream
recognize_faces(model)
