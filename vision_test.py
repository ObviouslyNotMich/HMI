import numpy as np
import cv2

import cv2

# Load the trained model
# model = cv2.face.EigenFaceRecognizer_create()
# model.read('eigenface_model.yml')

# Path to your custom cascade file
custom_cascade_path = 'cascade.xml'

# Load your custom trained cascade
face_cascade = cv2.CascadeClassifier(custom_cascade_path)

# Check if the cascade is loaded correctly
if face_cascade.empty():
    raise Exception("Failed to load custom cascade. Check the file path.")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop and resize face to the same size used during training
        face = gray[y:y+h, x:x+w]
        # face_resized = cv2.resize(face, (32, 32))  # Resize to the standard size

        # # Predict the face
        # label, confidence = model.predict(face_resized)
        # print(confidence)
        # if confidence < 1500:  # Adjust the threshold as needed
        #     text = f"Person {label}"
        # else:
        #     text = "Unknown"

        # Draw a rectangle around the face and put text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()