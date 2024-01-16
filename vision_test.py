import numpy as np
import cv2

# Load the trained model
model = cv2.face.EigenFaceRecognizer_create()
model.read('eigenface_model.yml')

# Path to your custom cascade file
custom_cascade_path = 'F:\Code\HMIvoice\cascade140.xml'

# Load your custom trained cascade
face_cascade = cv2.CascadeClassifier(custom_cascade_path)

if face_cascade.empty():
    raise Exception("Failed to load custom cascade. Check the file path.")

# Start video capture
cap = cv2.VideoCapture(0)

frame_skip = 5  # Skip every 5 frames
resize_scale = 0.5  # Scale down the frame to 50% for faster processing

# Parameters for smoothing the bounding box
smooth_factor = 5
last_boxes = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Adjust parameters as needed

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))

        label, confidence = model.predict(face_resized)
        if confidence < 3000:
            text = f"Person {label}"
        else:
            text = "Unknown"

        # Smoothing the bounding box
        last_boxes.append((x, y, w, h))
        if len(last_boxes) > smooth_factor:
            last_boxes.pop(0)
        
        avg_box = np.mean(last_boxes, axis=0).astype(int)
        cv2.rectangle(frame, (avg_box[0], avg_box[1]), (avg_box[0]+avg_box[2], avg_box[1]+avg_box[3]), (0, 255, 0), 2)
        cv2.putText(frame, text, (avg_box[0], avg_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
