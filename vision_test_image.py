import cv2

# Path to the image and cascade file
image_path = 'peeps_bunch.jpg'  # Replace with your image path
cascade_path = 'classifier/cascade.xml'  # Replace with your cascade path

# Load the image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the custom cascade
face_cascade = cv2.CascadeClassifier(cascade_path)

# Check if the cascade is loaded correctly
if face_cascade.empty():
    raise Exception("Failed to load custom cascade. Check the file path.")

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with detected faces
cv2.imshow('Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
