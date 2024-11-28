#!/usr/bin/env python
# coding: utf-8

# In[35]:


import cv2
import numpy as np

# Load pre-trained Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to detect faces and eyes
def detect_faces_and_eyes(image):
    if image is None:
        print("Error: Unable to read the image.")
        return None, 0, 0  # Returning 0 counts if image is None

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Draw rectangles around the detected faces and eyes
    detected_faces = len(faces)
    detected_eyes = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        detected_eyes += len(eyes)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return image, detected_faces, detected_eyes

# Paths to the images (ensure these paths are correct)
image_paths = [
    r"C:\Users\Administrator\Desktop\JupyterNotebooks\Lab 05\Sample Images\Pic6.jpeg",
    r"C:\Users\Administrator\Desktop\JupyterNotebooks\Lab 05\Sample Images\Pic7.jpeg",
    r"C:\Users\Administrator\Desktop\JupyterNotebooks\Lab 05\Sample Images\Pic8.jpeg",
    r"C:\Users\Administrator\Desktop\JupyterNotebooks\Lab 05\Sample Images\Pic9.jpeg",
    r"C:\Users\Administrator\Desktop\JupyterNotebooks\Lab 05\Sample Images\Pic10.jpeg"
]

# Expected counts (adjust as necessary)
expected_face_counts = [2, 1, 3, 0, 2]  # Example values
expected_eye_counts = [4, 2, 6, 0, 4]  # Example values

# Prepare to display images in a single window
results = []
target_height = 300  # Set a target height for resizing

for idx, image_path in enumerate(image_paths):
    image = cv2.imread(image_path)
    result_image, detected_faces, detected_eyes = detect_faces_and_eyes(image)

    # Check if result_image is None
    if result_image is None:
        print(f"Image at {image_path} could not be processed.")
        continue

    # Calculate accuracy percentages
    face_accuracy = (detected_faces / expected_face_counts[idx]) * 100 if expected_face_counts[idx] > 0 else 100
    eye_accuracy = (detected_eyes / expected_eye_counts[idx]) * 100 if expected_eye_counts[idx] > 0 else 100

    # For the 1st image in the 1st row, resize it while maintaining aspect ratio without distortion
    if idx == 0:
        aspect_ratio = result_image.shape[1] / result_image.shape[0]
        new_width = int(target_height * aspect_ratio)
        resized_image = cv2.resize(result_image, (new_width, target_height))

        # Create a blank image to hold the resized image and the accuracy text
        blank_image = np.zeros((target_height + 60, new_width, 3), dtype=np.uint8)  # Extra space for text
        blank_image[0:target_height, 0:new_width] = resized_image

        # Add accuracy text below the image
        cv2.putText(blank_image, f'Face Acc: {face_accuracy:.1f}%', (10, target_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(blank_image, f'Eye Acc: {eye_accuracy:.1f}%', (10, target_height + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        results.append(blank_image)

    # For other images, use original resize strategy
    else:
        # Resize the image to the target height while maintaining aspect ratio (fix stretch issue)
        aspect_ratio = result_image.shape[1] / result_image.shape[0]
        new_width = int(target_height * aspect_ratio)
        resized_image = cv2.resize(result_image, (new_width, target_height))

        # Create a blank image to hold the resized image and the accuracy text
        blank_image = np.zeros((target_height + 60, new_width, 3), dtype=np.uint8)  # Extra space for text
        blank_image[0:target_height, 0:new_width] = resized_image

        # Add accuracy text below the image
        cv2.putText(blank_image, f'Face Acc: {face_accuracy:.1f}%', (10, target_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(blank_image, f'Eye Acc: {eye_accuracy:.1f}%', (10, target_height + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        results.append(blank_image)

# Combine images into two rows (2 images in the first row, 3 in the second)
first_row = np.hstack(results[:2]) if len(results) >= 2 else np.array(results[:2])
second_row = np.hstack(results[2:]) if len(results) > 2 else np.array([])

# Ensure rows are displayed without stretching
if second_row.size > 0:
    if first_row.shape[1] != second_row.shape[1]:
        max_width = max(first_row.shape[1], second_row.shape[1])
        first_row = cv2.resize(first_row, (max_width, first_row.shape[0]))
        second_row = cv2.resize(second_row, (max_width, second_row.shape[0]))

    combined_image = np.vstack((first_row, second_row))
else:
    combined_image = first_row if first_row.size > 0 else np.zeros((1, 1, 3), dtype=np.uint8)

# Display the combined result
cv2.imshow('Face and Eye Detection Results', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




