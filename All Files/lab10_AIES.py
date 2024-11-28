#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/KhuzaimaHassan/AI-and-ES/blob/main/lab10_AIES.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


#Loading Required Libraries
import torch
import cv2
from matplotlib import pyplot as plt
import numpy as np


# In[ ]:


# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use 'yolov5s', 'yolov5m', 'yolov5l', or 'yolov5x' for different sizes


# In[ ]:


# Function to detect objects in an image
def detect_objects(image, conf_thresh=0.4, iou_thresh=0.4):
    # Perform inference
    results = model(image)

    # Extract bounding boxes, confidences, and class IDs
    boxes = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, conf, class
    filtered_boxes = boxes[boxes[:, 4] > conf_thresh]  # Filter based on confidence threshold

    # Apply Non-Maximum Suppression (NMS)
    x1, y1, x2, y2, conf, cls = filtered_boxes[:, 0], filtered_boxes[:, 1], filtered_boxes[:, 2], filtered_boxes[:, 3], filtered_boxes[:, 4], filtered_boxes[:, 5]
    indices = cv2.dnn.NMSBoxes(bboxes=np.column_stack((x1, y1, x2 - x1, y2 - y1)).tolist(), scores=conf.tolist(), score_threshold=conf_thresh, nms_threshold=iou_thresh).flatten()

    # Return the filtered and suppressed bounding boxes
    return filtered_boxes[indices]

# Function to print the detected objects
def print_objects(boxes, class_names):
    for box in boxes:
        x1, y1, x2, y2, conf, class_id = box
        print(f"Detected {class_names[int(class_id)]} with confidence {conf:.2f}")

# Function to plot bounding boxes on the image
def plot_boxes(image, boxes, class_names, plot_labels=True):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    for box in boxes:
        x1, y1, x2, y2, conf, class_id = box
        # Draw rectangle and text on the image
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        if plot_labels:
            plt.text(x1, y1 - 10, f"{class_names[int(class_id)]}: {conf:.2f}", color='red', fontsize=12, weight='bold', backgroundcolor='white')

    plt.axis('off')
    plt.show()

# Load and prepare the image
img_path = '/content/city_scene.jpg'
img = cv2.imread(img_path)
if img is None:
    raise ValueError("Image not found or path is incorrect")

# Convert to RGB
original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Perform object detection
boxes = detect_objects(original_image)

# Print detected objects
class_names = model.names
print_objects(boxes, class_names)

# Plot the image with bounding boxes
plot_boxes(original_image, boxes, class_names)


# In[ ]:


plt.savefig('/content/output.jpg')


# In[ ]:


# Install necessary packages
get_ipython().system('pip install ultralytics opencv-python matplotlib')
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the YOLOv8 model
model = YOLO("yolov8n.pt")


# In[ ]:


# Function to detect objects
def detect_objects(image, conf_thresh=0.4, iou_thresh=0.4):
    results = model(image)
    boxes = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])

            if conf > conf_thresh:
                boxes.append([x1, y1, x2, y2, conf, cls])

    if not boxes:
        return []

    boxes = np.array(boxes)
    x1, y1, x2, y2, conf, cls = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5]

    indices = cv2.dnn.NMSBoxes(
        bboxes=np.column_stack((x1, y1, x2 - x1, y2 - y1)).tolist(),
        scores=conf.tolist(),
        score_threshold=conf_thresh,
        nms_threshold=iou_thresh
    ).flatten()

    return boxes[indices]

# Function to print detected objects
def print_objects(boxes, class_names):
    for box in boxes:
        x1, y1, x2, y2, conf, class_id = box
        print(f"Detected {class_names[int(class_id)]} with confidence {conf:.2f}")

# Function to plot bounding boxes on the image
def plot_boxes(image, boxes, class_names, plot_labels=True):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    for box in boxes:
        x1, y1, x2, y2, conf, class_id = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        if plot_labels:
            plt.text(x1, y1 - 10, f"{class_names[int(class_id)]}: {conf:.2f}", color='red', fontsize=12, weight='bold', backgroundcolor='white')

    plt.axis('off')
    plt.show()

# Load and prepare the image
img_path = "football_image.jpg"
img = cv2.imread(img_path)
if img is None:
    raise ValueError("Image not found or path is incorrect")

# Convert to RGB
original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Perform object detection
boxes = detect_objects(original_image)

# Print detected objects
class_names = model.names
print_objects(boxes, class_names)

# Plot the image with bounding boxes
plot_boxes(original_image, boxes, class_names)


# In[ ]:


# Load and prepare the image
img_path = "football_image1.jpg"
img = cv2.imread(img_path)
if img is None:
    raise ValueError("Image not found or path is incorrect")

# Convert to RGB
original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Perform object detection
boxes = detect_objects(original_image)

# Print detected objects
class_names = model.names
print_objects(boxes, class_names)

# Plot the image with bounding boxes
plot_boxes(original_image, boxes, class_names)


# In[ ]:


# Load and prepare the image
img_path = "football_image3.jpeg"
img = cv2.imread(img_path)
if img is None:
    raise ValueError("Image not found or path is incorrect")

# Convert to RGB
original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Perform object detection
boxes = detect_objects(original_image)

# Print detected objects
class_names = model.names
print_objects(boxes, class_names)

# Plot the image with bounding boxes
plot_boxes(original_image, boxes, class_names)


# In[ ]:


from google.colab.patches import cv2_imshow
# Function to detect objects in a frame
def detect_objects(frame, conf_thresh=0.4, iou_thresh=0.4):
    results = model(frame)  # Perform inference
    boxes = []

    for result in results:  # Iterate through results
        for box in result.boxes:
            # Ensure all necessary values are extracted
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0])  # Class index

            if conf > conf_thresh:  # Filter by confidence threshold
                boxes.append([x1, y1, x2, y2, conf, cls])

    if not boxes:
        return np.array([])  # Return an empty array if no boxes

    # Convert to numpy array and ensure six columns
    boxes = np.array(boxes)
    return boxes

# Function to draw bounding boxes on a frame
def draw_boxes(frame, boxes, class_names):
    for box in boxes:
        # Properly unpack the six expected elements
        x1, y1, x2, y2, conf, class_id = box
        color = (0, 255, 0) if class_names[int(class_id)] == "person" else (255, 0, 0)  # Green for players, Blue for football
        label = f"{class_names[int(class_id)]}: {conf:.2f}"
        # Convert coordinates to integers before using OpenCV
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# Process video
def process_video(video_path, output_path, conf_thresh=0.4, iou_thresh=0.4):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file.")

    # Video writer setup
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    class_names = model.names  # COCO class names
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        boxes = detect_objects(frame, conf_thresh, iou_thresh)

        # Debugging: Print the detected boxes
        print(f"Detected boxes: {boxes}")

        # Draw bounding boxes
        frame = draw_boxes(frame, boxes, class_names)

        # Write the frame to the output video
        out.write(frame)

        # # Display the frame (optional, for visualization)
        # cv2_imshow(frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        #     break

    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()

# Set input and output video paths
video_path = "football.mp4"  # Replace with your input video path
output_path = "/content/footballprocessed.mp4"  # Replace with your desired output video path

# Process the video
process_video(video_path, output_path)


# In[ ]:


# Install necessary packages
get_ipython().system('pip install ultralytics opencv-python matplotlib')

from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 model fine-tuned for traffic sign detection
model = YOLO("yolov8n.pt")
# Function to detect traffic signs in a frame
def detect_objects(frame, conf_thresh=0.4, iou_thresh=0.4):
    results = model(frame)  # Perform inference
    boxes = []

    for result in results:  # Iterate through results
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0])  # Class index

            if conf > conf_thresh:  # Filter by confidence threshold
                boxes.append([x1, y1, x2, y2, conf, cls])

    if not boxes:
        return np.array([])  # Return an empty array if no boxes

    # Convert to numpy array and ensure six columns
    boxes = np.array(boxes)
    return boxes

# Function to draw bounding boxes on a frame
def draw_boxes(frame, boxes, class_names):
    for box in boxes:
        x1, y1, x2, y2, conf, class_id = box
        color = (0, 255, 0)  # Green for detected traffic signs
        label = f"{class_names[int(class_id)]}: {conf:.2f}"
        # Convert coordinates to integers before using OpenCV
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# Process video
def process_video(video_path, output_path, conf_thresh=0.4, iou_thresh=0.4):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file.")

    # Video writer setup
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    class_names = model.names  # Traffic sign class names
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        boxes = detect_objects(frame, conf_thresh, iou_thresh)

        # Draw bounding boxes
        frame = draw_boxes(frame, boxes, class_names)

        # Write the frame to the output video
        out.write(frame)

    #     # Display the frame (optional, for visualization)
    #     cv2.imshow("Traffic Sign Detection", frame)
    #     if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
    #         break

    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()

# Set input and output video paths
video_path = "/content/Traffic.mp4"  # Replace with your input video path
output_path = "/content/traffic-sign-output.mp4"  # Replace with your desired output video path

# Process the video
process_video(video_path, output_path)

