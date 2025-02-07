# Import necessary libraries
import os          # Provides functions to interact with the operating system, such as navigating directories
import pickle      # For saving and loading serialized Python objects, like lists and dictionaries
import mediapipe as mp  # MediaPipe is used for hand tracking and detecting hand landmarks
import cv2         # OpenCV is used for image processing (loading, converting images)
import numpy as np  # Numpy, a powerful library for numerical operations (not used in this code)

# Define the directory where the hand gesture data is stored
DATA_DIR = './data'

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
# Create an instance of the Hands model
# static_image_mode=True: Treats the input as static images (not a video stream)
# min_detection_confidence=0.3: The minimum confidence level for hand detection to be considered successful
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize lists to store the processed hand landmark data and corresponding labels (classes)
data = []     # To hold the x and y coordinates of hand landmarks
labels = []   # To hold the label index for each corresponding hand gesture

# Iterate over each folder in the data directory (each folder represents a different gesture class)
for dir_ in os.listdir(DATA_DIR):
    label_index = int(dir_)  # Convert the folder name (which is a string) to an integer to use as the label
    # Iterate over each image in the folder
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        # Read the image from the directory using OpenCV
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        # Convert the image from BGR to RGB color format, since OpenCV loads images in BGR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the RGB image to detect hand landmarks using MediaPipe
        results = hands.process(img_rgb)

        # Check if hand landmarks are detected in the image
        if results.multi_hand_landmarks:
            data_aux = []
            hand_landmarks = results.multi_hand_landmarks[0]  # Use only the first detected hand
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x)
                data_aux.append(landmark.y)
                data_aux.append(landmark.z)
            # No need for padding as we now always get 63 values (21*3)
            data.append(data_aux)
            labels.append(label_index)

# After processing all images, save the extracted data and labels into a pickle file
# This allows for later use in machine learning model training or analysis
with open('data.pickle', 'wb') as f:
    # Store the 'data' and 'labels' as a dictionary in the pickle file
    pickle.dump({'data': data, 'labels': labels}, f)
