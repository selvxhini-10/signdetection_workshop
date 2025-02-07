# Sign Language Detector
![image](https://github.com/user-attachments/assets/eadbfa9f-97d2-480c-9202-7750811c784f)


## Overview
This project demonstrates how to build a computer vision model to detect sign language gestures using **OpenCV**, **MediaPipe**, and **scikit-learn**. The model identifies hand gestures from images or video frames and maps them to corresponding sign language characters.

## Usage

1. Clone the repository. Go to the folder you want to store the repo first. 
```
git clone https://github.com/jogong2718/signdetection_workshop.git
```

2. Install python version 3.12.6 from [here](https://www.python.org/downloads/release/python-3126/). You can also use pyenv or some other package manager for more experienced devs. For this project we are using mediapipe which doesn't support python 3.13+

Verify you downloaded the version of python by running 

Windows:
```
py -0
```

Unix OS:
```
which -a python3
```

3. Then create a venv with python 3.12

Windows:
```
py -3.12 -m venv venv
```

Unix OS:
```
python3.12 -m venv venv
```

Activate your venv:

Windows:
```
venv/scripts/activate
```

Unix OS:
```
source venv/bin/activate
```

3. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the img collection file
5. Run data creation file
6. Run training file
7. Run testing file
## Steps to Build the Model

### 1. Data Collection
- **Capture Hand Gestures**: Use OpenCV and MediaPipe to capture hand gestures through the webcam.
- **Key Points Extraction**: MediaPipe is used to extract hand landmarks (key points) from each frame, representing the position and orientation of the hand.

### 2. Data Preprocessing
- **Normalize Landmarks**: Normalize the hand landmark coordinates to ensure consistency across different frames and scales.
- **Feature Engineering**: Use the extracted landmarks as features for training. These features represent the hand's posture in 3D space.
- **Label Encoding**: Assign labels to each gesture for classification (e.g., "A", "B", "C" for sign language letters).

### 3. Model Training
- **Choose Classifier**: Use scikit-learn to build a classifier (e.g., Random Forest, Support Vector Machine) to classify the hand gestures based on the extracted features.
- **Train the Model**: Feed the normalized hand landmarks and corresponding labels into the model for training.

### 4. Model Testing
- **Accuracy**: Evaluate the model's accuracy on a validation dataset.

