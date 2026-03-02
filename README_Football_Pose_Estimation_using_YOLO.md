# Football Pose Estimation using YOLO26s

## Overview

This repository contains a computer vision project that performs **real-time human pose estimation** on football match footage using deep learning. The system detects all visible players in each frame and overlays a colour-coded skeleton showing 17 body keypoints per person, including joints such as shoulders, elbows, wrists, hips, knees, and ankles.

The objective of this project is to demonstrate how pose estimation models can be applied to **sports analysis and player movement tracking**. The project showcases an inference pipeline suitable for athletic performance analysis, biomechanics research, and broadcast enhancement.

This project demonstrates:
- Video processing using OpenCV
- Deep learning–based multi-person pose estimation
- Real-time keypoint detection and skeleton visualisation
- Practical sports analytics use cases

Applications include player movement analysis, fatigue detection, tactical positioning, injury prevention, and sports broadcasting overlays.

---

## Project Flow

The system follows a structured pipeline:

1. Import required libraries
2. Load pretrained pose estimation model
3. Read video input from Google Drive
4. Extract frames sequentially
5. Run pose estimation on each frame
6. Extract 17 keypoints per detected player
7. Draw colour-coded skeleton and joint markers
8. Save the annotated output video back to Google Drive

```
Input → Frame Extraction → Model Inference → Keypoint Detection → Skeleton Visualisation → Output
```

This frame-by-frame processing enables pose tracking across the full duration of match footage.

---

## Installation

Install the required dependencies:

```
pip install ultralytics opencv-python numpy
```

---

## Project Structure

```
football_pose_estimation.ipynb
football-poseEstimation.mp4
football-poseEstimation_output.mp4
README.md
```

---

## Import Libraries

```python
import cv2
import numpy as np
from ultralytics import YOLO
```

### Explanation

- **cv2** – Handles video capture, frame processing, and drawing skeletons
- **numpy** – Supports keypoint array operations and coordinate handling
- **YOLO** – Loads and runs the pose estimation model

These libraries form the core of the pose estimation pipeline.

---

## Load the Pose Estimation Model

```python
model = YOLO("yolo26s-pose.pt")
```

### Explanation

This loads the pretrained YOLO26s-pose weights trained on the COCO Keypoints dataset. The model has learned to locate 17 specific anatomical landmarks on the human body. Once loaded, it performs both person detection and keypoint localisation in a single forward pass.

---

## Video Input Processing

```python
cap = cv2.VideoCapture(VIDEO_PATH)
```

Frame extraction loop:

```python
while True:
    ret, frame = cap.read()
    if not ret:
        break
```

### Explanation

The video is processed frame-by-frame rather than loading the entire file into memory. This method:
- Reduces memory usage
- Supports processing of long match recordings
- Maintains consistent frame order for temporal analysis

This approach is standard for sports video analysis pipelines.

---

## Pose Estimation on Frames

```python
results = model(frame, conf=0.40, verbose=False)[0]
```

### Explanation

Each frame is passed through the neural network, which returns:
- Person bounding box coordinates and confidence scores
- 17 keypoint locations per detected person
- Per-keypoint confidence scores (separate from person confidence)

The confidence threshold filters out weak detections and uncertain joint positions.

---

## Accessing Keypoint Data

```python
kpts_data = results.keypoints.data   # shape: (N, 17, 3)
```

### Explanation

The keypoint tensor has shape `(N, 17, 3)` where:
- `N` = number of people detected in the frame
- `17` = one entry per body keypoint (COCO format)
- `3` = `[x_pixel, y_pixel, confidence]` per keypoint

Each joint has its own confidence score, allowing uncertain or occluded joints to be hidden independently from the overall person detection.

---

## 17 Keypoints — COCO Format

```
 0: Nose          1: Left Eye       2: Right Eye
 3: Left Ear      4: Right Ear      5: Left Shoulder
 6: Right Shoulder  7: Left Elbow   8: Right Elbow
 9: Left Wrist   10: Right Wrist   11: Left Hip
12: Right Hip    13: Left Knee     14: Right Knee
15: Left Ankle   16: Right Ankle
```

---

## Drawing the Skeleton

```python
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # Head
    (5, 6), (5, 11), (6, 12), (11, 12),       # Torso
    (5, 7), (7, 9), (6, 8), (8, 10),          # Arms
    (11, 13), (13, 15), (12, 14), (14, 16),   # Legs
]

for kA, kB in SKELETON:
    xA, yA, cA = kpts[kA]
    xB, yB, cB = kpts[kB]
    if cA >= KPT_CONF and cB >= KPT_CONF:
        cv2.line(frame, (int(xA), int(yA)), (int(xB), int(yB)), color, 2)
```

### Explanation

Skeleton connections are drawn only when both endpoints exceed the keypoint confidence threshold. This ensures that joints obscured by occlusion or fast motion during tackles and sprints are hidden rather than drawn in incorrect positions.

### Colour Coding by Body Region

| Colour | Region | Connections |
|--------|--------|-------------|
| Yellow | Head | Nose, eyes, ears |
| Green | Torso | Shoulders to hips |
| Blue | Arms | Shoulder → elbow → wrist |
| Orange | Legs | Hip → knee → ankle |

---

## Display Output

```python
if frame_idx % DISPLAY_EVERY_N == 0:
    _, buf = cv2.imencode('.jpg', frame)
    display(IPImage(data=buf.tobytes()))
```

Cleanup:

```python
cap.release()
writer.release()
```

### Explanation

A preview frame is displayed in the Colab notebook every 15 frames to monitor progress without slowing the pipeline. Proper resource cleanup ensures all frames are flushed to the output file before re-encoding.

---

## How YOLO26-Pose Works

YOLO26-pose is a single-stage model that simultaneously detects people and estimates their body keypoints in one forward pass. Unlike two-stage approaches that first detect people and then estimate pose separately, YOLO26-pose:
- Predicts bounding boxes, class labels, and 17 keypoints together
- Runs efficiently in real time on a single GPU
- Handles multiple overlapping people in the same frame

The model is pretrained on the COCO Keypoints dataset which contains over 200,000 annotated human poses across diverse activities and scenes.

---

## Real-Time Sports Pose Analysis Challenges

Applying pose estimation to football footage introduces specific challenges:

- **Fast motion** – Sprinting and kicking cause motion blur across joints
- **Player occlusion** – Tackles and group plays obscure limbs
- **Varied scale** – Players at different distances appear at different sizes
- **Crowded scenes** – Multiple overlapping players in set pieces

The per-keypoint confidence threshold handles these gracefully by suppressing uncertain joints rather than placing them incorrectly.

---

## Practical Applications

This system can be used for:

- Player movement and biomechanics analysis
- Fatigue and injury risk detection
- Tactical formation and positioning analysis
- Running gait and technique assessment
- Broadcast overlay and viewer engagement
- Referee decision support systems

With additional logic, the system can generate per-player movement statistics and flag high-risk motion patterns.

---

## Detection Pipeline (Detailed)

Each frame goes through the following stages:

1. Frame Capture from video
2. Model Inference (person detection + keypoint localisation)
3. Confidence Filtering (person-level and keypoint-level)
4. Skeleton Construction (connecting joints above confidence threshold)
5. Colour-coded Visualisation (region-specific skeleton colours)
6. HUD Overlay (player count, processing speed)
7. Frame Write to output video

This loop runs continuously for the full video duration.

---


## Author

**Manasa Vijayendra Gokak**
Graduate Student – Data Science
