# Football Pose Tracking using MediaPipe

## Overview

This repository contains a computer vision project that performs **real-time human pose estimation** on football match footage using Google's **MediaPipe BlazePose** via the new Tasks API. The system detects all visible players in each frame and overlays a colour-coded skeleton showing 33 body landmarks per person, including detailed joints for the hands, feet, and face.

The objective of this project is to demonstrate how pose estimation can be applied to **sports analysis and player movement tracking** using a lightweight, cross-platform framework. The project showcases a two-model pipeline — YOLO26n for person detection and MediaPipe BlazePose for per-player landmark estimation.

This project demonstrates:
- Video processing using OpenCV
- Multi-person pose estimation using a two-model pipeline
- 33-landmark body keypoint detection with visibility filtering
- Colour-coded skeleton visualisation by body region
- Practical sports analytics use cases

Applications include player movement analysis, posture assessment, tactical positioning, biomechanics research, and broadcast overlay.

---

## Project Flow

The system follows a structured pipeline:

1. Install dependencies and download the BlazePose model file
2. Mount Google Drive
3. Define input and output paths
4. Load YOLO26n person detector and MediaPipe PoseLandmarker
5. Set detection thresholds and skeleton parameters
6. Define the coordinate remapping helper function
7. Run the main loop — detect players, estimate pose, draw skeleton
8. Re-encode output to H.264 and save to Google Drive
9. Play the output video inline in the notebook

```
Input → Frame Extraction → YOLO Person Detection → Crop + Pad Each Player
      → MediaPipe Pose per Crop → Coordinate Remap → Skeleton Draw → Output
```

---

## Installation

Install the required dependencies:

```
pip install mediapipe ultralytics opencv-python
```

Download the BlazePose model file:

```
wget -O pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
```

---

## Project Structure

```
PoseTracking_MediaPipe.ipynb
pose_landmarker.task
football-poseEstimation.mp4
football-mediapipe_output.mp4
README.md
```

---

## Import Libraries

```python
import mediapipe as mp
import cv2
import numpy as np
from ultralytics import YOLO
```

### Explanation

- **mediapipe** – Google's framework for running BlazePose pose estimation
- **cv2** – Handles video capture, frame processing, and skeleton drawing
- **numpy** – Supports array operations and coordinate handling
- **YOLO** – Loads YOLO26n for detecting all players in each frame

MediaPipe alone only finds one person per frame. YOLO provides bounding boxes for every player so MediaPipe can run on each one individually.

---

## Load the Models

```python
# MediaPipe PoseLandmarker — Tasks API
BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options                  = BaseOptions(model_asset_path=MODEL_PATH),
    running_mode                  = VisionRunningMode.IMAGE,
    num_poses                     = 1,
    min_pose_detection_confidence = 0.5,
    min_pose_presence_confidence  = 0.5,
    output_segmentation_masks     = False
)
pose_landmarker = PoseLandmarker.create_from_options(options)

# YOLO26n person detector
yolo_model = YOLO('yolo26n.pt')
```

### Explanation

Two models are loaded. YOLO26n is the nano detection model — its only job is returning bounding boxes around each player in the frame. MediaPipe PoseLandmarker is the pose model — it finds 33 body landmarks on a single person crop.

`VisionRunningMode.IMAGE` is used instead of VIDEO mode. IMAGE mode treats each crop as a fully independent image with no temporal tracking between frames. This is correct for multi-person use because each crop is a different player — sharing a tracking state between them would cause skeletons to drift off the wrong person's body.

The model is loaded from a `.task` file which is a pre-packaged format specific to the new MediaPipe Tasks API. The old `mp.solutions.pose.Pose()` approach no longer works in recent versions of MediaPipe.

---

## Video Input Processing

```python
cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break
```

### Explanation

The video is processed frame-by-frame. For each frame YOLO first detects all players, then MediaPipe runs on each player crop individually. This approach keeps memory usage low and works for any video length.

---

## Person Detection and Cropping

```python
yolo_results = yolo_model(frame, conf=PERSON_CONF, classes=[PERSON_CLASS_ID], verbose=False)[0]

for box in yolo_results.boxes:
    bx1, by1, bx2, by2 = map(int, box.xyxy[0])

    pad_x = int((bx2 - bx1) * PAD_FRAC)
    pad_y = int((by2 - by1) * PAD_FRAC)
    cx1 = max(0, bx1 - pad_x)
    cy1 = max(0, by1 - pad_y)
    cx2 = min(W, bx2 + pad_x)
    cy2 = min(H, by2 + pad_y)

    crop = frame[cy1:cy2, cx1:cx2]
```

### Explanation

YOLO returns tight bounding boxes around each person. A padding of 12% is added on each side before cropping. This gives MediaPipe surrounding context beyond the tight body boundary — without it, landmarks near the edges such as raised hands or extended feet get cut off or placed incorrectly. `max(0, ...)` and `min(W, ...)` clamp the padded coordinates so they never exceed the frame boundaries.

---

## Pose Estimation on Each Crop

```python
crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
mp_result = pose_landmarker.detect(mp_image)
```

### Explanation

Each player crop is converted from BGR to RGB because MediaPipe requires RGB input while OpenCV uses BGR. The crop is then wrapped in a `mp.Image` object which is the input format required by the Tasks API. `detect()` runs the full BlazePose pipeline on the crop and returns 33 landmarks per detected person, each with normalised `x`, `y`, `z` coordinates and a visibility score.

---

## Coordinate Remapping Helper

```python
def draw_pose_on_frame(frame, pose_landmarks_list, x1, y1, x2, y2):
    crop_w = x2 - x1
    crop_h = y2 - y1
    landmarks = pose_landmarks_list[0]

    pts = []
    for lm in landmarks:
        px = int(lm.x * crop_w) + x1
        py = int(lm.y * crop_h) + y1
        pts.append((px, py, lm.visibility))
```

### Explanation

MediaPipe returns landmarks normalised to the crop — `x=0.5` means the centre of the crop, not the centre of the full frame. To draw the skeleton correctly on the full video frame, each landmark must be remapped. Multiplying `lm.x` by the crop width gives the pixel position within the crop. Adding `x1` and `y1` (the crop's top-left corner in the full frame) shifts it to the correct position on the original video frame. Without this step every skeleton would be drawn in the top-left region of the frame regardless of where the player actually is.

---

## Drawing the Skeleton

```python
SKELETON_GROUPS = {
    'face':   [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10)],
    'torso':  [(11,12),(11,23),(12,24),(23,24)],
    'l_arm':  [(11,13),(13,15),(15,17),(15,19),(15,21),(17,19)],
    'r_arm':  [(12,14),(14,16),(16,18),(16,20),(16,22),(18,20)],
    'l_leg':  [(23,25),(25,27),(27,29),(27,31),(29,31)],
    'r_leg':  [(24,26),(26,28),(28,30),(28,32),(30,32)],
}

for (a, b) in connections:
    if pts[a][2] >= VIS_THRESHOLD and pts[b][2] >= VIS_THRESHOLD:
        cv2.line(frame, pts[a][:2], pts[b][:2], color, SKEL_THICKNESS, cv2.LINE_AA)
```

### Explanation

Skeleton connections are grouped by body region so each region can be drawn in a different colour. A connection is only drawn if both its endpoint landmarks have a visibility score above `VIS_THRESHOLD`. This ensures joints that are occluded during tackles or fast movement are hidden rather than drawn in incorrect positions.

### Colour Coding by Body Region

| Colour | Region | Connections |
|--------|--------|-------------|
| Yellow | Face | Nose, eyes, ears, mouth |
| Green | Torso | Shoulders to hips |
| Blue (dark) | Left arm | Shoulder → elbow → wrist → fingers |
| Blue (light) | Right arm | Shoulder → elbow → wrist → fingers |
| Orange (dark) | Left leg | Hip → knee → ankle → heel → toe |
| Orange (light) | Right leg | Hip → knee → ankle → heel → toe |

---

## 33 Landmarks — BlazePose Format

```
 0: Nose            1: L.Eye(inner)    2: L.Eye         3: L.Eye(outer)
 4: R.Eye(inner)    5: R.Eye           6: R.Eye(outer)  7: L.Ear
 8: R.Ear           9: Mouth(L)       10: Mouth(R)     11: L.Shoulder
12: R.Shoulder     13: L.Elbow        14: R.Elbow      15: L.Wrist
16: R.Wrist        17: L.Pinky        18: R.Pinky      19: L.Index
20: R.Index        21: L.Thumb        22: R.Thumb      23: L.Hip
24: R.Hip          25: L.Knee         26: R.Knee       27: L.Ankle
28: R.Ankle        29: L.Heel         30: R.Heel       31: L.Foot
32: R.Foot
```

BlazePose provides 33 landmarks compared to the 17 in the standard COCO format used by YOLO-pose. The extra landmarks cover inner and outer eye corners, individual fingers (pinky, index, thumb), heel, and toe — useful for detailed hand and foot analysis.

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
pose_landmarker.close()
```

### Explanation

A preview frame is displayed in the Colab notebook every 15 frames to monitor progress without slowing the pipeline. `pose_landmarker.close()` must be called at the end to release the MediaPipe graph resources cleanly.

---

## How MediaPipe BlazePose Works

MediaPipe BlazePose uses a two-step detector-tracker pipeline. A lightweight person detector first locates the person and establishes a region of interest. A second landmark model then predicts all 33 keypoints within that region at high speed. The landmark model outputs `x`, `y`, `z` coordinates and a visibility score per joint — visibility indicates confidence that the joint is actually visible and not occluded.

In this notebook, the detection step is handled externally by YOLO26n, so MediaPipe only runs the landmark model on each pre-cropped player region.

---

## Author

**Manasa Vijayendra Gokak**
Graduate Student – Data Science
