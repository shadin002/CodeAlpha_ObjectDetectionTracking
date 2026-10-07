# CodeAlpha Object Detection and Tracking

CodeAlpha Artificial Intelligence Internship — Task 4.

This project performs real-time object detection with a pretrained YOLO model and
multi-object tracking with persistent IDs. It supports webcam input, local video
files, stream URLs, and optional output-video saving.

## Requirements Covered

- Real-time video input with OpenCV
- Pretrained YOLO object detection
- Bounding boxes and class labels
- Object tracking with persistent IDs
- Real-time display
- Optional saved output video

## Technologies

- Python
- OpenCV
- Ultralytics YOLO
- ByteTrack / BoT-SORT

## Setup

```powershell
conda create -n codealpha-vision python=3.12 -y
conda activate codealpha-vision
python -m pip install -r requirements.txt
```

## Run Tests

```powershell
python -m unittest discover -s tests -v
```

## Run Webcam

```powershell
python app.py --source 0
```

On the first run, YOLO may automatically download `yolov8n.pt`.

Press **Q** in the video window to stop.

## Run a Video File

```powershell
python app.py --source "D:\Videos\traffic.mp4"
```

## Save Processed Output

```powershell
python app.py --source 0 --save
```

Default output:

```text
output/tracked_output.mp4
```

## How It Works

1. OpenCV captures video frames.
2. YOLO detects objects and predicts bounding boxes, classes, and confidence.
3. The tracker associates detections across consecutive frames.
4. Each tracked object receives a persistent ID.
5. The program draws the bounding box, class, confidence, tracking ID, and FPS.
6. The annotated video is displayed in real time.

## GitHub Repository Name

```text
CodeAlpha_ObjectDetectionTracking
```
