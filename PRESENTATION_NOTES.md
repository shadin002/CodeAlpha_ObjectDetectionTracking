# Presentation Notes

## Short explanation

This is my CodeAlpha AI Internship Task 4 project for Object Detection and Tracking.

OpenCV captures frames from a webcam or video file. A pretrained YOLO model detects
objects in each frame. Then ByteTrack associates detections between consecutive
frames and assigns tracking IDs. The application displays bounding boxes, object
classes, confidence scores, tracking IDs, and FPS in real time. It can also save
the processed video.

## Viva points

- Detection finds objects in each frame.
- Tracking associates the same object across frames.
- YOLO is used because it is fast and suitable for real-time detection.
- The project uses pretrained weights, so no detector training is required.
- A tracking ID helps preserve object identity while it moves through the scene.
