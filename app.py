from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.utils import build_label, parse_source, validate_file_source


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time YOLO object detection and multi-object tracking."
    )
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
    )
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output", default="output/tracked_output.mp4")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def draw_tracking_results(frame, result, class_names, fps_value: float):
    annotated = frame.copy()
    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = [None] * len(xyxy)

        for box, cls_id, conf, track_id in zip(
            xyxy, classes, confidences, track_ids
        ):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names.get(cls_id, str(cls_id))
            label = build_label(track_id, class_name, float(conf))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            text_y = max(y1, text_h + 8)
            cv2.rectangle(
                annotated,
                (x1, text_y - text_h - 8),
                (x1 + text_w + 8, text_y + baseline),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, text_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

    cv2.putText(
        annotated,
        f"FPS: {fps_value:.1f}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        annotated,
        "Press Q to quit",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


def create_writer(output_path: str, fps: float, width: int, height: int):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fps <= 0 or fps > 240:
        fps = 30.0

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")

    return writer


def run(args):
    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("--confidence must be greater than 0 and at most 1.")
    if not 0.0 < args.iou <= 1.0:
        raise ValueError("--iou must be greater than 0 and at most 1.")

    source = parse_source(args.source)
    validate_file_source(source)

    print(f"Loading YOLO model: {args.model}")
    print("The first run may download the model weights automatically.")
    model = YOLO(args.model)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video source: {source}. "
            "If using a webcam, close other apps that may be using it."
        )

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))

    if width <= 0 or height <= 0:
        width, height = 640, 480

    writer = None
    if args.save:
        writer = create_writer(args.output, source_fps, width, height)
        print(f"Saving processed video to: {args.output}")

    previous_time = time.perf_counter()
    smoothed_fps = 0.0

    print("Detection + tracking started.")
    if not args.no_display:
        print("Press Q in the video window to stop.")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Video ended or a frame could not be read.")
                break

            results = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                conf=args.confidence,
                iou=args.iou,
                verbose=False,
            )

            current_time = time.perf_counter()
            elapsed = max(current_time - previous_time, 1e-6)
            instant_fps = 1.0 / elapsed
            previous_time = current_time
            smoothed_fps = (
                instant_fps if smoothed_fps == 0.0
                else 0.90 * smoothed_fps + 0.10 * instant_fps
            )

            annotated = draw_tracking_results(
                frame, results[0], model.names, smoothed_fps
            )

            if writer is not None:
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow("CodeAlpha - Object Detection and Tracking", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print("Finished successfully.")


def main():
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
