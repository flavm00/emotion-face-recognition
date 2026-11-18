"""Real-time face recognition + emotion detection from a webcam.

Every face in the frame is matched against the reference photos in `known_faces/`
(file name = person's name) and its expression is classified into one of 7 emotions
by a CNN. Press `q` to quit.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from face_rec import FaceRec

BASE_DIR = Path(__file__).resolve().parent
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=BASE_DIR / "models" / "emotion_model.h5", type=Path)
    parser.add_argument("--faces", default=BASE_DIR / "known_faces", type=Path,
                        help="folder with one reference photo per person")
    parser.add_argument("--camera", default=0, type=int, help="webcam index")
    return parser.parse_args()


def predict_emotion(classifier, gray_face):
    """Resize a grayscale face crop to 48x48 and return the predicted emotion label."""
    roi = cv2.resize(gray_face, (48, 48), interpolation=cv2.INTER_AREA)
    roi = roi.astype("float32") / 255.0
    roi = roi.reshape(1, 48, 48, 1)
    prediction = classifier.predict(roi, verbose=0)[0]
    return EMOTION_LABELS[int(prediction.argmax())]


def main():
    args = parse_args()
    classifier = load_model(args.model)

    fr = FaceRec()
    fr.load_encoding_images(args.faces)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        face_locations, face_names = fr.detect_known_faces(frame)
        if len(face_names) == 0:
            cv2.putText(frame, "No faces", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top, left = max(top, 0), max(left, 0)
            bottom, right = min(bottom, height), min(right, width)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

            face = gray[top:bottom, left:right]
            if face.size == 0 or np.sum(face) == 0:
                continue
            label = f"{name} is {predict_emotion(classifier, face)}"
            cv2.putText(frame, label, (left, max(top - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Emotion Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
