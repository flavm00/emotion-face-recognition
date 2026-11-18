"""Face recognition helper built on the `face_recognition` library (dlib)."""
import glob
import os

import cv2
import face_recognition
import numpy as np


class FaceRec:
    def __init__(self, frame_resizing=0.25):
        self.known_face_encodings = []
        self.known_face_names = []
        # Frames are downscaled before detection for a much higher FPS.
        self.frame_resizing = frame_resizing

    def load_encoding_images(self, images_path):
        """Encode every image in `images_path`; the file name becomes the person's name."""
        paths = [p for p in glob.glob(os.path.join(str(images_path), "*.*"))
                 if p.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"{len(paths)} encoding images found.")

        for img_path in paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_img)
            if not encodings:
                print(f"  skipped {os.path.basename(img_path)}: no face found")
                continue
            name = os.path.splitext(os.path.basename(img_path))[0]
            self.known_face_encodings.append(encodings[0])
            self.known_face_names.append(name)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        """Return (face_locations, names) for all faces in a BGR frame.

        Locations are (top, right, bottom, left) in full-frame coordinates.
        """
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # OpenCV uses BGR, face_recognition expects RGB.
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            name = "Unknown"
            if self.known_face_encodings:
                # Pick the closest known face (Euclidean distance between 128-d encodings).
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best = int(np.argmin(distances))
                if matches[best]:
                    name = self.known_face_names[best]
            face_names.append(name)

        locations = (np.array(face_locations).reshape(-1, 4) / self.frame_resizing).astype(int)
        return locations, face_names
