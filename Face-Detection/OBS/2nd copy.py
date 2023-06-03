from typing import List
import face_recognition
import mediapipe as mp
import cv2
import numpy as np
import os


class MpDetector:
    def __init__(self):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)

    def detect(self, image, bgr=False):
        if bgr:
            image = image[:, :, ::-1]
        image_rows, image_cols, _ = image.shape
        detections = self.detector.process(image).detections
        if not detections:
            return False, []
        locations = detections[0].location_data.relative_bounding_box
        start_point = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
            locations.xmin, locations.ymin, image_cols, image_rows)
        end_point = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
            locations.xmin + locations.width, locations.ymin + locations.height, image_cols, image_rows)
        if (not start_point) or (not end_point):
            return False, []
        return True, image[start_point[1]:end_point[1], start_point[0]:end_point[0]]


def generate_embedding(croped_image, bgr=False):
    if bgr:
        croped_image = croped_image[:, :, ::-1]
    height, width, _ = croped_image.shape
    return face_recognition.face_encodings(croped_image, known_face_locations=[(0, width, height, 0)])[0]


def load_known_faces_embeddings(embeddings_path: str) -> List[np.ndarray]:
    embeddings = []
    with open(embeddings_path, 'rb') as f:
        while True:
            try:
                embeddings.append(np.load(f))
            except:
                break
    return embeddings


def main(embeddings_path: str):
    known_face_embeddings = load_known_faces_embeddings(embeddings_path)
    cap = cv2.VideoCapture(0)
    detector = MpDetector()
    while True:
        _, frame = cap.read()
        unknown_face_detecion_status, unknown_face_crop = detector.detect(frame, True)
        if unknown_face_detecion_status:
            unknown_embedding = generate_embedding(np.array(unknown_face_crop))
            distances = face_recognition.face_distance(known_face_embeddings, unknown_embedding)
            for i, distance in enumerate(distances):
                print(f"Distance from Person {i+1}: {distance}")
                if distance < 0.6:
                    cv2.putText(frame, f"Person {i+1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    break
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    embeddings_path = input("Enter the path of the saved face embeddings file: ")
    main(embeddings_path)
