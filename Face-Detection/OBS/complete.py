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
            return False, None, None, None
        locations = detections[0].location_data.relative_bounding_box
        start_point = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
            locations.xmin, locations.ymin, image_cols, image_rows)
        end_point = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
            locations.xmin + locations.width, locations.ymin + locations.height, image_cols, image_rows)
        if (not start_point) or (not end_point):
            return False, None, None, None
        return True, image[start_point[1]:end_point[1], start_point[0]:end_point[0]], start_point[0], start_point[1]


def generate_embedding(cropped_image, bgr=False):
    if bgr:
        cropped_image = cropped_image[:, :, ::-1]
    height, width, _ = cropped_image.shape
    return face_recognition.face_encodings(cropped_image, known_face_locations=[(0, width, height, 0)])[0]


def load_known_faces():
    known_face_embeddings = np.load("face_embeddings.npy")
    return known_face_embeddings


def identify_faces(known_face_embeddings, image):
    detector = MpDetector()
    face_detection_status, face_crop, start_x, start_y = detector.detect(image, True)
    if face_detection_status:
        current_face_embedding = generate_embedding(np.array(face_crop))
        face_distances = face_recognition.face_distance(known_face_embeddings, current_face_embedding)
        min_distance_index = np.argmin(face_distances)
        min_distance = face_distances[min_distance_index]
        if min_distance < 0.55:
            # Face recognized as a known person
            return True, min_distance_index, start_x, start_y
        else:
            # Unknown face
            print ("Unknown")
            return False, None, None, None
    else:
        # No face detected
        print("Noface")
        return False, None, None, None


# leaving_threshold = 100  # Adjust the threshold value as per your requirement
# entering_threshold = 100  # Adjust the threshold value as per your requirement


# def is_leaving(prev_x, prev_y, curr_x, curr_y, threshold):
#     if prev_x is not None and prev_y is not None and curr_x is not None and curr_y is not None:
#         distance = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
#         if distance > threshold:
#             return True
#     return False


# def is_entering(prev_x, prev_y, curr_x, curr_y, threshold):
#     if prev_x is not None and prev_y is not None and curr_x is not None and curr_y is not None:
#         distance = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
#         if distance > threshold:
#             return True
#     return False


def main():
    known_face_embeddings = load_known_faces()
    cap = cv2.VideoCapture(0)

    prev_x, prev_y = None, None

    # Define the names or identifiers of the known people
    known_people = ["Jack", "John", "Christy", "Ajay"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection using YOLO algorithm
        # ...

        # Perform face recognition and identification
        face_recognition_status, person_index, start_x, start_y = identify_faces(known_face_embeddings, frame)

        if face_recognition_status:
            # Known person recognized
            print("Person identified:", known_people[person_index])
            # You can perform further actions here, such as displaying the person's name, etc.

        # if is_leaving(prev_x, prev_y, start_x, start_y, leaving_threshold):
        #     if face_recognition_status:
        #         # Known person leaving
        #         print("Person is leaving:", known_people[person_index])
        #     else:
        #         # Unknown person leaving
        #         print("Unknown person is leaving!")
        #     # Perform actions when a person is leaving from the frame

        # if is_entering(prev_x, prev_y, start_x, start_y, entering_threshold):
        #     if not face_recognition_status:
        #         # Person entering
        #         print("Person is entering!")
            # Perform actions when a person is entering the frame

        prev_x, prev_y = start_x, start_y

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

print("\nSUCCESSFUL 🤣")
