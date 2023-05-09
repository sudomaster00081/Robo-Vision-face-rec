import mediapipe as mp
import cv2
import numpy as np
import os
import face_recognition

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def generate_embeddings(image_folder):
    face_embeddings = []
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        for filename in os.listdir(image_folder):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            image_rows, image_cols, _ = image.shape
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.detections:
                for detection in results.detections:
                    location = detection.location_data.relative_bounding_box
                    start_point = mp_drawing._normalized_to_pixel_coordinates(location.xmin, location.ymin, image_cols, image_rows)
                    end_point = mp_drawing._normalized_to_pixel_coordinates(location.xmin + location.width, location.ymin + location.height, image_cols, image_rows)
                    if (not start_point) or (not end_point):
                        continue
                    face_crop = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
                    embedding = face_recognition.face_encodings(face_crop)[0]
                    face_embeddings.append(embedding)
    face_embeddings = np.array(face_embeddings)
    np.save("face_embeddings.npy", face_embeddings)


if __name__ == "__main__":
    image_folder = "images/"
    generate_embeddings(image_folder)
