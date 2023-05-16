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


def capture_known_faces() -> List[np.ndarray]:
    
    known_face_embeddings = []
    detector = MpDetector()
    i = 0
    for filename in os.listdir('images'):
        print(f'loading Face {i}')
        image = cv2.imread(os.path.join('images', filename))
        face_detection_status, face_crop = detector.detect(image, True)
        if face_detection_status:
            cv2.imshow(f"Person {i+1}", face_crop)
            #cv2.waitKey(0)
            emb = generate_embedding(np.array(face_crop))
            known_face_embeddings.append(emb)
            print(f'DONE loading Face {i}')
            i += 1

            #input("Press Enter to continue...")

    cv2.destroyAllWindows()
    return known_face_embeddings


def main():
    known_face_embeddings = capture_known_faces()
    
    #print(known_face_embeddings)
    np.save("face_embeddings.npy", known_face_embeddings)
    cap = cv2.VideoCapture(0)
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

print("\nSUCCESSFULL 🤣")
