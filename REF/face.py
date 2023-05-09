from sys import argv
import face_recognition
import mediapipe as mp
import cv2
import numpy as np


class MpDetector : 
    def __init__(self) -> None:           
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.5 
        )

    def detect(self, image, bgr=False):
        '''
        param: image: cv2 object of rgb format
        '''
        if bgr:
            # if in bgr format convert to rgb 
            image = image[:, :, ::-1]  
        
        image_rows, image_cols, _ = image.shape

        detections = self.detector.process(image).detections

        if not detections:
            return False, []

        locations = detections[0].location_data.relative_bounding_box
        start_point = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(locations.xmin, locations.ymin, image_cols, image_rows)
        end_point = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
            locations.xmin + locations.width, locations.ymin + locations.height, 
            image_cols, image_rows
        )
        if (not start_point) or (not end_point):
            return False, []

        return True, image[start_point[1]:end_point[1], start_point[0]:end_point[0]]


def generate_embedding(croped_image, bgr=False):
    if bgr:
        # if in bgr format convert to rgb 
        croped_image = croped_image[:, :, ::-1]
    height, width, _ = croped_image.shape
    return face_recognition.face_encodings(croped_image, known_face_locations=[(0, width, height, 0)])[0]


def main(known_imgs:list, unknown_face:str):

    # 1st step detecting faces using "mediapipe" (more accurate and snapy)

    detector = MpDetector()
    
    known_face_embeddings = []
    for i, im in enumerate(known_imgs):
        im = cv2.imread(im)
        face_detection_status, face_crop = detector.detect(im, True)

        if not face_detection_status :
            # face not detected in either first image or in second image
            print(f"No face detected in {i+1}th  known image")
            exit(1)
        
        # if face is present calculate the face embeddings using "face_recognition"
        emb1 = generate_embedding(np.array(face_crop))
        known_face_embeddings.append(emb1)

    # calculating embeddings for unknonw face
    unknown_face = cv2.imread(unknown_face)
    unknown_face_detecion_status, unknown_face_crop =  detector.detect(unknown_face, True)
    if not unknown_face_detecion_status:
        print("No face found on unkown face image")
        exit(1)
    # calculating unkwon face embeddings
    unknown_embedding = generate_embedding(np.array(unknown_face_crop))

    # here the first parameter is a list of embeddings. of knwon faces, and the second one is 
    # the embedding of unknown face
    distances = face_recognition.face_distance(known_face_embeddings, unknown_embedding)
    
    for i, distance in enumerate(distances):
        print(f"\ndistance of unknown image from, known image {i+1}  : {distance} ")
        if distance < 0.6:
            # faces are matching
            print(f"Matching faces")
        else:
            print(f"Not matching faces")

    return

if __name__=="__main__":
    if len(argv)>2:
        main(argv[2:], argv[1])
    else:
        print(f"usage :\n\tpython3 face.py <path-to-unknown-image> <path-to-known-image1> \
<path-to-known-image2> <path-to-known-image3> ...")
        exit(1)

