from typing import List
import face_recognition
import mediapipe as mp
import cv2
import numpy as np
import os

person = {
    0 : 'Jack',
    1 : 'John',
    2 : 'Jill',
    3 : 'Ajay',
    'Unknown' : 'Unknown'
}

personlist = []

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
    # embeddings = []
    # with open(embeddings_path, 'rb') as f:
    #     while True:
    #         try:
    #             embeddings.append(np.load(f))
    #         except:
    #             break
    
    embeddings = np.load('face_embeddings.npy')
    return embeddings

def print_max_repeated_element(lst):
    # create a dictionary to count the repetitions of each element
    count_dict = {}
    for elem in lst:
        count_dict[elem] = count_dict.get(elem, 0) + 1
    
    # find the element with the maximum count
    max_count = 0
    max_elem = None
    for elem, count in count_dict.items():
        if count > max_count:
            max_count = count
            max_elem = elem
    
    # print the max repeated element if its count is greater than 9
    if max_count > 9:
        identified = max_elem
    else:
        identified = 'Unknown'
        
    return identified


def main(embeddings_path):
    known_face_embeddings = load_known_faces_embeddings(embeddings_path)
    #print(known_face_embeddings)
    cap = cv2.VideoCapture(0)
    detector = MpDetector()
    j=0
    while j < 20:
        _, image = cap.read()
        locations = face_recognition.face_locations(image)
        #print(locations)
        if len (locations) == 0 :
            j=j-1
            continue
        
        print(f"Proceeding With image {j}", end = " >>")
        unknown_face_detecion_status, unknown_face_crop = detector.detect(image, True)
        if unknown_face_detecion_status:
            unknown_embedding = generate_embedding(np.array(unknown_face_crop))
            distances = face_recognition.face_distance(known_face_embeddings, unknown_embedding)
            for i, distance in zip(range(0,len(distances)), distances):
                
                if distance < 0.8:
                    personlist.append(i)
                    print('abc',person[i])
                   
                    break
                else :
                    personlist.append('Unknown')
                    print('Unknown')
                
                    
        j = j + 1            
                    
        cv2.imshow("Webcam", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(personlist)
    identifiedPerson = print_max_repeated_element(personlist)
    print (f'\nHello {person[identifiedPerson] }')

if __name__ == "__main__":
    embeddings_path = 'face_embeddings.npy'
    main(embeddings_path)

