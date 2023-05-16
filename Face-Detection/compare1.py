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
        if locations:
            start_point = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
                locations.xmin, locations.ymin, image_cols, image_rows)
            end_point = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
                locations.xmin + locations.width, locations.ymin + locations.height, image_cols, image_rows)
            if (not start_point) or (not end_point):
                return False, []
            return True, image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        else:
            return False, []
