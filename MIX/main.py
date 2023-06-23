import cv2
import numpy as np
import dlib
import completewrking

def main():
    # Load YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

    # Initialize variables
    count = 0
    enter_count = 0
    leave_count = 0
    status = None

    # Load face detector and landmarks predictor from dlib
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Start video capture
    cap = cv2.VideoCapture(0)  # 0 for the default webcam, change if needed

    def eye_aspect_ratio(landmarks):
        landmarks_array = np.array([[landmark.x, landmark.y] for landmark in landmarks.parts()])

        left_eye = landmarks_array[36:42]  # Extract the landmarks for the left eye
        right_eye = landmarks_array[42:48]  # Extract the landmarks for the right eye

        # Calculate eye aspect ratio for the left eye
        left_eye_width = np.linalg.norm(left_eye[0] - left_eye[3])
        left_eye_height = np.linalg.norm((left_eye[1] + left_eye[2]) / 2 - left_eye[4])
        left_ear = left_eye_height / left_eye_width

        # Calculate eye aspect ratio for the right eye
        right_eye_width = np.linalg.norm(right_eye[0] - right_eye[3])
        right_eye_height = np.linalg.norm((right_eye[1] + right_eye[2]) / 2 - right_eye[4])
        right_ear = right_eye_height / right_eye_width

        # Average the eye aspect ratios of both eyes
        ear = (left_ear + right_ear) / 2.0
        return ear



    while True:
        ret, frame = cap.read()

        # Perform object detection
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Process the detections
        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:  # Class ID 0 represents people
                    # Compute the bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    # Save the detection results
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression to eliminate redundant overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes and count people
        count = 0
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = f"Person {i+1}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                count += 1

                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector(gray)

                # Iterate over detected faces
                for face in faces:
                    # Predict face landmarks
                    landmarks = landmark_predictor(gray, face)

                    # Determine if the person is looking at the camera
                    left_eye = landmarks.part(36)
                    right_eye = landmarks.part(45)

                    # Calculate eye aspect ratio
                    ear = eye_aspect_ratio(landmarks)


                    # Check if the person is looking
                    if ear > 0.2:  # Adjust the threshold as needed
                        print("Person Detected\nInitiating.......")
                        person_name = completewrking.main1()
                        # main()
                        print(person_name)
                        exit()

        # Detect entering and leaving events
        if count > 0:
            if status == None or status == "leaving":
                enter_count += 1
                status = "entering"
                print("Person Entered")
        else:
            if status == None or status == "entering":
                leave_count += 1
                status = "leaving"
                print("Person Left")

        # Display the frame with bounding boxes and count
        cv2.putText(frame, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Enter Count: {enter_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Leave Count: {leave_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Webcam Object Detection", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()