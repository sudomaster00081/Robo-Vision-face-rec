import cv2
import numpy as np
import math

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set up webcam
cap = cv2.VideoCapture(0)

def calculate_distance(focal_length, known_width, pixel_width):
    return (known_width * focal_length) / pixel_width

def detect_person(frame):
    height, width, channels = frame.shape

    # Preprocess frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Process detected objects
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == "person":
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate bounding box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-max suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    person_detected = False

    # Draw bounding boxes and print messages
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"Person {i + 1}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            person_detected = True

    return frame, person_detected


def start_detection():
    # Set known width and focal length for distance calculation
    known_width = 0.5  # Known width of the person (in meters)
    focal_length = 615  # Focal length of the camera (in pixels)

    while True:
        # Read frame from webcam
        ret, frame = cap.read()

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Detect person and calculate distance
        output_frame, person_detected = detect_person(frame)

        # Display distance message
        if person_detected:
            focal_pixel_width = output_frame.shape[1]  # Assuming the width of the frame is the focal pixel width
            distance = calculate_distance(focal_length, known_width, focal_pixel_width)
            if distance < 1:
                cv2.putText(output_frame, "You are close!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(output_frame, "Hello, person!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Live Stream", output_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


# Start detection
start_detection()
