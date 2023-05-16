import cv2
import numpy as np

#print(cv2.__version__)

# Load YOLOv3 model and configuration files
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Set minimum confidence level and threshold for non-maximum suppression
confidence_thresh = 0.5
nms_thresh = 0.4

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set input image size and scale factor
input_size = (416, 416)
scale_factor = 1/255.0

# Open webcam
cap = cv2.VideoCapture(0)

# Loop over frames from the webcam
while True :
    
    # Capture frame from webcam and resize it to input size
    ret, frame = cap.read()
    frame = cv2.resize(frame, input_size)

    # Create blob from input image and set it as network input
    blob = cv2.dnn.blobFromImage(frame, scale_factor, input_size, swapRB=True, crop=False)
    net.setInput(blob)

    # Run forward pass to get output from the network
    output_layers = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers)

    # Initialize lists to store detected objects' information
    boxes = []
    confidences = []
    class_ids = []

    # Loop over each output layer
    for output in layer_outputs:
        # Loop over each detection
        for detection in output:
            # Extract class ID and confidence level from detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter out weak detections below confidence threshold
            if confidence > confidence_thresh:
                # Compute bounding box coordinates in the original image scale
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                center_x, center_y, width, height = box.astype("int")
                x, y = int(center_x - (width / 2)), int(center_y - (height / 2))
                
                # Save detected object's information
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform non-maximum suppression to eliminate redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, nms_thresh)

    # Loop over the detected objects and draw bounding boxes and labels
    for i in indices:
        x, y, width, height = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Check if detected object is a person and closer than 1 meter
        if classes[class_ids[i]] == 'person' and width >= 40:
            distance = 0.1651 * 120 / width
            print(distance)
            if distance < 1:
                print("Hello, person!")

    # Show the output frame
    cv2.imshow("Object Detection", frame)
    cv2.waitKey(62)
    #qcv2.destroyAllWindows()