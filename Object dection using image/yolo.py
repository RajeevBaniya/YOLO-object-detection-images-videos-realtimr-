import numpy as np
import argparse
import time
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.6,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# Define our simplified class labels
LABELS = ["person", "animal", "bird", "other objects"]

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

COLORS[0] = [0, 255, 0]    # Green for person
COLORS[1] = [0, 0, 255]    # Red for animals
COLORS[2] = [255, 255, 0]  # Yellow for birds
COLORS[3] = [128, 128, 128]  # Gray for other objects

# Load the original COCO class labels for mapping
labelsPath = '../yolo-coco/coco.names.original'
ORIGINAL_LABELS = open(labelsPath).read().strip().split("\n")

# Function to map original COCO classes to our simplified categories
def map_to_simple_class(original_class_id):
    original_class = ORIGINAL_LABELS[original_class_id]
    
    # Person category
    if original_class == "person":
        return 0  # person
    
    # Animal category - expanded list of animals
    elif original_class in ["cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", 
                           "mouse", "rabbit", "tiger", "lion", "deer", "fox"]:
        return 1  # animal
    
    # Bird category - expanded
    elif original_class in ["bird", "duck", "eagle", "owl"]:
        return 2  # bird
    
    # Everything else is "other object"
    else:
        return 3  # other object

# derive the paths to the YOLO weights and model configuration
weightsPath = '../yolo-coco/yolov3.weights'
configPath = '../yolo-coco/yolov3.cfg'

# load our YOLO object detector trained on COCO dataset
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load our input image and grab its spatial dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
# Fix for compatibility with different OpenCV versions
if isinstance(net.getUnconnectedOutLayers()[0], np.ndarray):
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
else:
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs
for output in layerOutputs:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"]:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# Map the original COCO class to our simplified categories
			simple_class_id = map_to_simple_class(classID)

			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(simple_class_id)

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

# Function to calculate accuracy metrics
def calculate_accuracy(confidences, classIDs):
    class_confidences = {label: [] for label in LABELS}
    
    for i in range(len(classIDs)):
        class_label = LABELS[classIDs[i]]
        class_confidences[class_label].append(confidences[i])
    
    # Calculate average confidence for each class
    class_accuracy = {}
    for label, confs in class_confidences.items():
        if confs:
            avg_conf = sum(confs) / len(confs)
            class_accuracy[label] = avg_conf * 100  # Convert to percentage
        else:
            class_accuracy[label] = 0
    
    # Calculate overall accuracy (average of all confidences)
    all_confs = confidences if confidences else [0]
    overall_accuracy = sum(all_confs) / len(all_confs) * 100
    
    return class_accuracy, overall_accuracy

# Function to draw detection results with improved visualization
def draw_detection(image, boxes, confidences, classIDs):
    # Create a copy of the image to draw on
    output = image.copy()
    
    # Initialize counters for each class
    class_counts = {label: 0 for label in LABELS}
    
    # ensure at least one detection exists
    if len(boxes) > 0:
        # loop over the detections
        for i in range(len(boxes)):
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            # Update class count
            class_counts[LABELS[classIDs[i]]] += 1
            
            # Prepare text with class name and confidence
            text = "{}: {:.2f}%".format(LABELS[classIDs[i]], confidences[i] * 100)
            
            # Calculate text size and position
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background rectangle for text
            cv2.rectangle(output, (x, y - text_height - 10), 
                         (x + text_width, y), color, -1)
            
            # Draw text
            cv2.putText(output, text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add class counts and accuracy to the top of the image
    y_pos = 30
    
    # Calculate accuracy metrics
    class_accuracy, overall_accuracy = calculate_accuracy(confidences, classIDs)
    
    # Display overall accuracy
    acc_text = f"Overall Accuracy: {overall_accuracy:.2f}%"
    cv2.putText(output, acc_text, (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    y_pos += 30
    
    # Display class counts and accuracies
    for label, count in class_counts.items():
        if count > 0:
            accuracy = class_accuracy.get(label, 0)
            count_text = f"{label}: {count} (Acc: {accuracy:.2f}%)"
            cv2.putText(output, count_text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            y_pos += 30
    
    return output

# After detection and NMS, replace the drawing code with:
if len(idxs) > 0:
    # Create lists for the detections we're keeping
    final_boxes = []
    final_confidences = []
    final_classIDs = []
    
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        final_boxes.append(boxes[i])
        final_confidences.append(confidences[i])
        final_classIDs.append(classIDs[i])
    
    # Draw all detections
    output_image = draw_detection(image, final_boxes, final_confidences, final_classIDs)
    
    # Calculate and print accuracy metrics
    class_accuracy, overall_accuracy = calculate_accuracy(final_confidences, final_classIDs)
    print("\n--- DETECTION ACCURACY ---")
    print(f"Overall Detection Accuracy: {overall_accuracy:.2f}%")
    for label, accuracy in class_accuracy.items():
        if accuracy > 0:
            print(f"{label} Accuracy: {accuracy:.2f}%")
    print("-------------------------\n")
    
    # show the output image
    cv2.imshow("Multi-Object Detection", output_image)
    cv2.waitKey(0)
else:
    print("No objects detected with the current confidence threshold.")
    # Display empty image with message
    cv2.putText(image, "No objects detected", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Multi-Object Detection", image)
    cv2.waitKey(0)