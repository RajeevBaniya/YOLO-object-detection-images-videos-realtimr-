import numpy as np
import argparse
import imutils
import time
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.6,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


LABELS = ["person", "animal", "bird", "other objects"]


np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

COLORS[0] = [0, 255, 0]    # Green for person
COLORS[1] = [0, 0, 255]    # Red for animals
COLORS[2] = [255, 255, 0]  # Yellow for birds
COLORS[3] = [128, 128, 128]  # Gray for other objects


labelsPath = os.path.sep.join([args["yolo"], "coco.names.original"])
ORIGINAL_LABELS = open(labelsPath).read().strip().split("\n")


def map_to_simple_class(original_class_id):
    original_class = ORIGINAL_LABELS[original_class_id]
    
    if original_class == "person":
        return 0  # person
    
    # Animal category - expanded list of animals
    elif original_class in ["cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", 
                           "mouse", "rabbit", "tiger", "lion", "deer", "fox"]:
        return 1  # animal
    
    # Bird category - expanded
    elif original_class in ["bird", "duck", "eagle", "owl"]:
        return 2  # bird
    
    else:
        return 3  # other object

weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
if isinstance(net.getUnconnectedOutLayers()[0], np.ndarray):
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
else:
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]


vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)


try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))


except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

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

def draw_detection(frame, boxes, confidences, classIDs):
    output = frame.copy()
    
    class_counts = {label: 0 for label in LABELS}
    
    if len(boxes) > 0:
        for i in range(len(boxes)):
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            class_counts[LABELS[classIDs[i]]] += 1
            
            text = "{}: {:.2f}%".format(LABELS[classIDs[i]], confidences[i] * 100)
            
            # Calculate text size and position
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background rectangle for text
            cv2.rectangle(output, (x, y - text_height - 10), 
                         (x + text_width, y), color, -1)
            
            # Draw text
            cv2.putText(output, text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Calculate accuracy metrics
    class_accuracy, overall_accuracy = calculate_accuracy(confidences, classIDs)
    
    # Add class counts and accuracy to the top of the frame
    y_pos = 30
    
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

# Container for tracking average accuracy across the video
accuracy_tracker = {
    "overall": [],
    "person": [],
    "animal": [],
    "bird": [],
    "other objects": []
}

# For calculating rolling average accuracy
window_size = 30  # Average over last 30 frames

while True:
	(grabbed, frame) = vs.read()

	if not grabbed:
		break

	
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	
	boxes = []
	confidences = []
	classIDs = []

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			
			if confidence > args["confidence"]:
				
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				
				simple_class_id = map_to_simple_class(classID)

				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(simple_class_id)


	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	
	if len(idxs) > 0:
		final_boxes = []
		final_confidences = []
		final_classIDs = []
		

		for i in idxs.flatten():
			final_boxes.append(boxes[i])
			final_confidences.append(confidences[i])
			final_classIDs.append(classIDs[i])
		
		# Calculate accuracy metrics for this frame
		class_accuracy, overall_accuracy = calculate_accuracy(final_confidences, final_classIDs)
		
		# Update accuracy tracking
		accuracy_tracker["overall"].append(overall_accuracy)
		if len(accuracy_tracker["overall"]) > window_size:
			accuracy_tracker["overall"] = accuracy_tracker["overall"][-window_size:]
		
		for label in LABELS:
			if label in class_accuracy and class_accuracy[label] > 0:
				accuracy_tracker[label].append(class_accuracy[label])
				if len(accuracy_tracker[label]) > window_size:
					accuracy_tracker[label] = accuracy_tracker[label][-window_size:]
		
		# Draw detections
		output_frame = draw_detection(frame, final_boxes, final_confidences, final_classIDs)
		
		# Display rolling average accuracy
		if len(accuracy_tracker["overall"]) > 0:
			avg_acc = sum(accuracy_tracker["overall"]) / len(accuracy_tracker["overall"])
			avg_acc_text = f"Avg Accuracy: {avg_acc:.2f}%"
			cv2.putText(output_frame, avg_acc_text, (output_frame.shape[1] - 200, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
		
		cv2.imshow("Multi-Object Detection", output_frame)
		
	
		if writer is not None:
			writer.write(output_frame)
	else:
		# Handle empty detections
		cv2.putText(frame, "No objects detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.imshow("Multi-Object Detection", frame)
		
		if writer is not None:
			writer.write(frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

	
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)


		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

# Calculate and display final average accuracy
print("\n--- FINAL DETECTION ACCURACY ---")
if accuracy_tracker["overall"]:
    avg_overall = sum(accuracy_tracker["overall"]) / len(accuracy_tracker["overall"])
    print(f"Overall Average Accuracy: {avg_overall:.2f}%")

for label in LABELS:
    if accuracy_tracker[label]:
        avg_class = sum(accuracy_tracker[label]) / len(accuracy_tracker[label])
        print(f"{label} Average Accuracy: {avg_class:.2f}%")
print("-------------------------\n")

print("[INFO] cleaning up...")
writer.release()
vs.release()
cv2.destroyAllWindows()