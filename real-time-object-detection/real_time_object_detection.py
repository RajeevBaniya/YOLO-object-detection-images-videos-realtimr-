from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import os

#  argument parse and  for parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.6,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

#  class labels
LABELS = ["person", "animal", "bird", "other objects"]

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

COLORS[0] = [0, 255, 0]    # Green for person
COLORS[1] = [0, 0, 255]    # Red for animals
COLORS[2] = [255, 255, 0]  # Yellow for birds
COLORS[3] = [128, 128, 128]  # Gray for other objects

# YOLO weights and model configuration
weightsPath = os.path.sep.join(["..", "yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["..", "yolo-coco", "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

labelsPath = os.path.sep.join(["..", "yolo-coco", "coco.names.original"])
if not os.path.exists(labelsPath):
    
    labelsPath = os.path.join("yolo-coco", "coco.names.original")
    if not os.path.exists(labelsPath):
        
        ORIGINAL_LABELS = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
    else:
        ORIGINAL_LABELS = open(labelsPath).read().strip().split("\n")
else:
    ORIGINAL_LABELS = open(labelsPath).read().strip().split("\n")

# to map original COCO classes for simplified categories
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

# for the *output* layer names from YOLO
ln = net.getLayerNames()
if isinstance(net.getUnconnectedOutLayers()[0], np.ndarray):
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
else:
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

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
            
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            
            cv2.rectangle(output, (x, y - text_height - 10), 
                         (x + text_width, y), color, -1)
            
            
            cv2.putText(output, text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Calculate accuracy metrics
    class_accuracy, overall_accuracy = calculate_accuracy(confidences, classIDs)
    
    # Add class counts and accuracy to the top of the image
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
    
    # Only display FPS if the FPS counter has been started and not stopped
    try:
        fps_value = fps.fps()
        fps_text = f"FPS: {fps_value:.2f}"
        cv2.putText(output, fps_text, (10, output.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    except:
        # If FPS calculation fails, just skip it
        pass
    
    return output

# For tracking accuracy over time
accuracy_history = {
    "overall": [],
    "person": [],
    "animal": [],
    "bird": [],
    "other objects": []
}

# For calculating rolling average accuracy
window_size = 30  # Average over last 30 frames

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=800)  

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
        
        # Calculate accuracy for this frame
        class_accuracy, overall_accuracy = calculate_accuracy(final_confidences, final_classIDs)
        
        # Update accuracy history
        accuracy_history["overall"].append(overall_accuracy)
        if len(accuracy_history["overall"]) > window_size:
            accuracy_history["overall"] = accuracy_history["overall"][-window_size:]
            
        for label in LABELS:
            if label in class_accuracy and class_accuracy[label] > 0:
                accuracy_history[label].append(class_accuracy[label])
                if len(accuracy_history[label]) > window_size:
                    accuracy_history[label] = accuracy_history[label][-window_size:]
        
        output_frame = draw_detection(frame, final_boxes, final_confidences, final_classIDs)
        
        # Display rolling average accuracy
        if len(accuracy_history["overall"]) > 0:
            avg_acc = sum(accuracy_history["overall"]) / len(accuracy_history["overall"])
            avg_acc_text = f"Avg Accuracy: {avg_acc:.2f}%"
            cv2.putText(output_frame, avg_acc_text, (output_frame.shape[1] - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
        cv2.imshow("Multi-Object Detection", output_frame)
    else:
        # Handle empty detections
        cv2.putText(frame, "No objects detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Multi-Object Detection", frame)
        
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Print final accuracy statistics
print("\n--- DETECTION ACCURACY REPORT ---")
if accuracy_history["overall"]:
    avg_overall = sum(accuracy_history["overall"]) / len(accuracy_history["overall"])
    print(f"Overall Average Accuracy: {avg_overall:.2f}%")

for label in LABELS:
    if accuracy_history[label]:
        avg_acc = sum(accuracy_history[label]) / len(accuracy_history[label])
        print(f"{label} Average Accuracy: {avg_acc:.2f}%")
print("-------------------------\n")

cv2.destroyAllWindows()
vs.stop()