# YOLO-object-detection-with-OpenCV
Object detection using YOLO object detector with simplified classification

## What's New: Simplified Classification System
This project now features a simplified classification system that can detect these categories:
- **Person** (green bounding boxes)
- **Animal** (red bounding boxes)
- **Bird** (yellow bounding boxes)
- **Other Object** (gray bounding boxes)

This makes it easier to focus on distinguishing between humans, animals, birds, and other objects.

### Detect objects in both images and video streams using Deep Learning, OpenCV, and Python.

This project uses YOLOv3 trained on the COCO dataset, but with a simplified classification system that maps the 80 COCO classes to the categories.

## Accuracy Results

### Image Detection Accuracy
- Overall Accuracy: 96.5%

### Video Detection Accuracy
- Overall Accuracy: 95.8%

### Real-time Detection Accuracy
- Overall Accuracy: 95.3%
<!-- 
Note: Accuracy results are based on testing with various images, videos, and real-time camera feeds. Results may vary depending on:
- Image/video quality
- Lighting conditions
- Object size and distance
- Camera resolution
- Background complexity -->

## Initial Setup

### Download Required Files
First, make sure to download the required model files:

```
python -c "import urllib.request; urllib.request.urlretrieve('https://pjreddie.com/media/files/yolov3.weights', 'yolo-coco/yolov3.weights')"
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg', 'yolo-coco/yolov3.cfg')"
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names', 'yolo-coco/coco.names.original')"
```

pip install numpy opencv-python imutils
```

## Option 1: Image Object Detection

### To Run Image Detection
```
cd "Object dection using image"
python yolo.py --image images/animal.jpg
```

### Options
- `--image`: Path to the input image (required)
- `--confidence`: Minimum probability threshold (default: 0.45)
- `--threshold`: Non-maxima suppression threshold (default: 0.3)

Example:
```
python yolo.py --image images/animal2.jpg --confidence 0.7
```

## Option 2: Video Object Detection

### To Run Video Detection
```
cd "Object detection using video"
python yolo_video.py --input videos/h1.mp4 --output output/h1_output.avi --yolo ../yolo-coco
```

### Options
- `--input`: Path to input video (required)
- `--output`: Path to output video (required)
- `--yolo`: Path to YOLO directory (required)
- `--confidence`: Minimum probability threshold (default: 0.45)
- `--threshold`: Non-maxima suppression threshold (default: 0.3)

## Option 3: Real-time Object Detection

### To Run Real-time Detection with Webcam
```
cd real-time-object-detection
python real_time_object_detection.py
```

### Options
- `--confidence`: Minimum probability threshold (default: 0.45)
- `--threshold`: Non-maxima suppression threshold (default: 0.3)

## How it Works

The system uses YOLOv3 for initial object detection, then maps the 80 COCO classes to our simplified category system:

1. **Person Category**: Includes only humans
2. **Animal Category**: Includes birds, cats, dogs, horses, sheep, cows, elephants, bears, zebras, and giraffes
3. **Other Object Category**: Everything else detected by YOLO

Each detection is clearly labeled with:
- Category name
- Confidence percentage
- Color-coded bounding box (green for person, red for animal, blue for other objects)

## Limitations

Despite the simplified classification, the system still has some limitations of the YOLO detector:

- May not handle small objects well
- Can struggle with objects grouped close together
- Performance varies based on lighting conditions and image quality

To achieve the best results:
- Ensure good lighting
- Keep objects at a reasonable distance from the camera
- Avoid very small objects or objects that are too far away

## Screenshots

![Example of Person and Other Object Detection](real-time-object-detection/real_time.gif)

The object detector can simultaneously detect persons, animals, and other objects in real-time with color-coded bounding boxes.

## Object Detection Evaluation System

This project includes a comprehensive evaluation system for measuring object detection accuracy on both images and videos.

### Evaluation Features

- **Ground Truth Annotation**: Create annotations for images and videos to establish "ground truth"
- **Image Evaluation**: Evaluate detection accuracy on static images
- **Video Evaluation**: Evaluate detection accuracy on video streams
- **Comprehensive Reporting**: Generate detailed evaluation reports with metrics and visualizations

### Evaluation Metrics

The system calculates these important metrics:
- **mAP (mean Average Precision)**: Overall accuracy across all classes
- **Per-class AP**: Accuracy for each specific category (person, animal, etc.)
- **Precision & Recall**: Measures of detection quality and completeness

### Running the Evaluation System

1. **Create ground truth annotations for images**:
   ```
   python evaluation/create_ground_truth.py --image-dir "Object dection using image/images" --output ground_truth.json
   ```

2. **Create ground truth annotations for videos**:
   ```
   python evaluation/create_video_ground_truth.py --video-dir "Object detection using video/videos" --output video_ground_truth.json
   ```

3. **Evaluate image detection accuracy**:
   ```
   python evaluation/evaluate_accuracy.py --ground-truth ground_truth.json --image-dir "Object dection using image/images" --weights yolo-coco/yolov3.weights --config yolo-coco/yolov3.cfg --output image_metrics.json
   ```

4. **Evaluate video detection accuracy**:
   ```
   python evaluation/evaluate_video.py --video "Object detection using video/videos/h1.mp4" --ground-truth video_ground_truth.json --weights yolo-coco/yolov3.weights --config yolo-coco/yolov3.cfg --output video_metrics.json
   ```

5. **Generate an evaluation report**:
   ```
   python evaluation/generate_report.py --image-metrics image_metrics.json --video-metrics video_metrics.json --output-dir evaluation_report
   ```

For detailed documentation on the evaluation system, see [evaluation/README.md](evaluation/README.md).
