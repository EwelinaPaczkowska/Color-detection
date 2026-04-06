# Red Object Detection and Tracking

## Description
This project implements a computer vision system for detecting and tracking a red object in a video stream. The solution uses color segmentation in the HSV color space combined with morphological operations to improve detection quality.

The system identifies the object, tracks it frame-by-frame, and calculates its horizontal deviation from the center of the frame.

## Features
- Red color detection using HSV segmentation
- Noise removal using morphological operations (opening, closing)
- Object tracking using contour analysis and moments
- Visualization of detected object (bounding circle)
- Real-time video processing
- Display of object position and deviation from frame center
- Dual view: original video and processed mask

## Technologies
- Python
- OpenCV
- NumPy

## How it works
1. The video is loaded frame by frame.
2. Each frame is converted to HSV color space.
3. A mask is created to detect red color.
4. Morphological operations are applied to remove noise and fill gaps.
5. The largest contour is selected as the target object.
6. The object is marked with a circle.
7. The horizontal deviation from the center is calculated and displayed.

## Usage
Run the script with a video file:

```bash
python lab1_object_detection.py --video your_video.mp4
