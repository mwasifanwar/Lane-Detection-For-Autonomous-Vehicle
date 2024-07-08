This repository contains Python code to detect lane lines in images and videos using OpenCV and Python. Lane detection is a critical component in many autonomous vehicle systems and can be applied in various other domains where understanding lane positioning is important.

Requirements
Python 3.x
OpenCV (pip install opencv-python)
NumPy (pip install numpy)
Matplotlib (pip install matplotlib)

Usage
To detect lanes in image or a video, first run:

python lanes.py
Then Select from the main menu you want to detect lanes from the image or the video

The program will use test_video.mp4 as input. It will display the processed video with lane lines overlaid in real-time. Press 'q' to exit the video playback.

How It Works
Image Processing:
Converts the image to grayscale and applies Gaussian blur.
Detects edges using Canny edge detection.

Region of Interest:
Defines a polygonal mask to isolate the region where lane lines are expected.

Hough Transform:
Applies Hough Transform to detect line segments in the region of interest.

Lane Detection:
Groups and averages detected line segments to determine left and right lane lines.
Draws these averaged lines on the original image/video frame.

Display:
Combines the lane lines with the original image/video using weighted addition to visualize the detected lanes.
