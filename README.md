# Tello Drone Security Camera Tracking

A computer vision and autonomous control project built using the **DJI Tello drone**.

The project uses **YOLOv5** and **OpenCV** to detect security cameras in real time and automatically adjust the drone's movement to track and follow the detected camera.

## Features

- Real-time drone camera streaming
- Security camera detection
- Autonomous target tracking
- Left and right position correction
- Vertical position adjustment
- Forward and backward distance control
- Movement smoothing and stabilization
- DJI Tello flight control

## How It Works

The drone processes its live camera feed using a custom YOLOv5 model trained to recognize security cameras.

Once a camera is detected, the system compares its position with the center of the drone's camera frame and adjusts the drone's movement to keep the target centered.

The drone also uses the detected object's size to estimate whether it should move closer or farther away.

## Tech Stack

- Python
- DJI Tello
- YOLOv5
- PyTorch
- OpenCV
- NumPy
- DJITelloPy

## Project Goal

The goal of this project is to demonstrate autonomous drone perception and tracking by allowing a DJI Tello drone to detect, follow, and maintain alignment with a security camera using computer vision.
