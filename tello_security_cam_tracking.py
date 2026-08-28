import os
import time
import torch
import cv2 as cv
import numpy as np
from djitellopy import Tello

####################### [ CONFIG ] #######################
xMainPath = r"C:\Users\ALIENWARE\Desktop\yolov5-7.0\yolov5-7.0\runs\train\exp21\weights\best.pt"

shouldFly = True
screenSize = (960, 540)  # Output screen size

# Stability and smoothing parameters
DEAD_ZONE_X = 50
DEAD_ZONE_Y = 50
max_velocity = 50  # Maximum velocity for smoother movements
stability_threshold = 3  # Frames required for stable direction

# Target size range for maintaining a specific distance
TARGET_SIZE_MIN = 12000  # Adjust based on experimentation
TARGET_SIZE_MAX = 14000  # Adjust based on experimentation
####################### [ CONFIG ] #######################


def get_targets(results):
    processed_targets = []
    screen_center_x, screen_center_y = screenSize[0] / 2, screenSize[1] / 2

    for target in results.xywh[0]:
        # Extract values and filter based on label
        label = int(target[5].item())
        x, y, w, h = target[:4].tolist()
        confidence = target[4].item()

        # x and y are center coordinates in YOLOv5's xywh format
        cx = x
        cy = y

        # Determine the position relative to the screen center
        CurrentSideX = "LEFT" if cx < screen_center_x else "RIGHT" if cx > screen_center_x else "CENTER"
        CurrentDiffX = abs(cx - screen_center_x)

        CurrentSideY = "UP" if cy < screen_center_y else "DOWN" if cy > screen_center_y else "CENTER"
        CurrentDiffY = abs(cy - screen_center_y)

        processed_targets.append({
            "SideX": CurrentSideX,
            "DiffX": CurrentDiffX,
            "SideY": CurrentSideY,
            "DiffY": CurrentDiffY,
            "size": w * h,
            "confidence": confidence,
        })

    print(processed_targets)
    return processed_targets


# Load the YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path=xMainPath)

if torch.cuda.is_available():
    model.cuda()

# Initialize the Tello drone
tello = Tello()
tello.connect()

# Start the Tello video stream and wait for it to initialize
tello.streamon()
time.sleep(5)  # Allow time for the video stream to initialize

# Take off and stabilize
if shouldFly:
    tello.takeoff()
    tello.send_rc_control(0, 0, 0, 0)  # Send initial stop command to stabilize

# Get frame dimensions for the center reference
frame_width, frame_height = screenSize[0], screenSize[1]
frame_center_x, frame_center_y = frame_width // 2, frame_height // 2

# Initialize stability variables
history = {"SideX": None, "SideY": None}
stability_counter = 0

try:
    while True:
        print("---------------------------")
        frame = tello.get_frame_read().frame

        if frame is None or frame.size == 0:
            print("Warning: Frame is empty, skipping this frame.")
            continue

        frame_bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        img_resized = cv.resize(frame_bgr, (frame_width, frame_height))
        results = model(img_resized)

        all_target = get_targets(results)
        left_right_velocity = 0
        forward_backward_velocity = 0
        up_down_velocity = 0
        yaw_velocity = 0

        if len(all_target) > 0:
            target = all_target[0]

            if history["SideX"] == target["SideX"] and history["SideY"] == target["SideY"]:
                stability_counter += 1
            else:
                stability_counter = 0

            history["SideX"] = target["SideX"]
            history["SideY"] = target["SideY"]

            if stability_counter >= stability_threshold:
                # Apply dead zone to reduce unnecessary movements
                if target["DiffX"] > DEAD_ZONE_X:
                    if target["SideX"] == "RIGHT":
                        left_right_velocity = int((target["DiffX"] / frame_center_x) * max_velocity)
                    elif target["SideX"] == "LEFT":
                        left_right_velocity = -int((target["DiffX"] / frame_center_x) * max_velocity)

                if target["DiffY"] > DEAD_ZONE_Y:
                    if target["SideY"] == "UP":
                        up_down_velocity = int((target["DiffY"] / frame_center_y) * max_velocity)
                    elif target["SideY"] == "DOWN":
                        up_down_velocity = -int((target["DiffY"] / frame_center_y) * max_velocity)

                # FORWARD or BACKWARD based on object size (distance logic added here)
                if target["size"] < TARGET_SIZE_MIN:
                    forward_backward_velocity = int((TARGET_SIZE_MIN - target["size"]) / TARGET_SIZE_MIN * max_velocity)
                elif target["size"] > TARGET_SIZE_MAX:
                    forward_backward_velocity = -int((target["size"] - TARGET_SIZE_MAX) / TARGET_SIZE_MAX * max_velocity)
                else:
                    forward_backward_velocity = 0  # Within the ideal range, no forward/backward movement

                if shouldFly:
                    tello.send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity)
            else:
                print("Stabilizing...")
                if shouldFly:
                    tello.send_rc_control(0, 0, 0, 0)
        else:
            print("Hovering to stabilize...")
            if shouldFly:
                tello.send_rc_control(0, 0, 0, 0)

        results.render()
        cv.imshow("Tello Camera Detection", results.ims[0])

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.05)

except Exception as e:
    print(f"Error occurred: {e}")
    if shouldFly:
        tello.land()
finally:
    if shouldFly:
        tello.land()
    tello.streamoff()
    cv.destroyAllWindows()
