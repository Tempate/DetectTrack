#from ultralytics import YOLO
#from ultralytics.yolo.v8.detect.predict import DetectionPredictor

#model = YOLO("yolov8n.pt")

#import matplotlib.pyplot as plt
from lib.predictor import Predictor
from lib.detector import Detector
from lib.box import Box

import numpy as np
import cv2
import sys


QUIT_KEY = ord('q')

COLOR_REFERENCE = (0,255,0) # GREEN
COLOR_DETECTED  = (255,0,0) # BLUE
COLOR_PREDICTED = (0,0,255) # RED


# Read the video with opencv
try:
    cap = cv2.VideoCapture(sys.argv[1])
except IndexError:
    print("[-] Please provide an mp4 video file as an argument.")

# Initialize the viewer
cv2.startWindowThread()
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", 1500, 900)

new_box = []
objects = []

selecting_new_object = False


def mouse_event(event, x, y, flags, frame):
    global selecting_new_object

    # During mouse events we draw on a copy of the frame to
    # remove the previously drawn objects during mouse events.
    # This is specially useful when hovering.
    frame_ = frame.copy()

    if event == cv2.EVENT_MOUSEMOVE:
        if selecting_new_object:
            # Show the box that would result from clicking where 
            # the mouse is hovering over.
            box = Box(new_box[0], [x,y], color=COLOR_REFERENCE)
            box.draw(frame_)
            cv2.imshow("Frame", frame_)

    elif event == cv2.EVENT_LBUTTONDOWN:
        # Save the click's location to draw a new box
        new_box.append([x,y])

        if len(new_box) == 1:
            # A new object is being selected
            selecting_new_object = True

            cv2.circle(frame_, (x,y), radius=3, color=COLOR_REFERENCE, thickness=3)
            cv2.imshow("Frame", frame_)

        if len(new_box) == 2:
            # The box is complete. Save it.
            pt1 = new_box.pop()
            pt2 = new_box.pop()

            minx = min(pt1[0], pt2[0])
            miny = min(pt1[1], pt2[1])
            maxx = max(pt1[0], pt2[0])
            maxy = max(pt1[1], pt2[1])

            tl = [minx, miny]   # Top left
            br = [maxx, maxy]   # Bottom right

            # The object has already been selected
            selecting_new_object = False

            # We save the box as a reference
            box = Box(tl, br, color=COLOR_REFERENCE)

            # We create a new detector for the box
            box_detected = Box(tl, br, color=COLOR_DETECTED)
            detector = Detector(frame_, box_detected)

            # We create a new predictor for the box
            box_predicted = Box(tl, br, color=COLOR_PREDICTED)
            predictor = Predictor(box_predicted)
            
            # Save the trackers for each object
            objects.append([box,detector,predictor])
            
            # Draw the reference box
            box.draw(frame_)
            cv2.imshow("Frame", frame_)

while True:
    success, frame = cap.read()

    # Check the success value of reading each frame
    if not success:
        break
    
    # Load the YOLO model and predict the labels of objects
    # model.predict(frame, show=True)

    # Draw the boxes around the objects
    for i in range(len(objects)-1, -1, -1):
        box, detector, predictor = objects[i]

        # Draw the reference box
        box.draw(frame)

        # Draw the detected boxes
        box_track_window = detector.detect(frame)
        detector.box.reset_box(*box_track_window)
        detector.box.draw(frame)

        # Draw the predicted boxes
        prediction = predictor.predict()
        predictor.box.set_center(prediction[:2])
        predictor.box.draw(frame)

        # Update the predictor based on the current 
        # position of the reference box
        predictor.correct(detector.box.center)

        # Move the refenrece box upwards at a constant speed
        box.move_up(4)

        # Remove the box if the constant-speed guess is outside the screen
        if box.center[1] <= 0:
            del objects[i]
            break

    # Draw the frame
    cv2.imshow("Frame", frame)
    
    # Check for mouse clicks
    cv2.setMouseCallback("Frame", mouse_event, frame)

    # Check if the user wants to quit
    key = cv2.waitKey(0)

    if key == QUIT_KEY:
        break
        
cap.release()
cv2.destroyAllWindows()
