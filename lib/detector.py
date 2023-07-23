import numpy as np
import cv2


# A wrapper around opencv's meanShift function.
class Detector:
    def __init__(self, frame, box):
        self.box = box

        ## Process the region that has been boxed

        # 1. Get the region of the frame inside of the box
        self.track_window = (box.tl[0],box.tl[1],box.w,box.h)

        box_frame = frame[
            box.tl[1]:box.br[1],
            box.tl[0]:box.br[0]
        ]

        # 2. Convert the box's frame from BGR to HSV format
        box_frame_hsv = cv2.cvtColor(box_frame, cv2.COLOR_BGR2HSV)
  
        # 3. Apply a mask to the box's frame in HSV format
        mask = cv2.inRange(box_frame_hsv, 
                np.array((0., 20., 32.)),
                np.array((180., 255., 255)))
  
        # 4. Calculate the histogram
        self.box_hist = cv2.calcHist([box_frame_hsv], [0], 
                                     mask, [180], [0,180])
        cv2.normalize(self.box_hist, self.box_hist, 0, 255, 
                      cv2.NORM_MINMAX)

        ## Set up the termination criteria for the shift: 
        ## either 10 iterations or 1 pixel movement
        self.term_crit = (cv2.TERM_CRITERIA_EPS | 
                          cv2.TERM_CRITERIA_COUNT, 10, 1)

    def detect(self, frame):
        ## Detect the box in the frame
        
        # 1. Perform thresholding on the video frames
        """
        _, frame = cv2.threshold(frame, 180, 50, 
                                 cv2.THRESH_TOZERO_INV)
        """
    
        # 2. Convert the frame from BGR to HSV format
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        # 3. Backpropagate
        dst = cv2.calcBackProject([frame_hsv], 
                                 [0], 
                                 self.box_hist, 
                                 [0, 180], 1)
        
        # 4. Find the new location of the box
        ret, self.track_window = cv2.meanShift(dst,
                                         self.track_window,
                                         self.term_crit)
        
        return self.track_window
