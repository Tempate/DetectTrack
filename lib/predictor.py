import numpy as np
import cv2


class Predictor():
    def __init__(self, box):
        self.box = box
        self.filter = cv2.KalmanFilter(4,2)
    
        # Initialize the matrices for the filter
        self.filter.measurementMatrix = np.array([
            [1,0,0,0],
            [0,1,0,0]
        ], np.float32)

        self.filter.transitionMatrix = np.array([
            [1,0,1,0],
            [0,1,0,1],
            [0,0,1,0],
            [0,0,0,1]
        ], np.float32)

        self.filter.processNoiseCov = np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ], np.float32) * 0.03

    def correct(self, measured_object_pos):
        # Updates the predicted state from the measurement
        x = np.float32(measured_object_pos[0])
        y = np.float32(measured_object_pos[1])
        self.filter.correct(np.array([[x], [y]]))

    def predict(self):
        # Predict where the box is going to be next
        return self.filter.predict()