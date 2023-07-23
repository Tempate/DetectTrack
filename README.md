# DetectTrack

This project allows users to select an object and track it in subsequent frames by combining an object detector (opencv's MeanShift) with an object tracker (opencv's KalmanFilter). 

## Running the code

1. Install the requirements:

```
pip install -r requirements.txt
```

2. Run main.py with a video file:

```
python main.py video.mp4
```

## Interacting with the GUI

Users can:
* **Skip to the next frame** of the video by pressing the `SPACEBAR`.

* **Select a region** to be tracked by clicking on two points.

    The detector works best when all four corners of the selected region are inside the object that is being tracked.

* **Quit the program** by pressing `q`.

## Trackers

The program displays three different trackers so that the user can compare them frame by frame.

* In **green**, a reference tracker moving upwards at a constant speed.
* In **blue**, the region detected by Mean Shift that resembles the original section.
* In **red**, the tracked region resulting from applying the Kalman Filter to Mean Shift.  


## Structure of the code

The code consists of four files:

1. **Box.py**: defines the box objects that delimit the regions both selected by the user and detected and tracked by the program.
2. **Detector.py**: is a wrapper around opencv's Mean Shift function.
3. **Predictor.py**: is a wrapper around opencv's KalmanFilter function. 
4. **main.py**: combines the other files to display the three trackers.


## Diary

The file `diary.txt` contains my thought process while working on the project. I have tried to edit it as little as possible so it remains faithful to my first incursion into Computer Vision.