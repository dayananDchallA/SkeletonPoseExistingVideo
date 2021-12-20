# pip install mediapipe
from typing import List, Mapping, Optional, Tuple, Union
import math
import numpy as np

import cv2
import mediapipe as mp
import time

# initialize mediapipe pose solution
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

inFile = "C:\PlayGround\Test\dance.mp4"
# For webcam input:
cap = cv2.VideoCapture(inFile)

while cap.isOpened():
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # draw extracted pose on black white image
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    mp_draw.DrawingSpec((255, 0, 0), 2, 2),mp_draw.DrawingSpec((255, 0, 255), 2, 2))
    
    main = results.pose_landmarks
    main1 = results.pose_landmarks
    if main:
        for landmark in main.landmark:
            landmark.x=landmark.x+0.2
            #landmark.y=landmark.y
        mp_draw.draw_landmarks(
            img,
            main,
            mp_pose.POSE_CONNECTIONS,mp_draw.DrawingSpec((255, 0, 0), 2, 2),mp_draw.DrawingSpec((255, 0, 255), 2, 2))
    
    if main1:
        for landmark in main1.landmark:
            landmark.x=landmark.x-0.5
        mp_draw.draw_landmarks(
            img,
            main1,
            mp_pose.POSE_CONNECTIONS,mp_draw.DrawingSpec((255,0, 0), 2, 2),mp_draw.DrawingSpec((255, 255, 0), 2, 2))
        
    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q') :
        # break out of the while loop
        break
        
        
# Release the VideoCapture object.
cap.release()
 
# Close the windows.
cv2.destroyAllWindows()