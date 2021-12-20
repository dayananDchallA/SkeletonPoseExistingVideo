# pip install mediapipe
from typing import List, Mapping, Optional, Tuple, Union
import math
import numpy as np

import cv2
import mediapipe as mp
import time


_VISIBILITY_THRESHOLD = 0.5

# initialize mediapipe pose solution
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()


def _normalized_to_pixel_coordinates(x, y, image_width,image_height):
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value):
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(x) and is_valid_normalized_value(y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(x * image_width), image_width - 1)
  y_px = min(math.floor(y * image_height), image_height - 1)
  return x_px, y_px
  
  
def drawPoseLandMarks(inImage,pose_coordinates):
    for idx, landmark_px in pose_coordinates.items():
        landmark_px_right = (landmark_px[0]+150,landmark_px[1])
        cv2.circle(inImage, landmark_px_right, 4, (0, 0, 255), cv2.FILLED)
        
        landmark_px_left = (landmark_px[0]-150,landmark_px[1])
        cv2.circle(inImage, landmark_px_left, 4, (0, 0, 255), cv2.FILLED)
        

def drawPoseConnections(image, connections,pose_coordinates):
    if connections:
            num_landmarks = len(results.pose_landmarks.landmark)
            # Draws the connections if the start and end landmarks are both visible.
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                    raise ValueError(f'Landmark index is out of range. Invalid connection '
                         f'from landmark #{start_idx} to landmark #{end_idx}.')
                if start_idx in pose_coordinates and end_idx in pose_coordinates:
                    
                    start = (pose_coordinates[start_idx][0]+150,pose_coordinates[start_idx][1])
                    end = (pose_coordinates[end_idx][0]+150,pose_coordinates[end_idx][1])
                    cv2.line(image, start,end,  (255,228,0),3)
                    start = (pose_coordinates[start_idx][0]-150,pose_coordinates[start_idx][1])
                    end = (pose_coordinates[end_idx][0]-150,pose_coordinates[end_idx][1])
                    cv2.line(image, start,end,  (255,98,20),3)
                    
def getPoseCoordinates(poseImg,landmarks):
    image_rows, image_cols, _ = poseImg.shape
    coordinates = {}
    for idx, landmark in enumerate(landmarks.landmark):
        if ((landmark.HasField('visibility') and landmark.visibility < _VISIBILITY_THRESHOLD) or (landmark.HasField('presence') and
            landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,image_cols, image_rows)
        
        if landmark_px:
            coordinates[idx] = landmark_px
    return coordinates

if __name__ == "__main__":
    # take video input for pose detection
    # you can put here video of your choice
    inFile = "C:\PlayGround\Test\dance.mp4"
    cap = cv2.VideoCapture(inFile)
    while True:
        success, img = cap.read()
    
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            mp_draw.draw_landmarks(img, results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            idx_to_coordinates = getPoseCoordinates(img,results.pose_landmarks)
            drawPoseConnections(img, mp_pose.POSE_CONNECTIONS,idx_to_coordinates)
            drawPoseLandMarks(img,idx_to_coordinates)
        
        
        # Extract and draw pose on plain white image
        h, w, c = img.shape   # get shape of original frame
        opImg = np.zeros([h, w, c])  # create blank image with original frame size
        opImg.fill(255)  # set white background. put 0 if you want to make it black

        # draw extracted pose on black white image
        mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                               mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                               )
        cv2.imshow("Image", img)
        #display extracted pose on blank images
        cv2.imshow("Extracted Pose", opImg)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q') :
            # break out of the while loop
            break