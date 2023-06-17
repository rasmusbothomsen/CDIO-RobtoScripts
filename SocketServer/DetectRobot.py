import time
import numpy as np
import cv2
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.best_first import BestFirst
from pathfinding.core import heuristic, diagonal_movement
import math

        



def scaleImage(image,scale):
    scale_percent = scale # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    new_img_size = (resized.shape[1] - (resized.shape[1] % 32), resized.shape[0] - (resized.shape[0] % 32))
    resized_img = cv2.resize(resized, new_img_size)
    
    return resized_img


cam = cv2.VideoCapture(0)
time.sleep(2)
result, imagecp = cam.read()

imagecp = scaleImage(imagecp,80)
imagecp = cv2.cvtColor(imagecp,cv2.COLOR_BGR2RGB)
image = imagecp.copy()

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  

for cont in contours:
    perimeter = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.04 * perimeter, True)
    area =  cv2.contourArea(cont)
    if len(approx) == 3 and area > 300 and  area < 1000:
        # Triangle detected
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(approx)
        roi = imagecp[y:y+h, x:x+w]

        # Check the color inside the triangle ROI
        green_mask = cv2.inRange(roi, (0, 50, 0), (0, 255, 0))
        blue_mask = cv2.inRange(roi, (0, 0, 50), (0, 0, 255))
        if cv2.countNonZero(green_mask) > cv2.countNonZero(blue_mask):
            # The triangle contains green and blue
            cv2.putText(image, "Green", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # The triangle does not contain green and blue
            cv2.putText(image, "Blue", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        


cv2.imshow("Shapes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



