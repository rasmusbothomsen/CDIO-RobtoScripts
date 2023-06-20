import math
import cv2
import numpy as np

def calculate_distance(coord1, coord2):
        delta_x = coord2[0] - coord1[0]
        delta_y = coord2[1] - coord1[1]
        # delta_x = abs(delta_x)
        # delta_y = abs(delta_y)
        distance = np.linalg.norm((delta_x,delta_y))
        return distance

def getRealRobotPos(robotpos, centrum): 
    b = calculate_distance(centrum,robotpos )
    a = 1640
    c = math.sqrt(math.pow(a,2) + math.pow(b,2))
    angle_cb = math.acos(b / c)
    angle_cb = math.degrees(angle_cb)

    robotheight = 210
    robotlastangle = 180 - 90 - angle_cb
    robotb = robotheight * math.tan(math.radians(robotlastangle))

    vector_length = math.sqrt(a**2 + b**2)

    shortened_length = b - robotb

    pixel_x = robotpos[0] + (shortened_length * (robotpos[0] - centrum[0]) / vector_length)
    pixel_y = centrum[1] + (shortened_length * (robotpos[1] - centrum[1]) / vector_length)

    print(str(pixel_x) + "," + str(pixel_y))

    newRobotCoordinate = np.array((pixel_x, pixel_y),np.int32)

    return newRobotCoordinate

