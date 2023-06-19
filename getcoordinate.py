import math
import cv2
import numpy as np

def calculate_distance(coord1, coord2):
        delta_x = coord2[0] - coord1[0]
        delta_y = coord2[1] - coord1[1]
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
        return distance

def getRealRobotPos(robotpos, centrum): 
    b = calculate_distance(robotpos, centrum)
    a = 1720
    c = math.sqrt(math.pow(a,2) + math.pow(b,2))
    angle_cb = math.acos(b / c)
    angle_cb = math.degrees(angle_cb)

    robotheight = 210
    robotlastangle = 180 - 90 - angle_cb
    robotb = robotheight * math.tan(math.radians(robotlastangle))

    vector_length = math.sqrt(a**2 + b**2)

    shortened_length = vector_length - robotb

    pixel_x = robotpos[0] + (shortened_length * (robotpos[0] - centrum[0]) / vector_length)
    pixel_y = centrum[1] + (shortened_length * (robotpos[1] - centrum[1]) / vector_length)

    print(str(pixel_x) + "," + str(pixel_y))

    newRobotCoordinate = (pixel_x, pixel_y)

    return newRobotCoordinate


newpos=getRealRobotPos((200,200), (250,250))
print(str(newpos))






