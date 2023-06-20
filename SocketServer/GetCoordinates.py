# import math
# import cv2
# import numpy as np

# def calculate_distance(coord1, coord2):
#     delta_x = coord2[0] - coord1[0]
#     delta_y = coord2[1] - coord1[1]
#     distance = math.sqrt(math.pow(delta_x,2) + math.pow(delta_y,2))
#     return distance

# def normalizedPos(centrum, robot):
#     b_length = calculate_distance(centrum, robot)
#     a_length = 1640
#     c_length = math.sqrt(math.pow(a_length, 2) * math.pow(b_length,2))

#     angle_bc = math.acos(b_length / c_length)
#     print("radians angle:" + str(angle_bc))
#     angle_bc = math.degrees(angle_bc)
#     print("degrees angle:" + str(angle_bc))

#     robot_a_length = 210
#     robot_b_length = abs(robot_a_length / math.tan(angle_bc))
#     print(f'robot b length', robot_b_length)

#     vector_cToB_x = centrum[0] - robot[0]
#     vector_cToB_y = centrum[1] - robot[1]
#     #vector_cToB = (vector_cToB_x, vector_cToB_y)

#     x = (robot_b_length/b_length)*(vector_cToB_x)
#     y = (robot_b_length/b_length)*(vector_cToB_y)

#     newvector = (x,y)

#     x = robot[0] + newvector[0]
#     y = robot[1] + newvector[1]

#     return (x,y)

import math
import cv2
import numpy as np

def calculate_distance(coord1, coord2):
    delta_x = coord2[0] - coord1[0]
    delta_y = coord2[1] - coord1[1]
    distance = math.sqrt(math.pow(delta_x,2) + math.pow(delta_y,2))
    return distance

def calcpos(robot, centrum):

    b = calculate_distance(robot, centrum)
    a = 1640
    c = math.sqrt(math.pow(a,2) + math.pow(b,2))
    angle_cb = math.acos(b / c)
    angle_cb = math.degrees(angle_cb)

    robotheight = 210
    robotlastangle = 180 - 90 - angle_cb
    robotb = robotheight * math.tan(math.radians(robotlastangle))

    vector_length = math.sqrt(a**2 + b**2)
    shortened_length = vector_length - robotb
    pixel_x = centrum[0] + (shortened_length * (robot[0] - centrum[0]) / vector_length)
    pixel_y = centrum[1] + (shortened_length * (robot[1] - centrum[1]) / vector_length)

    return np.array([pixel_x, pixel_y])