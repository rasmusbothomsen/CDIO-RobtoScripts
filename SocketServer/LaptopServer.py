import socket
from NavigationController import NavigationController
import cv2
import numpy as np
import math

HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 1234

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to a specific interface and port
s.bind((HOST, PORT))

# Listen for incoming connections
s.listen(1)

print('Server listening on port', PORT)

# Wait for a client to connect



robotPosition = (379,630)

cam = cv2.VideoCapture(0)
result, image = cam.read()

#image = cv2.imread("/Users/berfinfloraturan/Desktop/robot2.jpg")
controller = NavigationController(image)
failed = False
# This is done in cases where the triangle is not found on the 100% size of the image, then we try again on the 80% size
try:
    controller.detectRobot(image)
except:
    failed = True

controller.scale_image(80)
if(failed):
    controller.detectRobot(controller.image)

robotDirection = controller.getRobotPosition()
robotPosition = robotDirection["back"]

robotPosition = (379,630)
lat1 = robotDirection['front'][0]
long1 = robotDirection['front'][1]
lat2 = robotDirection['back'][0]
long2 = robotDirection['back'][1]
robotAngle = controller.getRobotAngle(lat1, long1, lat2, long2)
print(robotAngle)

imageCp = controller.image.copy()
cv2.circle(imageCp,(robotDirection['front']),10,(255,0,0),-1)
controller.show_image(imageCp)
circles,ballImage,orangeBall = controller.find_circles(imageCp,130,130,130)
controller.show_image(ballImage)
controller.create_binary_mesh(50)
controller.show_image(controller.image)
path = controller.find_path(robotPosition,(circles[0][:2]))

conn, addr = s.accept()
print('Connected by', addr)

data = conn.recv(1024)
print('Received', data)
for x in range(len(path)-1):
    cv2.line(imageCp,path[x],path[x+1],(255,0,0),2)


if circles is not None:
        for (x, y, r) in circles:
            cv2.circle(imageCp, (x, y), r, (0, 0, 255), 2)

controller.show_image(imageCp)
for x in range(len(path)-1):
    print(f"Node {x} out of {len(path)-1}")
    print(f"Angle of robot:{robotAngle}")
    vector1 = controller.VectorOf2Points(path[x],path[x+1])
    print(f"Path vector {vector1}")
    robotVectorAngle = controller.angle_between(robotAngle,vector1)
    vector1Len = np.linalg.norm(vector1)
    if(x == len(path)-2):
         print(f"Last run {vector1Len-400} new val {vector1Len-400}")
         vector1Len = vector1Len-400
    if(robotVectorAngle<=3):
        robotVectorAngle=0
    print(f"TurnAngle: {robotVectorAngle}")
    print(f"Vector length: {vector1Len}")
    if(np.cross(robotAngle,vector1)<0):
        conn.sendall(bytes(f"TurnLeft|{robotVectorAngle}",'utf-8'))
        robotVectorAngle = -robotVectorAngle
    else:
        conn.sendall(bytes(f"TurnRight|{robotVectorAngle}",'utf-8'))
    robotAngleRad = (conn.recv(1024).decode())
    robotAngle = controller.rotate_vector(robotAngle,robotVectorAngle)
    print(f"Robot Return angle: {robotAngle}")
    print(vector1Len)
    if(x==(len(path)-1)):
        break
    conn.sendall(bytes(f"Forward|{vector1Len*(1.567136150234741784037558685446)}",'utf-8'))
    data = conn.recv(1024)

conn.sendall(bytes("GrabBall",'utf-8'))
conn.sendall(bytes("End",'utf-8'))
conn.close()

cv2.waitKey(0)
cv2.destroyAllWindows()
