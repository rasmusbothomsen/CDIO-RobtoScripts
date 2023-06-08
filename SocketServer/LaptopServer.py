import socket
from NavigationController import NavigationController
import cv2
import numpy as np
import math

HOST = '192.168.137.1'  # Listen on all available interfaces
PORT = 1234

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to a specific interface and port
s.bind((HOST, PORT))

# Listen for incoming connections
s.listen(1)

print('Server listening on port', PORT)

# Wait for a client to connect
conn, addr = s.accept()
print('Connected by', addr)

data = conn.recv(1024)
print('Received', data)

robotPosition = (1184,383)
robotAngle = (0,-1)
image = cv2.imread(r"C:\Users\rasmu\OneDrive\Billeder\Filmrulle\WIN_20230607_15_22_56_Pro.jpg")
controller = NavigationController(image)
controller.scale_image(80)
imageCp = controller.image.copy()
circles,_,orangeBall = controller.find_circles(imageCp,130,130,130)
controller.create_binary_mesh()
path = controller.find_path(robotPosition,circles[0][:2])

for x in range(len(path)-1):
    cv2.line(imageCp,path[x],path[x+1],(255,0,0),2)
cv2.imshow("im",imageCp)

for x in range(len(path)-1):
    print(f"Node {x} out of {len(path)-1}")
    print(f"Angle of robot:{robotAngle}")
    vector1 = controller.VectorOf2Points(path[x],path[x+1])
    print(f"Path vector {vector1}")
    robotVectorAngle = controller.angle_between(robotAngle,vector1)
    vector1Len = np.linalg.norm(vector1)
    if(robotVectorAngle<=3):
        robotVectorAngle=0
    print(f"TurnAngle: {robotVectorAngle}")
    print(f"Vector length: {vector1Len}")
    if(np.cross(robotAngle,vector1)<0):
        conn.sendall(bytes(f"TurnLeft|{robotVectorAngle}",'utf-8'))
        robotAngle = -robotAngle
    else:
        conn.sendall(bytes(f"TurnRight|{robotVectorAngle}",'utf-8'))
    robotAngleRad = (conn.recv(1024).decode())
    robotAngle = controller.rotate_vector(robotAngle,robotVectorAngle)
    print(f"Robot Return angle: {robotAngle}")
    print(vector1Len)
    if(x==(len(path)-1)):
        break
    conn.sendall(bytes(f"Forward|{vector1Len*(1.5817223198594024604569420035149)}",'utf-8'))
    data = conn.recv(1024)

conn.sendall(bytes("GrabBall",'utf-8'))
conn.sendall(bytes("End",'utf-8'))
conn.close()

cv2.waitKey(0)
cv2.destroyAllWindows()
