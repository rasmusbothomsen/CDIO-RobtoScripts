#!/usr/bin/env python3
import socket
from newTest import RobotController

HOST = '172.20.10.2'  # Replace with your laptop's IP address
PORT = 1234

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
controller = RobotController(40,20)

# Connect to the server
s.connect((HOST, PORT))

# Send data to the server
s.sendall(b'Hello, server!')
controller.StartRobot()
# Receive a response from the server
while True:
    data = s.recv(1024).decode()
    print(data)
    if("TurnLeft" in data):
        retrunAngle = controller.Turn(-float(data.split("|")[1]))
        s.sendall(bytes(str(retrunAngle),'utf-8'))
    if("TurnRight" in data):
        retrunAngle = controller.Turn(float(data.split("|")[1]))
        s.sendall(bytes(str(retrunAngle),'utf-8'))
    elif("Forward" in data):
        controller.DriveDistance(float(data.split("|")[1]))
        s.sendall(bytes("as",'utf-8'))
    elif("GrabBall" in data):
        controller.GrapBall()
    elif("End" in data):
        controller.dispose()
        break
    

# Close the connection
s.close()
