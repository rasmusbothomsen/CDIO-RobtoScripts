#!/usr/bin/env python3
import socket
from newTest import RobotController

HOST = '192.168.137.1'  # Replace with your laptop's IP address
PORT = 1234

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(None)
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
        controller.GrapBall(float(data.split("|")[1]))
        s.sendall(bytes("Ball",'utf-8'))
    elif("Unload" in data):
        controller.Unload()
        s.sendall(bytes("unload open",'utf-8'))
    elif("Close" in data):
        controller.CloseUnLoad()
        s.sendall(bytes("unload closed",'utf-8'))
    elif("End" in data):
        controller.dispose()
        print("Ending Now dont know why")
        break
    

# Close the connection
s.close()
