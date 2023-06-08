#!/usr/bin/env pybricks-micropython
import socket
import SocketTest


HOST = '192.168.137.1'  # Replace with your laptop's IP address
PORT = 1234

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
s.connect((HOST, PORT))

# Send data to the server
s.sendall(b'Hello, server!')

# Receive a response from the server
while True:
    data = s.recv(1024).decode()
    print(data)
    if("TurnLeft" in data):
        retrunAngle = SocketTest.turnToAngleLeft(float(data.split("|")[1]))
        s.sendall(bytes(str(retrunAngle),'utf-8'))
    if("TurnRight" in data):
        retrunAngle = SocketTest.turnToAngleRight(float(data.split("|")[1]))
        s.sendall(bytes(str(retrunAngle),'utf-8'))
    elif("Forward" in data):
        SocketTest.moveForward(float(data.split("|")[1]))
        s.sendall(bytes("as",'utf-8'))
    elif("GrabBall" in data):
        SocketTest.grab_ball()
    elif("End" in data):
        break
    

SocketTest.playSound()
# Close the connection
s.close()
