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
    s.sendall(bytes(data,'utf-8'))
    if("turnleft" in data):
        SocketTest.turnToAngleLeft(angle,int(data.split("|")[1]))
    elif ("turnright" in data):
        SocketTest.turn(-300,300,int(data.split("|")[1]))
    elif ("grap" in data):
        SocketTest.grap(False, int(data.split("|")[1]))
    elif ("ungrap" in data):
        SocketTest.grap(True, int(data.split("|")[1]))
    elif ("unload" in data):
        SocketTest.unload(False, int(data.split("|")[1]))
    elif ("close" in data):
        SocketTest.unload(True, int(data.split("|")[1]))
    elif(data == "exit"):
        break
    elif("move" in data):
        SocketTest.moveForward(int(data.split("|")[1]))
    else:
        SocketTest.DisplayText(data)
        
    

SocketTest.playSound()
# Close the connection
s.close()
