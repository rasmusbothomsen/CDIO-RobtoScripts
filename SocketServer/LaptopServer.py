import socket
from NavigationController import NavigationController
import cv2
import numpy as np
import math
from StateServer import Stateserver
from StateServer import State

HOST = '172.20.10.2'
PORT = 1234

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to a specific interface and port
s.bind((HOST, PORT))

# Listen for incoming connections
s.listen(1)

print('Server listening on port', PORT)

# Wait for a client to connect



stateserver = Stateserver()
state = State()

stateserver.initImageAnalysis(state, s)
while(state.anyBallsLeft == True):
    stateserver.imageAnalysis(s , state)

cv2.waitKey(0)
cv2.destroyAllWindows()
