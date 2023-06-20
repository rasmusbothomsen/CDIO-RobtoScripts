import socket
from NavigationController import NavigationController
from StateServer import Stateserver
import cv2
import numpy as np
import math

HOST = '192.168.137.1'  # Listen on all available interfaces
PORT = 1234
stateServer = Stateserver()

mesh_Image = stateServer.imageCapture()

stateServer.InitBinaryMesh(mesh_Image)

stateServer.SetUpSocketConnection(HOST,PORT)
stateServer.SetGoal(np.array([582, 550]),np.array([1464,546]),mesh_Image)
while(stateServer.runState.anyBallsLeft):
    print("Dection Balls")
    stateServer.DectionAndpathing(mesh_Image)
    print("Sending Path")
    try:
        stateServer.Translatepath()
    except Exception as e:
        print(e)

    mesh_Image = stateServer.imageCapture()

stateServer.CloseConnection()



