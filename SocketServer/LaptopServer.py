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

stateServer.setQuadrants(np.array([531,326.25]), np.array([1443,315]), np.array([1434,798.75]), np.array([411,792]),mesh_Image)
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



