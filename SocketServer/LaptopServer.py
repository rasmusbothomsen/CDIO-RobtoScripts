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

stateServer.navigationController.show_image(mesh_Image)

stateServer.DectionAndpathing(mesh_Image)

stateServer.SetUpSocketConnection(HOST,PORT)

stateServer.Translatepath()

stateServer.CloseConnection()



