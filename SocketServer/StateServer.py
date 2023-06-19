from NavigationController import NavigationController
import cv2
import socket
import numpy as np

class State:
    def __init__(self):
        print("State init")

    initialCount = 0
    loadOffCount = 0
    anyBallsLeft = True

class Stateserver:
    def __init__(self):
        print("Stateserver init")
        self.navigationController = NavigationController()
        self.cam = cv2.VideoCapture(1)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.runState = State()

    def imageCapture(self):
        result, image = self.cam.read()
        return image

    def InitBinaryMesh(self,image):
        self.binaryMesh = self.navigationController.create_binary_mesh(50,image)

    
    def DectionAndpathing(self,image):
        failed = False
        # This is done in cases where the triangle is not found on the 100% size of the image, then we try again on the 80% size
        try:
            self.navigationController.detectRobot(image)
        except:
            failed = True

        self.localImage = self.navigationController.scale_image(80,image)
        if(failed):
            self.navigationController.detectRobot(self.localImage)

        robotDirection = self.navigationController.getRobotPosition()
        self.robotPosition = robotDirection["front"]
        self.robotAngle = self.navigationController.VectorOf2Points(robotDirection['back'],robotDirection['front'])


        imageCp = self.localImage
        cv2.circle(imageCp,(robotDirection['front']),10,(255,0,0),-1)
        self.navigationController.show_image(imageCp)
        circles,ballImage,orangeBall = self.navigationController.find_circles(imageCp,130,130,130)
        self.runState.initialCount = len(circles)
        if(self.runState.initialCount <= 6):
            self.runState.loadOffCount = 0
        else:
            self.runState.loadOffCount = len(circles) - 6

        self.path = self.navigationController.find_path(self.robotPosition,(circles[0][:2]))

    def SetUpSocketConnection(self,ip,port):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to a specific interface and port
        self.s.bind((ip, port))

        # Listen for incoming connections
        self.s.listen(1)
        print('Server listening on port', port)

        self.conn, self.addr = self.s.accept()
        print('Connected by', self.addr)

        data = self.conn.recv(1024)
        print('Received', data)


    def CloseConnection(self):
        self.conn.sendall(bytes("End",'utf-8'))
        self.conn.close() 




    def UpdateState(self):
        pass

    def Translatepath(self):
        for x in range(len(self.path)-1):
            print(f"Node {x} out of {len(self.path)-1}")
            print(f"Angle of robot:{robotAngle}")
            vector1 = self.navigationController.VectorOf2Points(self.path[x],self.path[x+1])
            print(f"path vector {vector1}")
            robotVectorAngle = self.navigationController.angle_between(robotAngle,vector1)
            vector1Len = np.linalg.norm(vector1)
            if(x == len(self.path)-2):
                print(f"Last run {vector1Len-400} new val {vector1Len-400}")
                vector1Len = vector1Len-400
            if(robotVectorAngle<=3):
                robotVectorAngle=0
            print(f"TurnAngle: {robotVectorAngle}")
            print(f"Vector length: {vector1Len}")
            if(np.cross(robotAngle,vector1)<0):
                self.conn.sendall(bytes(f"TurnLeft|{robotVectorAngle}",'utf-8'))
                robotVectorAngle = -robotVectorAngle
            else:
                self.conn.sendall(bytes(f"TurnRight|{robotVectorAngle}",'utf-8'))
            robotAngleRad = (self.conn.recv(1024).decode())
            robotAngle = self.navigationController.rotate_vector(robotAngle,robotVectorAngle)
            print(f"Robot Return angle: {robotAngle}")
            print(vector1Len)
            if(x==(len(self.path)-1)):
                break
            self.conn.sendall(bytes(f"Forward|{vector1Len*(1.567136150234741784037558685446)}",'utf-8'))
            data = self.conn.recv(1024)

        self.conn.sendall(bytes("GrabBall",'utf-8'))
        self.UpdateState()