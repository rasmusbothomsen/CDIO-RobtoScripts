from NavigationController import NavigationController
import cv2
import socket
import numpy as np
import UndisTest
from time import sleep

class State:
    def __init__(self):
        print("State init")

        self.initialCount = 0
        self.loadOffCount = 0
        self.anyBallsLeft = True
        self.DropOffState = False
        self.BigGoal = []
        self.SmallGoal = []
        self.Q1 = []
        self.Q2 = []
        self.Q3 = []
        self.Q4 = []


class Stateserver:
    def __init__(self):
        print("Stateserver init")
        self.navigationController = NavigationController()
       
        self.runState = State()

    def imageCapture(self):
        cam = cv2.VideoCapture(1,cv2.CAP_DSHOW)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        sleep(0.5)
        result, image = cam.read()
        image = UndisTest.undistort(image)
        del(cam)
        return image

    def InitBinaryMesh(self,image,Test=False):
        imagecp = image.copy()
        imagecp = self.navigationController.scale_image(80,imagecp)
        self.binaryMesh = self.navigationController.create_binary_mesh(60,imagecp,Test)

    def SetGoal(self,BigGoal, smallGoal,image):
        self.runState.BigGoal = BigGoal / (image.shape[1], image.shape[0])
        self.runState.SmallGoal = smallGoal / (image.shape[1], image.shape[0])

    def setQuadrants(self,Q1, Q2, Q3, Q4 ,image):
        self.runState.Q1 = Q1 / (image.shape[1], image.shape[0])
        self.runState.Q2 = Q2 / (image.shape[1], image.shape[0])
        self.runState.Q3 = Q3 / (image.shape[1], image.shape[0])
        self.runState.Q4 = Q4 / (image.shape[1], image.shape[0])
    
    def GetGoals(self,image):
        return (self.runState.BigGoal * (image.shape[1], image.shape[0])).astype(np.int32),(self.runState.SmallGoal * (image.shape[1], image.shape[0])).astype(np.int32)
   
    def GetQuadrants(self,image):return (self.runState.Q1 * (image.shape[1], image.shape[0])).astype(np.int32),(self.runState.Q2 * (image.shape[1], image.shape[0])).astype(np.int32),(self.runState.Q3 * (image.shape[1], image.shape[0])).astype(np.int32),(self.runState.Q4 * (image.shape[1], image.shape[0])).astype(np.int32)
   
    def DectionAndpathing(self,image):
        failed = False
        # This is done in cases where the triangle is not found on the 100% size of the image, then we try again on the 80% size
        try:
            self.navigationController.detectRobot(image)
        except:
            failed = True

        localImage = self.navigationController.scale_image(80,image)
        if(failed):
            # self.navigationController.show_image(localImage)
            self.navigationController.detectRobot(localImage)

        robotDirection = self.navigationController.getRobotPosition(localImage)
        self.robotPosition = robotDirection["center"]
        self.robotAngle = self.navigationController.VectorOf2Points(robotDirection['back'],robotDirection['front'])

        imageCp = localImage
        cv2.circle(imageCp,(robotDirection['front']),5,(255,0,0),-1)
        cv2.circle(imageCp,(robotDirection['back']),5,(0,0,255),-1)
        self.navigationController.show_image(imageCp)
        try:
            circles,ballImage,orangeBall = self.navigationController.find_circles(imageCp,130,130,130)
        except Exception as e:
            circles = []
            print(e)
        print(f"Balls in image{len(circles)}")
        if (self.runState.initialCount == 0):
            self.runState.initialCount = len(circles)
            if(self.runState.initialCount <= 6):
                self.runState.loadOffCount = 0
            else:
                self.runState.loadOffCount = len(circles) - 6
        sucess = False
        x = 0
        while not sucess and x < len(circles):
            if(len(circles) <= self.runState.loadOffCount):
                # self.path,sucess = self.navigationController.find_path(self.robotPosition,self.GetGoals(localImage)[0])
                self.runState.DropOffState = True
            else: 
                self.path = self.navigationController.generate_path_with_obstacles(self.binaryMesh,self.robotPosition,(circles[x][:2]),100)
                x = x+1
                sucess = True
                self.runState.DropOffState = False
        if(not sucess):
            pass
            self.path,sucess = self.navigationController.find_path(self.robotPosition,self.GetGoals(localImage)[0])

        
        for x in range(len(self.path)-1):
            cv2.line(imageCp,self.path[x],self.path[x+1],(255,0,0),2)
        self.navigationController.show_image(imageCp)
        

    def SetUpSocketConnection(self,ip,port):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.settimeout(None)

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
        if self.runState.initialCount < 1:
            self.runState.anyBallsLeft = False
        print(self.runState.initialCount)

    def SendCommand(self,command):
        self.conn.sendall(command)

    def DriveForward(self,length):
        return bytes(f"Forward|{length*(1.7030744337)}",'utf-8')
    
    def Turn(self,angle,left):
       if left:
           return bytes(f"TurnLeft|{angle}",'utf-8')
       else:
           return bytes(f"TurnRight|{angle}",'utf-8')
           
    def Translatepath(self):
        for x in range(len(self.path)-1):
            print(f"Node {x} out of {len(self.path)-1}")
            print(f"Angle of robot:{self.robotAngle}")
            vector1 = self.navigationController.VectorOf2Points(self.path[x],self.path[x+1])
            print(f"path vector {vector1}")
            robotVectorAngle = self.navigationController.angle_between(self.robotAngle,vector1)
            vector1Len = np.linalg.norm(vector1)
            
            if(robotVectorAngle<=3):
                robotVectorAngle=0
            print(f"TurnAngle: {robotVectorAngle}")
            print(f"Vector length: {vector1Len}")
            if(np.cross(self.robotAngle,vector1)<0):
                self.SendCommand(self.Turn(robotVectorAngle,True))
                robotVectorAngle = -robotVectorAngle
            else:
                self.SendCommand(self.Turn(robotVectorAngle,False))
            robotAngleRad = (self.conn.recv(1024).decode())
            self.robotAngle = self.navigationController.rotate_vector(self.robotAngle,robotVectorAngle)
            print(f"Robot Return angle: {self.robotAngle}")
            print(vector1Len)
            if(x == len(self.path)-2):
                print(f"Last run {vector1Len/4} new val {vector1Len/4}")
                if self.runState.DropOffState:
                    vector1Len = vector1Len*0.75
                else:
                    break
            if(x==(len(self.path)-1)):
                break
            self.SendCommand(self.DriveForward(vector1Len))
            data = self.conn.recv(1024).decode()

        if(self.runState.DropOffState):
            robotVectorAngle = self.navigationController.angle_between(self.robotAngle,(1,0))
            if(np.cross(self.robotAngle,(1,0))<0):
                self.SendCommand(self.Turn(robotVectorAngle,True))
                robotVectorAngle = -robotVectorAngle
            else:
                self.SendCommand(self.Turn(robotVectorAngle,False))
            self.conn.recv(1024)
            
            self.SendCommand(self.DriveForward(-20))
            self.conn.recv(1024)
            self.SendCommand(bytes("Unload",'utf-8'))
            sleep(2)
            self.conn.recv(1024)
            self.SendCommand(bytes("Close",'utf-8'))
            self.runState.initialCount = 0
            
            
            
        else:
            self.conn.sendall(bytes(f"GrabBall|{vector1Len*1.567136150234741784037558685446}",'utf-8'))
        self.conn.recv(1024)