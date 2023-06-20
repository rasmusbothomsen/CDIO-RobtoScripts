#!/usr/bin/env python3

from ev3dev2.motor import OUTPUT_A, OUTPUT_D,OUTPUT_B,OUTPUT_C, MoveDifferential, SpeedRPM, Motor
from ev3dev2.wheel import EV3Tire, Wheel,EV3EducationSetRim
import asyncio
from ev3dev2.sensor.lego import GyroSensor,UltrasonicSensor
from ev3dev2.sensor import INPUT_4,INPUT_1
from time import sleep
from SocketServer.catterpillerWheel import catterpillerWheel
import logging



class RobotController(object):

    def __init__(self,driveSpeed,TurnSpeed):    
        logging.basicConfig( level=logging.DEBUG,format='%(message)s')
        self.mdiff = MoveDifferential(OUTPUT_D, OUTPUT_A,wheel_class= catterpillerWheel,wheel_distance_mm=116.0)
        self.mdiff.gyro = GyroSensor(INPUT_4)
        self.driveSpeed = driveSpeed
        self.TurnSpeed = TurnSpeed
        self.GrapMotor = Motor(OUTPUT_C)
        self.Unloader = Motor(OUTPUT_B)
        self.UltraSensor = UltrasonicSensor(INPUT_1)

    def StartRobot(self):
        self.mdiff.gyro.calibrate()
        self.mdiff.odometry_start(0,0)

    def Turn(self,degrees):
        self.mdiff.turn_degrees(self.TurnSpeed,degrees,use_gyro=True)
        return 100

    def GrapBall(self,distance):
        success = True
        self.mdiff.on_for_distance(self.driveSpeed/2,distance,block=False)
        while self.UltraSensor.distance_centimeters > 7 and success:
            if not self.mdiff.is_running:
                success = False
            print(self.UltraSensor.distance_centimeters)
        self.mdiff.off()
        if(success):
            self.GrapMotor.on_for_degrees(40,1500)
            self.GrapMotor.on_for_degrees(40,-1500)
        
    
    def Unload(self):
        self.Unloader.on_for_degrees(20,150,block=False)

    def CloseUnLoad(self):
        self.Unloader.on_for_degrees(20,-150,block=False)

        
        
        
    async def LookForBall(self):
        while self.UltraSensor.distance_centimeters() > 7:
            pass
        self.mdiff.off()
        self.GrapMotor.on_for_degrees(40,1500)
        self.GrapMotor.on_for_degrees(40,-1500)




    def DriveDistance(self,distance):
        self.mdiff.on_for_distance(self.driveSpeed,distance)

    
    def dispose(self):
        self.mdiff.odometry_stop()