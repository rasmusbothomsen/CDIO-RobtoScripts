#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile
import math


ev3 = EV3Brick()
leftMotor = Motor(port= Port.B)
rightMotor = Motor(port= Port.A)
base = DriveBase(leftMotor,rightMotor,31.5,110.5)
base.settings(100,100,100,100)

# Write your program here.
#leftMotor.run_angle(500,360)

def drive_Forwards(t,s):
    # leftMotor.run_time(time=t,speed=s)
    # rightMotor.run_time(time=t,speed=s)
    leftMotor.run_angle(s,t,wait=False)
    rightMotor.run_angle(s,t)


def turn(rightspeed,lefspeed):
    leftMotor.run(lefspeed)
    rightMotor.run(rightspeed)

def stop():
    leftMotor.brake()
    rightMotor.brake()



gyroScope = GyroSensor(port=Port.S1)

def newTurnToAngle(angle):
    gyroScope.reset_angle(0)
    desired_angle = gyroScope.angle() + angle
    base.turn(angle,Stop.BRAKE)
    newAngle = gyroScope.angle()-desired_angle
    base.turn(newAngle,Stop.BRAKE,wait=False)
    

def turnToAngle(angle,turnRight):
    keepTurning = True
    rotation = 0
    gyroScope.reset_angle(0)
    rightSpeed = 200
    rightSpeed = rightSpeed*-1 if turnRight else rightSpeed
    leftspeed = rightSpeed*-1
    turnedSpeed = False

    while(keepTurning):
        rotation = gyroScope.angle()
        turn(rightSpeed,leftspeed)
        if(abs(rotation)>angle and not turnedSpeed):
            stop()
            wait(100)
            rotation = gyroScope.angle()
            rightSpeed =(rightSpeed*-1)/2
            leftspeed =(leftspeed*-1)/2
            turnedSpeed = True


        elif(abs(rotation)<angle and turnedSpeed):
            stop()
            wait(100)
            rotation = gyroScope.angle()
            rightSpeed =(rightSpeed*-1)/2
            leftspeed =(leftspeed*-1)/2
            turnedSpeed = False
            
        elif(abs(rotation) == angle):
            stop()
            wait(100)
            rotation = gyroScope.angle()
            if(abs(rotation) == angle):
                keepTurning = False
                ev3.screen.draw_text(0,1,"Rotation: "+str(rotation))
                wait(2000)
        elif(math.floor(rightSpeed)==0):
            keepTurning = False

            

newTurnToAngle(90)




