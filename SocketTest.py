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
leftmotor = Motor(port=Port.D)
rightmotor = Motor(port=Port.B)
grapper = Motor(port=Port.C)
unloader = Motor(port=Port.A)




def playSound():
    ev3.speaker.beep()

def DisplayText(text):
    ev3.screen.clear()
    ev3.screen.draw_text(1,1,text=text)

def moveMotor(degree):
    leftMotor = Motor(port= Port.A)
    leftMotor.run_angle(100,degree,wait=False)

def moveForward(time):
    leftmotor.run_time(600,time,wait=False, then= Stop.BRAKE)
    rightmotor.run_time(600,time,wait=False, then= Stop.BRAKE)

def turn(leftSpeed,RightSpeed,time):
    leftmotor.run_time(leftSpeed,time,wait=False, then= Stop.BRAKE)
    rightmotor.run_time(RightSpeed,time,wait=False, then= Stop.BRAKE)

def grap(unload, angle):
    if(unload):
        angle *= -1
    grapper.run_angle(400,angle, wait= False)

def unload(close, angle):
    if(close):
        angle *= -1
    unloader.run_angle(400,angle, wait= False)

