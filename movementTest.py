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
leftMotor = Motor(port= Port.A)


def stop():
    leftMotor.brake()



lastButton = Button.CENTER
while True:
    pressed = ev3.buttons.pressed()
    
    ev3.screen.draw_text(1,1,text=pressed)

    if lastButton != pressed:
        stop()

 
    if(Button.UP in pressed):
        leftMotor.run(600)
    if(Button.DOWN in pressed):
        leftMotor.run(-600)
    if(Button.LEFT in pressed):
        break

    lastButton = pressed
