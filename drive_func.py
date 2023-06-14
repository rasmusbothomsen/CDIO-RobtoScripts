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
leftMotor = Motor(port= Port.D)
rightMotor = Motor(port= Port.A)
#107.77
robot = DriveBase(leftMotor, rightMotor, wheel_diameter=31.5, axle_track=116.0)
robot.settings(100, 100, 40, 40)

robot.turn(360)
robot.turn(90)
robot.turn(90)








