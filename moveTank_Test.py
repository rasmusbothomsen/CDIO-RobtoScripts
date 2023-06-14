#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile
import math
import ev3dev2.motor
from ev3dev2.motor import MoveDifferential

#Defines the robots configurations. 
ev3 = EV3Brick()
right_motor = Motor(port=Port.A)
grabber = Motor(port=Port.B)
unloader = Motor(port=Port.C)
left_motor = Motor(port=Port.D)

sensor = UltrasonicSensor(port=Port.S1)
gyro = GyroSensor(port=Port.S4)

robot = ev3dev2.motor.MoveTank(left_motor, right_motor, desc=None, motor_class='ev3dev2.motor.ServoMotor')
#robot = DriveBase(right_motor, left_motor, wheel_diameter=31.5, axle_track=107.77)

robot.follow_gyro_angle(20,20)