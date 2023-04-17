#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile


ev3 = EV3Brick()

right_motor = Motor(port=Port.A)
left_motor = Motor(port=Port.B)
grabber_motor = Motor(port=Port.C)

robot = DriveBase(right_motor, left_motor, wheel_diameter=34.2, axle_track=111)


robot.turn(90)
robot.turn(90)
robot.turn(90)
robot.turn(90)