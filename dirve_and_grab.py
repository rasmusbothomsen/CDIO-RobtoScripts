#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile



ev3 = EV3Brick()
ballsensor = UltrasonicSensor(port=Port.S1)
right_motor = Motor(port=Port.A)
left_motor = Motor(port=Port.B)
grabber_motor = Motor(port=Port.C)

robot = DriveBase(right_motor, left_motor, wheel_diameter=34.2, axle_track=111)



def driveBoth():
    



def search_drive(): 
    driveBoth()
    while True: 
        if ballsensor.distance() < 70 and ballsensor.distance() > 60:
                grab_ball()


def grab_ball(): 
    robot.drive(0,0)
    grabber_motor.run_angle(600, 1620)
    grabber_motor.run_angle(600, -1620)
    drive_and_grab()

def drive_and_grab(): 
    while True:
        robot.drive(150,0)
        if ballsensor.distance() < 250: 
            search_drive()


drive_and_grab()