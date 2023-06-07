#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile
import math

#Defines the robots configurations. 
ev3 = EV3Brick()
right_motor = Motor(port=Port.A)
grabber = Motor(port=Port.B)
unloader = Motor(port=Port.C)
left_motor = Motor(port=Port.D)

sensor = UltrasonicSensor(port=Port.S1)
gyro = GyroSensor(port=Port.S4)

robot = DriveBase(right_motor, left_motor, wheel_diameter=31.5, axle_track=107.77)

#Steering and overshooting for the margin of our failure rate. 
steering = 0.6666666666666666666666666666666666
overshoot = 0.055555555555555555555555555555555

#Not used very often - makes the robot beep. 
def playSound():
    ev3.speaker.beep()

#Not used very often - For displaying Text
def DisplayText(text):
    ev3.screen.clear()
    ev3.screen.draw_text(1,1,text=text)

#This function makes the robot drive straight
def moveForward(distance):
    robot.straight(distance, wait=True)

#Drives the robot backwards
def moveBackwards(distance):
    robot.straight(-distance, wait=True)

#Turns the robot right
def turnToAngleRight(angle):
    desiredAngle = gyro.angle()+angle
    robot.drive(0, (40))
    while gyro.angle() < (desiredAngle - (overshoot*angle)):
        wait(1)
    robot.drive(0, 0)

#Turns the robot left
def turnToAngleLeft(angle):
    desiredAngle = gyro.angle()-angle
    robot.drive(0, (-40))
    while gyro.angle() > (desiredAngle - (overshoot*angle)):
        wait(1)
    robot.drive(0, 0)

#Drives slow towards the ball and grabs the ball with the grapper. 
def grab_ball(distance):
    while True:
        robot.straight(distance, wait=False)
        if sensor.distance() < 70 and sensor.distance() > 60:
             robot.state()
             grabber.run_angle(600, 1620)
             grabber.run_angle(600, -1620)
#Unloads balls
def unload_balls(): 







