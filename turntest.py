#!/usr/bin/env python3
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile
import math


ev3 = EV3Brick()

right_motor = Motor(port=Port.A)
left_motor = Motor(port=Port.D)

robot = DriveBase(left_motor=left_motor, right_motor=right_motor, wheel_diameter=31.5, axle_track=107.77)
robot.settings(40,40,40,40)
gyro_sensor = GyroSensor(Port.S4)
steering = 0.6666666666666666666666666666666666
overshoot = 0

def turnToAngleCW(angle):
    gyro_sensor.reset_angle(0)
    robot.drive(0, 20)
    while gyro_sensor.angle() < (angle ):
        wait(1)
    robot.drive(0, 0)
    wait(100)
    if(abs(gyro_sensor.angle() - angle) <= 1):
        return
    if gyro_sensor.angle() > angle:
        print("overshot:cw " + str(gyro_sensor.angle() - angle))
        turnToAngleCCW(gyro_sensor.angle() - angle)
    elif gyro_sensor.angle() < angle:
        print("undershot:cw " + str(gyro_sensor.angle() - angle))
        turnToAngleCW(abs(gyro_sensor.angle() - angle))

def turnToAngleCCW(angle):
    gyro_sensor.reset_angle(0)
    robot.drive(0, -20)
    while gyro_sensor.angle() > (angle ):
        wait(1)
    robot.drive(0, 0)
    wait(100)
    if(abs(gyro_sensor.angle() - angle) <= 1):
        print(abs(gyro_sensor.angle()))
        return
    if gyro_sensor.angle() < angle:
        print("overshot:ccw " + str(gyro_sensor.angle() - angle))
        turnToAngleCW(gyro_sensor.angle() - angle)
    elif gyro_sensor.angle() > angle:
        print("undershot:ccw " + str(gyro_sensor.angle() - angle))
        turnToAngleCCW(gyro_sensor.angle() - angle)



# wait(200)
gyro_sensor.reset_angle(0)
robot.turn(90)
robot.turn(90)
robot.turn(90)
robot.turn(90)
print(gyro_sensor.angle())
# wait(200)
# turnToAngleCCW(-90)
# wait(200)
# turnToAngleCCW(-90)
# wait(200)
# ev3.screen.clear()
# ev3.screen.draw_text(0,1,"Rotation: "+str(gyro_sensor.angle()))

# wait(2000)

# turnToAngleCW(90)
# wait(200)
# turnToAngleCW(90)
# wait(200)
# turnToAngleCW(90)
# wait(200)
# turnToAngleCW(90)
# wait(200)
# ev3.screen.clear()
# ev3.screen.draw_text(0,1,"Rotation: "+str(gyro_sensor.angle()))
