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
left_motor = Motor(port=Port.D)

robot = DriveBase(left_motor=left_motor, right_motor=right_motor, wheel_diameter=31.5, axle_track=111)
gyro_sensor = GyroSensor(Port.S4)
steering = 0.6666666666666666666666666666666666
overshoot = 0.055555555555555555555555555555555

def turnToAngleCW(angle):
    desiredAngle = gyro_sensor.angle()+angle
    robot.drive(0, (40))
    while gyro_sensor.angle() < (desiredAngle - (overshoot*angle)):
        wait(1)
    robot.drive(0, 0)


def turnToAngleCCW(angle):
    desiredAngle = gyro_sensor.angle()-angle
    robot.drive(0, (-40))
    while gyro_sensor.angle() > (desiredAngle - (overshoot*angle)):
        wait(1)
    robot.drive(0, 0)


gyro_sensor.reset_angle(0)

turnToAngleCCW(-90)
wait(200)
turnToAngleCCW(-90)
wait(200)
turnToAngleCCW(-90)
wait(200)
turnToAngleCCW(-90)
wait(200)
ev3.screen.clear()
ev3.screen.draw_text(0,1,"Rotation: "+str(gyro_sensor.angle()))

wait(2000)

turnToAngleCW(90)
wait(200)
turnToAngleCW(90)
wait(200)
turnToAngleCW(90)
wait(200)
turnToAngleCW(90)
wait(200)
ev3.screen.clear()
ev3.screen.draw_text(0,1,"Rotation: "+str(gyro_sensor.angle()))
