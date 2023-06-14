#!/usr/bin/env python3

from ev3dev2.motor import OUTPUT_A, OUTPUT_D, MoveDifferential, SpeedRPM
from ev3dev2.wheel import EV3Tire
from ev3dev2.sensor.lego import GyroSensor
from ev3dev2.sensor import INPUT_4
from time import sleep
STUD_MM = 8

# test with a robot that:
# - uses the standard wheels known as EV3Tire
# - wheels are 16 studs apart
mdiff = MoveDifferential(OUTPUT_D, OUTPUT_A, EV3Tire, 120)
mdiff.gyro = GyroSensor(INPUT_4)
mdiff.gyro.calibrate()
# mdiff.odometry_coordinates_log()

# # Enable odometry
mdiff.odometry_start(0,0)
# mdiff.turn_degrees(20,90,True,True,1,True)
# sleep(1.0)
# mdiff.turn_degrees(20,90,True,True,1,True)
# sleep(1.0)
# mdiff.turn_degrees(20,90,True,True,1,True)
# sleep(1.0)
# mdiff.turn_degrees(20,90,True,True,1,True)
# sleep(1.0)
mdiff.on_for_distance(20,1000,True,True)






# # Disable odometry
mdiff.odometry_stop()