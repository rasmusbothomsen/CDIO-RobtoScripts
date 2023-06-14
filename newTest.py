#!/usr/bin/env python3

from ev3dev2.motor import OUTPUT_A, OUTPUT_D, MoveDifferential, SpeedRPM
from ev3dev2.wheel import EV3Tire, Wheel
from ev3dev2.sensor.lego import GyroSensor
import SocketTest
from ev3dev2.sensor import INPUT_4
from time import sleep
from SocketServer.catterpillerWheel import catterpillerWheel
import logging
logging.basicConfig( level=logging.DEBUG,
    format='%(message)s')
STUD_MM = 8


mdiff = MoveDifferential(OUTPUT_D, OUTPUT_A,wheel_class= catterpillerWheel,wheel_distance_mm=116.0)

mdiff.gyro = GyroSensor(INPUT_4)
mdiff.gyro.calibrate()
# mdiff.odometry_coordinates_log()

# # Enable odometry
mdiff.odometry_start(0,0)
mdiff.on_for_distance(20,500)






# mdiff.turn_degrees(20,90,True,True,1,True)
# sleep(1.0)
# mdiff.turn_degrees(20,90,True,True,1,True)
# sleep(1.0)
# mdiff.turn_degrees(20,90,True,True,1,True)
# sleep(1.0)
# mdiff.turn_degrees(20,90,True,True,1,True)
# sleep(1.0)
# # mdiff.on_for_distance(20,1000,True,True)
# # Disable odometry
mdiff.odometry_stop()