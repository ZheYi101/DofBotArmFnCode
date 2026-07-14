# !/usr/bin/env python
# coding: utf-8
import time

import Arm_Lib


class snake_move:
    def __init__(self):
        self.sbus = Arm_Lib.Arm_Device()
        self.grap_joint = 135

    def arm_move(self, joints_target):
        self.sbus.Arm_serial_servo_write(5, 90, 500)
        time.sleep(1)
        self.sbus.Arm_serial_servo_write(6, 30, 500)
        time.sleep(1.5)
        self.sbus.Arm_serial_servo_write(6, self.grap_joint, 500)
        time.sleep(0.6)
        self.sbus.Arm_serial_servo_write6_array(joints_target, 1000)
        time.sleep(1.5)
        self.sbus.Arm_serial_servo_write(6, 30, 500)
        time.sleep(0.5)
        self.sbus.Arm_serial_servo_write6_array([90, 135, 0, 45, 0, 180], 500)
        time.sleep(1)

    def snake_run(self, name):
        targets = {
            "red": [117, 19, 66, 56, 90, self.grap_joint],
            "blue": [44, 66, 20, 28, 90, self.grap_joint],
            "green": [136, 66, 20, 29, 90, self.grap_joint],
            "yellow": [65, 22, 64, 56, 90, self.grap_joint],
        }
        target = targets.get(name)
        if target is not None:
            self.arm_move(target)
