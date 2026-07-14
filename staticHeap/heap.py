#!/usr/bin/env python3
#coding=utf-8
import time
from Arm_Lib import Arm_Device
# 创建机械臂对象
Arm = Arm_Device()
time.sleep(.1)
# 定义夹积木块函数，enable=1：夹住，=0：松开
def arm_clamp_block(enable):
     if enable == 0:
         Arm.Arm_serial_servo_write(6, 60, 400)
     else:
         Arm.Arm_serial_servo_write(6, 143, 100)
         time.sleep(.5)
 
# 定义移动机械臂函数,同时控制 1-5 号舵机运动，p=[S1,S2,S3,S4,S5]
def arm_move(p, s_time = 200):
    for i in range(5):
        id = i + 1
        if id == 5:
            time.sleep(.1)
            Arm.Arm_serial_servo_write(id, p[i], int(s_time*1.2))
        elif id == 1 :
            Arm.Arm_serial_servo_write(id, p[i], int(3*s_time/4))
        else:
            Arm.Arm_serial_servo_write(id, p[i], int(s_time))
        time.sleep(.01)
    time.sleep(s_time/1000)
# 定义不同位置的变量参数
p_mould = [90, 130, 0, 0, 90]
p_top = [90, 80, 50, 50, 270]
p_layer_4 = [89, 91, 27, 27, 270]
p_layer_3 = [90, 65, 44, 17, 270]
p_layer_2 = [90, 65, 25, 36, 270]
p_layer_1 = [90, 48, 35, 30, 270]
p_Yellow = [65, 22, 64, 56, 270]
p_Red = [118, 19, 66, 56, 270]
p_Green = [136, 66, 20, 28,270]
p_Blue = [44, 66, 20, 28, 270]
# 让机械臂移动到一个准备抓取的位置
arm_clamp_block(0)
arm_move(p_mould, 1000)
time.sleep(1)
# 夹取黄色区域的方块堆叠到中间最底层的位置。
arm_move(p_top, 1000)
arm_move(p_Yellow, 1000)
arm_clamp_block(1)
arm_move(p_top, 1000)
arm_move(p_layer_1, 1000)
arm_clamp_block(0)
time.sleep(.1)
arm_move(p_mould, 1100)
 
# time.sleep(1)
# 夹取红色区域的方块堆叠到中间第二层的位置。
arm_move(p_top, 1000)
arm_move(p_Red, 1000)
arm_clamp_block(1)
arm_move(p_top, 1000)
arm_move(p_layer_2, 1000)
arm_clamp_block(0)
time.sleep(.1)
arm_move(p_mould, 1100)
# time.sleep(1)
# 夹取绿色区域的方块堆叠到中间第三层的位置。
arm_move(p_top, 1000)
arm_move(p_Green, 1000)
arm_clamp_block(1)

arm_move(p_top, 1000)
arm_move(p_layer_3, 1000)
arm_clamp_block(0)
time.sleep(.1)
arm_move(p_mould, 1100)
 
time.sleep(1)
# 夹取蓝色区域的方块堆叠到中间第四层的位置。
arm_move(p_top, 1000)
arm_move(p_Blue, 1000)
arm_clamp_block(1)
arm_move(p_top, 1000)
arm_move(p_layer_4, 1000)
arm_clamp_block(0)
time.sleep(.1)
arm_move(p_mould, 1100)
 
# time.sleep(1)
del Arm # 释放掉 Arm 对象