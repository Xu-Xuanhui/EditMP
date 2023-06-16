import numpy as np
import socket
import time
import struct
import trajs_data.util
#import rtde

import math
import random
#import pyrealsense2 as rs
import datetime as dt
import sys
from trajs_data.ur_kinematics import *

import numpy as np
from numpy import linalg

#########################

import cmath
import math
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt
from math import pi as pi

global mat
mat = np.matrix

# ****** Coefficients ******


# global d1, a2, a3, a7, d4, d5, d6
# d1 = 0.1273
# a2 = -0.612
# a3 = -0.5723
# a7 = 0.075
# d4 = 0.163941
# d5 = 0.1157
# d6 = 0.0922
#
# global d, a, alph

d = mat([0.089159, 0, 0, 0.10915, 0.09465, 0.0823]) #ur5
# d = mat([0.1273, 0, 0, 0.163941, 0.1157, 0.0922])  # ur10 mm
a =mat([0 ,-0.425 ,-0.39225 ,0 ,0 ,0])# ur5
# a = mat([0, -0.612, -0.5723, 0, 0, 0])  # ur10 mm
alph = mat([math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0 ])  #ur5
# alph = mat([pi / 2, 0, 0, pi / 2, -pi / 2, 0])  # ur10


#########################
np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
#HOST = "192.168.100.100"
HOST = "192.168.1.1" 
PORT = 30003

def AH(n, th, c):
    T_a = mat(np.identity(4), copy=False)
    T_a[0, 3] = a[0, n - 1]
    T_d = mat(np.identity(4), copy=False)
    T_d[2, 3] = d[0, n - 1]

    Rzt = mat([[cos(th[n - 1, c]), -sin(th[n - 1, c]), 0, 0],
               [sin(th[n - 1, c]), cos(th[n - 1, c]), 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]], copy=False)

    Rxa = mat([[1, 0, 0, 0],
               [0, cos(alph[0, n - 1]), -sin(alph[0, n - 1]), 0],
               [0, sin(alph[0, n - 1]), cos(alph[0, n - 1]), 0],
               [0, 0, 0, 1]], copy=False)

    A_i = T_d * Rzt * T_a * Rxa

    return A_i

# def move_to_angle(target_angle):
#     tool_acc = 0.5  # Safe: 0.5
#     tool_vel = 0.1  # Safe: 0.2
#     tool_pos_tolerance = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
#     tcp_command = "movej([%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (
#         target_angle[0], target_angle[1], target_angle[2], target_angle[3], target_angle[4],
#         target_angle[5],
#         tool_acc, tool_vel)
#     tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     tcp_socket.connect((HOST, PORT))
#     tcp_socket.send(str.encode(tcp_command))  # 利用字符串的encode方法编码成bytes，默认为utf-8类型
#     tcp_socket.close()


def tcpv2tcpm(vector):
    pad=np.array([0,0,0,1])
    position=vector[0:3]
    #print(position)
    position=position[:,np.newaxis]
    #print(position)
    rv=vector[3:6]
    rm = trajs_data.util.rv2rm(rv[0],rv[1],rv[2])
    rm_34=np.hstack((rm,position))
    rm_44=np.vstack((rm_34,pad))

    return rm_44

# def get_current_robot_tcp():
#     dic = {'MessageSize': 'i', 'Time': 'd', 'q target': '6d', 'qd target': '6d', 'qdd target': '6d', 'I target': '6d',
#            'M target': '6d', 'q actual': '6d', 'qd actual': '6d', 'I actual': '6d', 'I control': '6d',
#            'Tool vector actual': '6d', 'TCP speed actual': '6d', 'TCP force': '6d', 'Tool vector target': '6d',
#            'TCP speed target': '6d', 'Digital input bits': 'd', 'Motor temperatures': '6d', 'Controller Timer': 'd',
#            'Test value': 'd', 'Robot Mode': 'd', 'Joint Modes': '6d', 'Safety Mode': 'd', 'empty1': '6d',
#            'Tool Accelerometer values': '3d',
#            'empty2': '6d', 'Speed scaling': 'd', 'Linear momentum norm': 'd', 'SoftwareOnly': 'd', 'softwareOnly2': 'd',
#            'V main': 'd',
#            'V robot': 'd', 'I robot': 'd', 'V actual': '6d', 'Digital outputs': 'd', 'Program state': 'd',
#            'Elbow position': '3d', 'Elbow velocity': '3d'}
#     # dic = collections.OrderedDict(dic)
#     dic_list = ['MessageSize', 'Time', 'q target', 'qd target', 'qdd target', 'I target',
#                 'M target', 'q actual', 'qd actual', 'I actual', 'I control',
#                 'Tool vector actual', 'TCP speed actual', 'TCP force', 'Tool vector target',
#                 'TCP speed target', 'Digital input bits', 'Motor temperatures', 'Controller Timer',
#                 'Test value', 'Robot Mode', 'Joint Modes', 'Safety Mode', 'empty1', 'Tool Accelerometer values',
#                 'empty2', 'Speed scaling', 'Linear momentum norm', 'SoftwareOnly', 'softwareOnly2', 'V main',
#                 'V robot', 'I robot', 'V actual', 'Digital outputs', 'Program state', 'Elbow position',
#                 'Elbow velocity']
#     # print(len(dic), len(dic_list))
#     # print(tuple(dic))
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     s.connect((HOST, PORT))
#
#     data = s.recv(1108)
#     # print(data)
#
#     names = []
#     ii = range(len(dic))
#     # print(len(dic))
#     # for key,i in zip(dic,ii):
#     for i, key in enumerate(dic_list):
#         fmtsize = struct.calcsize(dic[key])
#         data1, data = data[0:fmtsize], data[fmtsize:]
#         # print(i, key, fmtsize)
#         fmt = "!" + dic[key]
#         names.append(struct.unpack(fmt, data1))
#         dic[key] = dic[key], struct.unpack(fmt, data1)
#     a = dic["Tool vector actual"]
#     # a = dic["q actual"]
#     a2 = np.array(a[1])
#     # a2[2] -= 0.4
#     return (a2)

# def move_to_tcp(target_tcp):
#     tool_acc = 1.2  # Safe: 0.5
#     tool_vel = 0.25  # Safe: 0.2
#     tool_pos_tolerance = [0.001, 0.001, 0.001, 0.05, 0.05, 0.05]
#     tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (
#         target_tcp[0], target_tcp[1], target_tcp[2], target_tcp[3], target_tcp[4],
#         target_tcp[5],
#         tool_acc, tool_vel)
#     tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     tcp_socket.connect((HOST, PORT))
#     tcp_socket.send(str.encode(tcp_command))  # 利用字符串的encode方法编码成bytes，默认为utf-8类型
#     tcp_socket.close()
#     # 确保已达到目标点，就可以紧接着发送下一条指令
#     actual_pos = get_current_robot_tcp()
#     target_rpy = util.rv2rpy(target_tcp[3], target_tcp[4], target_tcp[5])
#     rpy = util.rv2rpy(actual_pos[3], actual_pos[4], actual_pos[5])
#     while not (all([np.abs(actual_pos[j] - target_tcp[j]) < tool_pos_tolerance[j] for j in range(3)])
#                and all([np.abs(rpy[j] - target_rpy[j]) < tool_pos_tolerance[j+3] for j in range(3)])):
#         actual_pos = get_current_robot_tcp()
#         rpy = util.rv2rpy(actual_pos[3], actual_pos[4], actual_pos[5])
#         time.sleep(0.01)

if __name__=="__main__":
    
    c=Kinematic()
    x = -280.19/1000
    y = -152.28/1000
    # # forward_m=c.forward_kinematic(np.array([-15.26, -69.38, 90.68, -19.50, 98.24, -1.4]) / 180*np.pi)
    # # print("forward:\n",forward_m)
    target_tcpv = np.array([-583.78/1000,-153.53/1000,-290.72/1000 + 0.4,1.132,-1.331,-1.341])
    target_tcpv[0] = x
    target_tcpv[1] = y

    target_tcpm=tcpv2tcpm(target_tcpv)
    # print("target_tcpm:\n",target_tcpm)
    inverse_result = c.inverse_kinematic(target_tcpm)
    print("inverse_result:\n",inverse_result[0]/math.pi*180)
    forward_m = c.forward_kinematic(np.array(inverse_result[0]))
    print("forward:\n",forward_m[0:3,3]*1000)
    target_tcpv = np.array([-583.78 / 1000, -153.53 / 1000, -290.72 / 1000 + 0.4, 1.032, -1.031, -1.341])
    target_tcpv[0] = x
    target_tcpv[1] = y

    target_tcpm = tcpv2tcpm(target_tcpv)
    # print("target_tcpm:\n",target_tcpm)
    inverse_result = c.inverse_kinematic(target_tcpm)
    print("inverse_result:\n", inverse_result[0] / math.pi * 180)
    forward_m = c.forward_kinematic(np.array(inverse_result[0]))
    print("forward:\n", forward_m[0:3, 3] * 1000)
    
