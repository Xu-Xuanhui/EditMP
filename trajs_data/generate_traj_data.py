import numpy as np
from numpy import dot
import random
from numpy.linalg import norm
import matplotlib.pyplot as plt
from trajs_data.ceshi_simulate import *
import mpl_toolkits.mplot3d
import torch
from sympy import *
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.animation as animation
from PIL import Image
global x, y

def tcp2angle(x, y):
    c = Kinematic()
    target_tcpv = np.array([-583.78 / 1000, -153.53 / 1000, -290.72 / 1000 + 0.4, 1.132, -1.331, -1.341])
    target_tcpv[0] = x/ 1000
    target_tcpv[1] = y/ 1000
    # print(target_tcpv[0], target_tcpv[1])

    target_tcpm = tcpv2tcpm(target_tcpv)

    agnles = c.inverse_kinematic(target_tcpm)
    return agnles[0]
def angle2tcp_np(tcp):
    c = Kinematic()
    # print('angele2tcp',tcp.dtype)
    forward_m = c.forward_kinematic(tcp)
    #forward_m = c.forward_kinematic(np.array(inverse_result[0]))
    # print("forward:\n",forward_m[0:3,3]*1000)
    return forward_m[0:3,3]*1000

def generate_tcp(path, epoch):
    
    data = np.load(path)
    
    Length = 8
    print(data.shape)
    for j in range(len(data)):
        # ax = plt.subplot(projection = '3d')
        # ax = plt.plot()

        #plt.figure()
        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        plt.ylim(-300, 300)
        plt.xlim(-600, 100)
        ractangle = data[j]

        x = np.zeros(Length)
        y = np.zeros(Length)
        z = np.zeros(Length)
        #print(angle2tcp(ractangle[0]))
        for i in range(Length):
            x[i] = angle2tcp_np(ractangle[i])[0]
            y[i] = angle2tcp_np(ractangle[i])[1]
            z[i] = angle2tcp_np(ractangle[i])[2]
            # x[i] = ractangle[i][0]
            # y[i] = ractangle[i][1]
            # z[i] = ractangle[i][2]

            plt.scatter(x[i], y[i])
            #ax.text(x[i] + 0.15, y[i] - 0.15, str(i), ha='center', va='bottom', fontsize=10.5)
        # print(x, y, z)
        plt.plot(x, y)
       # ax.plot3D(x,y,z)
    #plt.show()

        plt.savefig('/home/robot/Documents/EditMP/trajs_data/results/' + str(epoch) + '/trajs_'+ str(j) + '.png')
        # plt.savefig('demonstration/trajs_' + str(j) + '.png')




if __name__=="__main__":
    epoch = 0
    path = "/home/robot/Documents/MPGAN/trajs_data/ractangles_8_test.npy"
    generate_tcp(path, epoch)







