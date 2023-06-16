#!/usr/bin/python

import math
import numpy
import numpy as np
import torch
import torch.nn as nn
import trajs_data.binary_tree as bt

# UR5 params
d1 = 0.089459
a2 = -0.42500
a3 = -0.39225
d4 = 0.10915
d5 = 0.09465
d6 = 0.0823
PI = math.pi
ZERO_THRESH = 0.00000001


class LinkParam:
    def __init__(self, theta, alpha, d, a):
        self.theta = theta
        self.alpha = alpha
        self.d = d
        self.a = a

class matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = x1 @ x2
        return x
"""
采用二叉树保存8组逆解（效率相对较低），
也可以将theta1--theta6值均作为类的成员，在进行下一组求解过程时采用嵌套for循环获得下组解，
最后对列表中的解进行解析，获得8组解
"""


class Kinematic:
    def __init__(self):
        self.link_params = [[0, math.pi / 2, d1, 0],
                            [0, 0, 0, a2],
                            [0, 0, 0, a3],
                            [0, math.pi / 2, d4, 0],
                            [0, -math.pi / 2, d5, 0],
                            [0, 0, d6, 0]] #theta, alpha, d, a
        self.link_params = np.array(self.link_params)
        self.link_params = torch.from_numpy(self.link_params).cuda().float()
        self.mat = matmul()
    @staticmethod
    def adjacent_trans_sdf_matrix(link_param):

        trans_mat = torch.ones((len(link_param), 16)).cuda()
        trans_mat[:, 0] = torch.cos(link_param[:, 0])
        trans_mat[:, 1] = -torch.sin(link_param[:, 0]) * torch.cos(link_param[:, 1])
        trans_mat[:, 2] = torch.sin(link_param[:, 0]) * torch.sin(link_param[:, 1])
        trans_mat[:, 3] = link_param[:, 3] * torch.cos(link_param[:, 0])
        trans_mat[:, 4] = torch.sin(link_param[:, 0])
        trans_mat[:, 5] = torch.cos(link_param[:, 0]) * torch.cos(link_param[:, 1])
        trans_mat[:, 6] = -torch.cos(link_param[:, 0]) * torch.sin(link_param[:, 1])
        trans_mat[:, 7] = link_param[:, 3] * torch.sin(link_param[:, 0])
        trans_mat[:, 8] = trans_mat[:, 8] * 0
        trans_mat[:, 9] = torch.sin(link_param[:, 1])
        trans_mat[:, 10] = torch.cos(link_param[:, 1])
        trans_mat[:, 11] = link_param[:, 2]
        trans_mat[:, 12] = trans_mat[:, 12] * 0
        trans_mat[:, 13] = trans_mat[:, 13] * 0
        trans_mat[:, 14] = trans_mat[:, 14] * 0
        trans_mat[:, 15] = 1

        return trans_mat

    @staticmethod
    def sign(x):
        return (x > 0) - (x < 0)

    # 将theta转换为[-2pi, 2pi]范围内
    @staticmethod
    def convert2_2pi_range(theta):
        while theta < -2 * numpy.pi or theta > 2 * numpy.pi:
            theta -= Kinematic.sign(theta) * 2 * numpy.pi
        return theta

    def set_link_params(self, q):
        for i in range(6):
            self.link_params[i, 0] = q[i]

    # 正运动学 返回末端空间的齐次矩阵
    def forward_kinematic(self, q):
        # self.set_link_params(q)

        self.link_params = self.link_params.repeat(len(q), 1, 1)

        self.link_params[:, :, 0] = q



        trans_mats = self.adjacent_trans_sdf_matrix(self.link_params[:, 0]).unsqueeze(1) # 保存所有变换矩阵
        # print(trans_mats.shape)
        for i in range(5):
            trans_temp = self.adjacent_trans_sdf_matrix(self.link_params[:, i+1])

            trans_mats = torch.cat((trans_mats, trans_temp.unsqueeze(1)),1)



        np_trans_06 = numpy.eye(4)
        np_trans_06 = torch.from_numpy(np_trans_06).cuda().float().repeat(len(q), 1, 1)
        # print(trans_mats.shape)

        for i in range(6):

            np_trans_06 = self.mat(np_trans_06, trans_mats[:, i, :].view((len(q), 4, 4)))
        return np_trans_06[:,0:3, 3]


def angle2tcp(angel):
    # print(angel.shape)
    c = Kinematic()
    d = c.forward_kinematic(angel)
    # print(d.shape)
    return d



def main():
    c = Kinematic()
    angel = torch.Tensor([14.99070, -90.58562,  142.90221, -52.25525,  114.80469, -0.54981])/180*math.pi
    angel = angel.cuda().repeat(64, 1)

    d = c.forward_kinematic(angel)


    # -280.18999745 -152.27998286  109.27994883
    # test2
    # q = [1]*6
    # q = [1, 0.5, 1, 1, 1, 1]
    # target = c.forward_kinematic(q)
    # print(target)
    # print(target.shape)
    # print("+" * 10)
    # inverse_result = c.inverse_kinematic(target)
    # print(inverse_result)
    # # print()
    # print("+" * 10)
    # for index in range(6):
    #     result = inverse_result[index, ...]
    #     print(c.forward_kinematic(result))
    #     print()
    # print("+" * 10)
    # q_desire = [-1.5, 1.2, 1.6, -1.8, 2.5, -1.4]
    # best_result = c.get_best_sol([1]*6, q_desire)
    # print(best_result)

    # test3
    # theta = numpy.pi
    # theta = c.convert2_2pi_range(theta)
    # print(theta)


if __name__ == '__main__':
    main()
    # link_params = [[0, math.pi / 2, d1, 0],
    #                     [0, 0, 0, a2],
    #                     [0, 0, 0, a3],
    #                     [0, math.pi / 2, d4, 0],
    #                     [0, -math.pi / 2, d5, 0],
    #                     [0, 0, d6, 0]]  # theta, alpha, d, a
    # link_params = np.array(link_params)
    # link_params = torch.from_numpy(link_params).cuda().float().repeat(2,1,1)
    # angel = torch.Tensor([14.99070, -90.58562, 142.90221, -52.25525, 114.80469, -0.54981])
    # angel = angel.cuda().repeat(2,1)
    # print(link_params[:,:,0])
    # link_params[:,:,0] = angel
    # print(link_params)
