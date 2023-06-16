#!/usr/bin/python
"""
author:zgb
"""
import math
import numpy
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


"""
采用二叉树保存8组逆解（效率相对较低），
也可以将theta1--theta6值均作为类的成员，在进行下一组求解过程时采用嵌套for循环获得下组解，
最后对列表中的解进行解析，获得8组解
"""


class Kinematic:
    def __init__(self):
        self.link_params = [LinkParam(0, math.pi / 2, d1, 0),
                            LinkParam(0, 0, 0, a2),
                            LinkParam(0, 0, 0, a3),
                            LinkParam(0, math.pi / 2, d4, 0),
                            LinkParam(0, -math.pi / 2, d5, 0),
                            LinkParam(0, 0, d6, 0)]
        self.result = numpy.zeros((8, 6))  # 逆解结果保存

    @staticmethod
    def adjacent_trans_sdf_matrix(link_param):
        trans_mat = [0.0] * 16
        trans_mat[0] = math.cos(link_param.theta)
        trans_mat[1] = -math.sin(link_param.theta) * math.cos(link_param.alpha)
        trans_mat[2] = math.sin(link_param.theta) * math.sin(link_param.alpha)
        trans_mat[3] = link_param.a * math.cos(link_param.theta)
        trans_mat[4] = math.sin(link_param.theta)
        trans_mat[5] = math.cos(link_param.theta) * math.cos(link_param.alpha)
        trans_mat[6] = -math.cos(link_param.theta) * math.sin(link_param.alpha)
        trans_mat[7] = link_param.a * math.sin(link_param.theta)
        trans_mat[8] = 0
        trans_mat[9] = math.sin(link_param.alpha)
        trans_mat[10] = math.cos(link_param.alpha)
        trans_mat[11] = link_param.d
        trans_mat[12:] = [0, 0, 0, 1]
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
            self.link_params[i].theta = q[i]

    # 正运动学 返回末端空间的齐次矩阵
    def forward_kinematic(self, q):
        self.set_link_params(q)
        trans_mats = []  # 保存所有变换矩阵
        for i in range(6):
            trans_temp = self.adjacent_trans_sdf_matrix(self.link_params[i])
            trans_mats.append(trans_temp)
        np_trans_mat = numpy.array(trans_mats)
        np_trans_06 = numpy.eye(4)
        for i in range(6):
            np_trans_06 = numpy.dot(np_trans_06, np_trans_mat[i].reshape((4, 4)))
        return np_trans_06

    # 从二叉树中获得逆解结果 二叉树的排列顺序
    #   root --> theta1 --> theta5 --> theta6 --> theta3 --> theta2 --> theta4
    @staticmethod
    def __get_result_from_tree(result_tree):
        result = numpy.zeros((8, 6))
        count = 0
        for theta4_node in result_tree.get_seq_node_index(6):
            theta2_node = theta4_node.parent
            theta3_node = theta2_node.parent
            theta6_node = theta3_node.parent
            theta5_node = theta6_node.parent
            theta1_node = theta5_node.parent
            theta_data = [theta1_node.data, theta2_node.data,
                          theta3_node.data, theta4_node.data,
                          theta5_node.data, theta6_node.data]
            for index in range(len(theta_data)):
                theta_data[index] = Kinematic.convert2_2pi_range(theta_data[index])
            result[count, ...] = numpy.array(theta_data)
            count += 1
        return result

    # 逆运动学 针对末端齐次矩阵 返回所有可能的解
    def inverse_kinematic(self, trans_mat):
        result_tree = bt.BinaryTree([1])  # 结果保存树 根节点 不表示关节角度值
        result_tree.root.is_root = True

        trans_mat = numpy.array(trans_mat).reshape(4, 4)
        n_ = trans_mat[:, 0]
        o_ = trans_mat[:, 1]
        a_ = trans_mat[:, 2]
        p_ = trans_mat[:, 3]
        # Joint1
        m = d6 * a_[1] - p_[1]
        n = d6 * a_[0] - p_[0]
        r = math.pow(m, 2) + math.pow(n, 2) - math.pow(d4, 2)
        if r > 0:  # r < 0 奇异位置
            theta1 = math.atan2(m, n) - math.atan2(d4, math.sqrt(r))
            result_tree.root.left_child = bt.Node(theta1, parent=result_tree.root)
            theta1 = math.atan2(m, n) - math.atan2(d4, -math.sqrt(r))
            result_tree.root.right_child = bt.Node(theta1, parent=result_tree.root)

        # Joint5
        # 根据theta1的取值计算theta5
        for theta1_node in result_tree.get_seq_node_index(1):
            temp = a_[0] * math.sin(theta1_node.data) - a_[1] * math.cos(theta1_node.data)
            if math.fabs(temp) <= 1:
                theta1_node.left_child = bt.Node(math.acos(temp), parent=theta1_node)
                theta1_node.right_child = bt.Node(-math.acos(temp), parent=theta1_node)

        # Joint6
        # 根据theta1 theta5的取值计算theta6
        for theta5_node in result_tree.get_seq_node_index(2):
            theta1_node = theta5_node.parent
            m = n_[0] * math.sin(theta1_node.data) - n_[1] * math.cos(theta1_node.data)
            n = o_[0] * math.sin(theta1_node.data) - o_[1] * math.cos(theta1_node.data)
            theta6 = math.atan2(m, n) - math.atan2(math.sin(theta5_node.data), 0)
            theta5_node.left_child = bt.Node(theta6, parent=theta5_node)  # 对于只有一个解的情况，将其放在左子树

        # Joint3
        # 根据theta1 theta6计算theta3
        for theta6_node in result_tree.get_seq_node_index(3):
            theta1_node = theta6_node.parent.parent
            c1 = math.cos(theta1_node.data)
            s1 = math.sin(theta1_node.data)
            c6 = math.cos(theta6_node.data)
            s6 = math.sin(theta6_node.data)

            m = d5 * (s6 * (n_[0] * c1 + n_[1] * s1) + c6 * (o_[0] * c1 + o_[1] * s1)) - d6 * (
                    a_[0] * c1 + a_[1] * s1) + p_[0] * c1 + p_[1] * s1
            n = p_[2] - d1 - a_[2] * d6 + d5 * (o_[2] * c6 + n_[2] * s6)
            if math.pow(m, 2) + math.pow(n, 2) <= math.pow(a2 + a3, 2):
                temp = math.acos((m * m + n * n - a2 * a2 - a3 * a3) / (2 * a2 * a3))
                theta6_node.left_child = bt.Node(temp, parent=theta6_node)
                theta6_node.right_child = bt.Node(-temp, parent=theta6_node)

        # Joint2
        # 利用theta6 theta1 theta3 计算theta2
        for theta3_node in result_tree.get_seq_node_index(4):
            theta6_node = theta3_node.parent
            theta1_node = theta6_node.parent.parent
            c1 = math.cos(theta1_node.data)
            s1 = math.sin(theta1_node.data)
            c6 = math.cos(theta6_node.data)
            s6 = math.sin(theta6_node.data)
            c3 = math.cos(theta3_node.data)
            s3 = math.sin(theta3_node.data)
            m = d5 * (s6 * (n_[0] * c1 + n_[1] * s1) + c6 * (o_[0] * c1 + o_[1] * s1)) - d6 * (
                    a_[0] * c1 + a_[1] * s1) + p_[0] * c1 + p_[1] * s1
            n = p_[2] - d1 - a_[2] * d6 + d5 * (o_[2] * c6 + n_[2] * s6)
            s2 = ((a3 * c3 + a2) * n - a3 * s3 * m) / (a2 * a2 + a3 * a3 + 2 * a2 * a3 * c3)
            c2 = (m + a3 * s3 * s2) / (a3 * c3 + a2)
            theta2 = math.atan2(s2, c2)
            theta3_node.left_child = bt.Node(theta2, parent=theta3_node)

        # Joint4
        # 利用theta6 theta1 theta2 theta3计算
        for theta2_node in result_tree.get_seq_node_index(5):
            theta3_node = theta2_node.parent
            theta6_node = theta3_node.parent
            theta1_node = theta6_node.parent.parent
            s1 = math.sin(theta1_node.data)
            c1 = math.cos(theta1_node.data)
            s6 = math.sin(theta6_node.data)
            c6 = math.cos(theta6_node.data)
            theta4 = math.atan2(-s6 * (n_[0] * c1 + n_[1] * s1) - c6 * (o_[0] * c1 + o_[1] * s1),
                                o_[2] * c6 + n_[2] * s6) - theta2_node.data - theta3_node.data
            theta2_node.left_child = bt.Node(theta4, parent=theta2_node)

            self.result = Kinematic.__get_result_from_tree(result_tree)
        return self.result

    def get_best_sol(self, weights, q_desire):
        if numpy.all(self.result != 0):
            best_sol_ind = numpy.argmin(numpy.sum((weights * (self.result - numpy.array(q_desire))) ** 2, 1))
            return self.result[best_sol_ind]
        return None



def main():
    c = Kinematic()

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
