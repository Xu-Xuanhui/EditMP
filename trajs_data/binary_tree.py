# 二叉树的具体实现
class Node:
    def __init__(self, data=0, left_child=None, right_child=None, parent=None):
        self.data = data
        self.left_child = left_child  # 左孩子
        self.right_child = right_child  # 右孩子
        self.parent = parent
        self.is_root = False

    def __str__(self):
        return str(self.data)


'''
针对Node的操作
'''


# 先序遍历构建树，并根据先序遍历顺序记录其父节点
def create_binary_tree(data_input):
    if not data_input:
        return None
    if data_input[0] != 0:
        node = Node(data_input[0])  # 给Root赋给空间并且初始化
        data_input.pop(0)  # pop处第一个数据，这样第一个数据就是原来的第二个位置上的数据如原来是[1,2,3]这样就变成[2,3]
        # print(data_input)  # 输出当前列表中的元素
        node.left_child = create_binary_tree(data_input)
        if node.left_child is not None:
            node.left_child.parent = node
        node.right_child = create_binary_tree(data_input)
        if node.right_child is not None:
            node.right_child.parent = node
        return node  # 将得到的节点返回
    else:
        node = None  # 当data=0的时候就递归停止，就不继续产生节点
        data_input.pop(0)


# 由于采用先序遍历构造树，所以只有先序遍历才能正确导出其父节点
def pre_order(root):
    if root is not None:
        if root.parent is not None:
            print(root.data, '<--', root.parent.data, end=' ')
        else:
            print(root.data, '<--', None, end=' ')
        pre_order(root.left_child)
        pre_order(root.right_child)


# 用来生成先序遍历输入的链表
def pre_order_output(root, out_data):
    if root is not None:
        out_data.append(root.data)
        pre_order_output(root.left_child, out_data)
        pre_order_output(root.right_child, out_data)
    else:
        out_data.append(0)


# 层序遍历，返回按层排布的列表节点结果
def sequence_order(root):
    if not root:
        return

    current_line = 0
    queue = [[current_line, root]]

    line_seq_node = [root]
    out_seq_node = [line_seq_node[:]]
    # print(line_seq_data)
    line_seq_node.clear()
    while len(queue) > 0:
        line, node = queue.pop(0)
        if line != current_line:
            current_line = line
            out_seq_node.append(line_seq_node[:])  # 通过切片获得列表的复制
            # print(line_seq_data)
            line_seq_node.clear()
        if node.left_child:
            queue.append([current_line + 1, node.left_child])  # 将本节点的左子节点入队
            line_seq_node.append(node.left_child)
        if node.right_child:
            queue.append([current_line + 1, node.right_child])  # 将本节点的右子节点入队
            line_seq_node.append(node.right_child)
    # print(out_seq_data)
    return out_seq_node


# 二叉树对外接口 封装Node
class BinaryTree:
    def __init__(self, data_input):  # 初始化变量
        self.root = create_binary_tree(data_input)
        # self.root.is_root = True
        self.__seq_data = []  # 按层排布的列表节点内容
        self.__seq_node = []  # 按层排布的列表节点
        # self.__b_update = False  # 标记是否__seq_node被更新

    # 针对一个parent属性未被赋值的树 重新构建获得parent属性
    def set_parent(self):
        out_data = []
        pre_order_output(self.root, out_data)
        self.root = create_binary_tree(out_data)

    # 获得同属于一层的树节点集合
    def get_seq_node(self):
        self.__seq_node = sequence_order(self.root)
        return self.__seq_node

    def get_seq_node_index(self, index):
        self.get_seq_node()
        if not self.__seq_node:
            print('None tree')
            return
        elif 0 <= index < len(self.__seq_node):
            # print(len(self.__seq_node))
            return self.__seq_node[index]
        else:
            print(len(self.__seq_node))
            print('index out of range')

    # 获得同属于一层的树节点集合的数据
    def get_seq_data(self):
        self.get_seq_node()
        self.__seq_data = []
        for node_list in self.__seq_node:
            self.__seq_data.append([node.data for node in node_list])
        return self.__seq_data

    # 先序遍历并打印出当前节点对应的父节点
    def pre_order(self):
        return pre_order(self.root)


def main():
    # 创建方式1
    data_input = [2, 3, 4, 0, 5, 0, 0, 0, 6, 7, 0, 0, 8, 0, 0]  # 创建二叉树需要的数据当data=0的时候就递归停止
    root = BinaryTree(data_input)
    print(root.root)
    root.pre_order()  # 2 <-- None 3 <-- 2 4 <-- 3 5 <-- 4 6 <-- 2 7 <-- 6 8 <-- 6
    print()
    print(len(root.get_seq_node()))
    print(root.get_seq_node_index(0)[0])
    for node in root.get_seq_node_index(1):
        print(node.data, end=' ')
    print()
    print(root.get_seq_data())

    # 创建方式2
    node4 = Node(4)
    node5 = Node(5)
    node6 = Node(6)
    node2 = Node(2, node4, node5)
    node3 = Node(3, right_child=node6)
    root2 = Node(1, node2, node3)  # 已完成树的构建但缺少parent信息
    bt = BinaryTree([0])
    bt.root = root2
    bt.pre_order()
    print()
    bt.set_parent()  # 重新构建获得父节点信息
    bt.pre_order()
    print()

    # 创建方式3
    root3, node3_2, node3_3, node3_4, node3_5, node3_6 = [Node(int(x)) for x in '123456']
    root3.left_child = node3_2
    root3.right_child = node3_3
    node3_2.left_child = node3_4
    node3_2.right_child = node3_5
    node3_3.right_child = node3_6

    bt3 = BinaryTree([0])
    bt3.root = root3
    bt3.pre_order()
    print()

    # 测试 树中元素的引用 结果表明 该种访问方式修改了树中的元素
    node = root.get_seq_node_index(0)[0]
    print(node.data)
    node.data = 1
    print(root.get_seq_node_index(0)[0].data)  # 输出结果 2 /n 1
    root.pre_order()  # 1 <-- None 3 <-- 1 4 <-- 3 5 <-- 4 6 <-- 1 7 <-- 6 8 <-- 6


if __name__ == '__main__':
    main()
