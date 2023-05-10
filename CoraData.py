import random

import numpy as np
import torch
import scipy.sparse as sp


# 读取cora数据
class CoraData:
    def __init__(self, path_of_cora):
        self.cora_path = path_of_cora

        # 重新将论文从0开始编码
        self.index_of_pg = dict()
        # 重新将论文标签以数字形式呈现
        self.index_of_pg_label = dict()

        self.feature_of_pg = []
        self.label_of_pg = []
        self.edge_of_pg = []

        self.__dataset_loader()
        self.num_nodes = len(self.feature_of_pg)
        self.num_edges = len(self.edge_of_pg)
        self.num_of_class = len(self.index_of_pg_label)
        self.feature_dim = len(self.feature_of_pg[0])

    # 读取cora数据
    def __dataset_loader(self):
        # path = "./cora/cora"
        path_cites = self.cora_path + "/cora.cites"
        path_contents = self.cora_path + "/cora.content"

        with open(path_contents, 'r', encoding='utf-8') as file_content:
            for node in file_content.readlines():
                node_cont = node.split()
                self.index_of_pg[node_cont[0]] = len(self.index_of_pg)  # 按照进入字典的顺序进行重新排列文章
                self.feature_of_pg.append([int(i) for i in node_cont[1:-1]])

                label = node_cont[-1]
                if label not in self.index_of_pg_label.keys():
                    self.index_of_pg_label[label] = len(self.index_of_pg_label)  # 按照进入字典的顺序进行重新排列label
                self.label_of_pg.append(self.index_of_pg_label[label])

        with open(path_cites, 'r', encoding='utf-8') as file_cite:
            for edge in file_cite.readlines():
                cited, citing = edge.split()
                # 本身为有向图，这里设置为无向
                edge_fir = [self.index_of_pg[citing], self.index_of_pg[cited]]
                edge_sec = [self.index_of_pg[cited], self.index_of_pg[citing]]
                if edge_fir not in self.edge_of_pg:
                    self.edge_of_pg.append(edge_fir)
                if edge_sec not in self.edge_of_pg:
                    self.edge_of_pg.append(edge_sec)

    # 得到一个稀疏邻接矩阵
    def get_adjacent(self):
        graph_w = np.ones(self.num_edges)
        np_edge = np.array(self.edge_of_pg)
        adj = sp.coo_matrix((graph_w, (np_edge[:, 0], np_edge[:, 1])),
                            shape=[self.num_nodes, self.num_nodes])
        return adj

    # 得到一个随机隐藏边的邻接矩阵
    def random_adjacent_sampler(self, drop_edge=0.1):
        new_edge_of_pg = []
        half_edge_num = int(len(self.edge_of_pg)/2)
        sampler = np.random.rand(half_edge_num)
        for i in range(int(half_edge_num)):
            if sampler[i] >= drop_edge:
                new_edge_of_pg.append(self.edge_of_pg[2 * i])
                new_edge_of_pg.append(self.edge_of_pg[2 * i + 1])
        new_edge_of_pg = np.array(new_edge_of_pg)
        graph_w = np.ones(len(new_edge_of_pg))
        adj = sp.coo_matrix((graph_w, (new_edge_of_pg[:, 0], new_edge_of_pg[:, 1])),
                            shape=[self.num_nodes, self.num_nodes])
        return adj

    # 对邻接矩阵A进行D^{-1/2}AD^{-1/2}操作，但是是否自环可选
    @staticmethod
    def normalization(adj, self_link=True):
        adj = sp.coo_matrix(adj)
        if self_link:
            adj += sp.eye(adj.shape[0])  # 增加自连接
        row_sum = np.array(adj.sum(1))  # 对列求和，得到每一行的度
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_hat = sp.diags(d_inv_sqrt)
        return d_hat.dot(adj).dot(d_hat).tocoo()   # 返回coo_matrix形式

    # 输出一些可用的data
    def putout_data(self):
        print(f"the shape of feature is {len(self.feature_of_pg),len(self.feature_of_pg[0])}")
        print(f"the shape of label is {len(self.label_of_pg)}")
        print(f"the shape of edge is {len(self.edge_of_pg),len(self.edge_of_pg[0])}")

    # 划分数据分类训练，验证，测试集
    @staticmethod
    def data_partition_node(data_size=2708):
        mask = torch.randperm(data_size)
        train_mask = mask[:140]
        val_mask = mask[140:640]
        test_mask = mask[1708:2708]
        return train_mask, val_mask, test_mask


# 划分边分类的训练，验证，测试集，为了链路连接使用
def data_partition_edge(edge_of_pg, num_graph_node):
    # 输出nparray
    # 首先对edge_of_pg做处理，将对称边去除，这里由于我们对数据集的处理就是按照对称边同时放入的，因此取奇数即可
    edge_of_pos_pg = edge_of_pg[::2]
    edge_of_neg_pg = []

    neg_node_num = int(5 * np.sqrt(len(edge_of_pos_pg)))

    if num_graph_node <= neg_node_num:
        # 这一段是在num_graph_node*num_graph_node的上三角阵里找负边
        for i in range(num_graph_node):
            for j in range(i+1, num_graph_node):
                edge = [i, j]
                inverse_edge = [j, i]
                if (edge not in edge_of_pos_pg) and (inverse_edge not in edge_of_pos_pg):
                    edge_of_neg_pg.append(edge)
    else:
        # 随机产生[0,num_graph_node]内的500个随机数
        sampler_row = random.sample(range(0, num_graph_node), neg_node_num)
        for row in sampler_row:
            for col in range(row+1, neg_node_num):
                edge = [row, col]
                inverse_edge = [col, row]
                if (edge not in edge_of_pos_pg) and (inverse_edge not in edge_of_pos_pg):
                    edge_of_neg_pg.append(edge)

    edge_of_pos_pg = np.array(edge_of_pos_pg)
    edge_of_neg_pg = np.array(edge_of_neg_pg)

    num_pos_edge = len(edge_of_pos_pg)
    perm = np.random.permutation(num_pos_edge)  # 随机排列
    num_of_train_pos_edge = int(num_pos_edge * 0.85)  # 取出比例条边
    train_pos = perm[:num_of_train_pos_edge]  # 对随机排列的边取出 前num_of_train_pos_edge条边
    train_pos_edge_index = edge_of_pos_pg[train_pos]
    num_of_val_pos_edge = int(num_pos_edge * 0.05)
    val_pos = perm[num_of_train_pos_edge:(num_of_train_pos_edge+num_of_val_pos_edge)]
    validate_pos_edge_index = edge_of_pos_pg[val_pos]
    test_pos = perm[(num_of_train_pos_edge+num_of_val_pos_edge):]
    test_pos_edge_index = edge_of_pos_pg[test_pos]

    num_neg_edge = len(edge_of_neg_pg)
    perm = np.random.permutation(num_neg_edge)  # 随机排列
    num_of_train_neg_edge = int(num_neg_edge * 0.85)  # 取出比例条边
    train_neg = perm[:num_of_train_neg_edge]  # 对随机排列的边取出 前num_of_train_pos_edge条边
    train_neg_edge_index = edge_of_neg_pg[train_neg]
    num_of_val_neg_edge = int(num_neg_edge * 0.05)
    val_neg = perm[num_of_train_neg_edge:(num_of_train_neg_edge+num_of_val_neg_edge)]
    validate_neg_edge_index = edge_of_neg_pg[val_neg]
    test_neg = perm[(num_of_train_neg_edge+num_of_val_neg_edge):]
    test_neg_edge_index = edge_of_neg_pg[test_neg]

    return train_pos_edge_index, validate_pos_edge_index, test_pos_edge_index, train_neg_edge_index, \
                                                        validate_neg_edge_index, test_neg_edge_index


# 从训练集的neg边中随机采样出和pos边一样数目的边
def negative_edge_sampling(train_neg_edge_index, train_pos_edge_index):
    num_pos_edge = len(train_pos_edge_index)
    num_neg_edge = len(train_neg_edge_index)
    perm = np.random.permutation(num_neg_edge)  # 随机排列
    train_neg = perm[:num_pos_edge]
    sampler_train_neg_edge_index = train_neg_edge_index[train_neg]
    return sampler_train_neg_edge_index


# 将pos边标记为1，neg边标记为0，pos与neg合并为一个label
# pos_edge_index大小为E*2，同理neg_edge_index
def get_link_labels(pos_edge_index, neg_edge_index, device):
    num_of_edge = pos_edge_index.size(0) + neg_edge_index.size(0)
    link_labels = torch.zeros(num_of_edge, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(0)] = 1.
    return link_labels


# 供外部使用，得到一个稀疏邻接矩阵
def get_adjacent(edge_of_pg, num_graph_node, symmetric_of_edge=False):
    if not symmetric_of_edge:
        new_edge_of_pg = convert_symmetric(edge_of_pg)
    else:
        new_edge_of_pg = np.copy(edge_of_pg)
    num_edges = len(new_edge_of_pg)
    graph_w = np.ones(num_edges)
    np_edge = np.array(new_edge_of_pg)
    adj = sp.coo_matrix((graph_w, (np_edge[:, 0], np_edge[:, 1])),
                        shape=[num_graph_node, num_graph_node])

    return adj


# 对edge列表做一次对称
def convert_symmetric(edge_of_pg):
    new_edge_of_pg = []
    for edge_index in edge_of_pg:
        symmetric_edge_index = [edge_index[1], edge_index[0]]
        if symmetric_edge_index not in edge_of_pg:
            new_edge_of_pg.append(symmetric_edge_index)

    new_edge_of_pg.extend(edge_of_pg)
    return np.array(new_edge_of_pg)


# 供外部使用，得到一个随机隐藏边的邻接矩阵
def random_adjacent_sampler(edge_of_pg, num_graph_node, drop_edge=0.1, symmetric_of_edge=False):
    if not symmetric_of_edge:
        new_edge_of_pg = []
        edge_num = int(len(edge_of_pg))
        sampler = np.random.rand(edge_num)
        for i in range(int(edge_num)):
            if sampler[i] >= drop_edge:
                new_edge_of_pg.append(edge_of_pg[i])
        new_edge_of_pg = np.array(new_edge_of_pg)
        new_edge_of_pg = convert_symmetric(new_edge_of_pg)
        graph_w = np.ones(len(new_edge_of_pg))
        adj = sp.coo_matrix((graph_w, (new_edge_of_pg[:, 0], new_edge_of_pg[:, 1])),
                            shape=[num_graph_node, num_graph_node])
    else:
        new_edge_of_pg = []
        half_edge_num = int(len(edge_of_pg) / 2)
        sampler = np.random.rand(half_edge_num)
        for i in range(int(half_edge_num)):
            if sampler[i] >= drop_edge:
                new_edge_of_pg.append(edge_of_pg[2 * i])
                new_edge_of_pg.append(edge_of_pg[2 * i + 1])
        new_edge_of_pg = np.array(new_edge_of_pg)
        graph_w = np.ones(len(new_edge_of_pg))
        adj = sp.coo_matrix((graph_w, (new_edge_of_pg[:, 0], new_edge_of_pg[:, 1])),
                            shape=[num_graph_node, num_graph_node])
    return adj


# 供外部使用，与cora里的normalization函数一致
def normalization(adj, self_link=True):
    adj = sp.coo_matrix(adj)
    if self_link:
        adj += sp.eye(adj.shape[0])  # 增加自连接
    row_sum = np.array(adj.sum(1))  # 对列求和，得到每一行的度
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_hat = sp.diags(d_inv_sqrt)
    return d_hat.dot(adj).dot(d_hat).tocoo()   # 返回coo_matrix形式
