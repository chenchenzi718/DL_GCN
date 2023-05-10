from CoraData import *


# 读取Citeseer数据
class CiteseerData:
    def __init__(self, path_of_citeseer):
        self.path_of_citeseer = path_of_citeseer

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

    def __dataset_loader(self):
        # path = "./cora/cora"
        path_cites = self.path_of_citeseer + "/citeseer.cites"
        path_contents = self.path_of_citeseer + "/citeseer.content"

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

                # 需要注意citeseer是错误的数据集，这里做特殊处理，将.content中不存在的节点对应的边删除
                if (cited not in self.index_of_pg.keys()) or (citing not in self.index_of_pg.keys()):
                    continue

                # 本身为有向图，这里设置为无向
                edge_fir = [self.index_of_pg[citing], self.index_of_pg[cited]]
                edge_sec = [self.index_of_pg[cited], self.index_of_pg[citing]]
                if edge_fir not in self.edge_of_pg:
                    self.edge_of_pg.append([self.index_of_pg[citing], self.index_of_pg[cited]])
                if edge_sec not in self.edge_of_pg:
                    self.edge_of_pg.append([self.index_of_pg[cited], self.index_of_pg[citing]])

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
        half_edge_num = int(len(self.edge_of_pg) / 2)
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
        return d_hat.dot(adj).dot(d_hat).tocoo()  # 返回coo_matrix形式

    # 输出一些可用的data
    def putout_data(self):
        print(f"the shape of feature is {len(self.feature_of_pg), len(self.feature_of_pg[0])}")
        print(f"the shape of label is {len(self.label_of_pg)}")
        print(f"the shape of edge is {len(self.edge_of_pg), len(self.edge_of_pg[0])}")

    # 划分数据分类训练，验证，测试集
    @staticmethod
    def data_partition_node(data_size=3312):
        mask = torch.randperm(data_size)
        train_mask = mask[:180]
        val_mask = mask[180:800]
        test_mask = mask[2312:3312]
        return train_mask, val_mask, test_mask
