import json

from CoraData import *
from torch_geometric.datasets import PPI

# PPISplitData与PPIData是利用了pyg数据集写的类
# PPIDataFromJson是直接读取json文件和.npy文件得到的数据


class PPISplitData:
    def __init__(self, split="train"):
        self.ppi_model = None
        self.num_nodes = 0
        self.num_edges = 0
        self.num_of_class = 0
        self.feature_dim = 0
        self.feature_of_pg = None
        self.edge_of_pg = None
        self.label_of_pg = None
        self.undirected = True
        self.split = split

        self.generate_ppi_model(split=split)

    # 产生PPI数据集
    def generate_ppi_model(self, split="train"):
        ppi_model = PPI(root="../ppi", split=split)
        data_of_ppi = ppi_model.data
        self.num_nodes = data_of_ppi.num_nodes
        self.num_edges = data_of_ppi.num_edges
        self.num_of_class = ppi_model.num_classes
        self.feature_dim = data_of_ppi.num_node_features
        self.feature_of_pg = data_of_ppi['x']
        self.edge_of_pg = data_of_ppi['edge_index']
        self.label_of_pg = data_of_ppi['y']
        self.undirected = data_of_ppi.is_undirected()

    # 打印信息
    def print(self):
        print(f"num_nodes of "+self.split+f" set is {self.num_nodes}")
        print(f"num_edges of dataset is {self.num_edges}")
        print(f"num_of_class of dataset is {self.num_of_class}")
        print(f"feature_dim of data is {self.feature_dim}")
        print(f"shape of nodes feature is {self.feature_of_pg.size()}")
        print(f"shape of edge_index is {self.edge_of_pg.size()}")
        print(f"shape of label of nodes is {self.label_of_pg.size()}")
        print(f"the edge_index is undirected: {self.undirected}")
        return


class PPIData:
    def __init__(self):
        self.split_train_set = PPISplitData(split="train")
        self.split_val_set = PPISplitData(split="val")
        self.split_test_set = PPISplitData(split="test")

        self.num_nodes = 0
        self.num_edges = 0
        self.num_of_class = 0
        self.feature_dim = 0
        self.feature_of_pg = None
        self.edge_of_pg = None
        self.label_of_pg = None
        self.undirected = True

        self.generate_whole_dataset()

    # 将三个数据集拼接在一起
    def generate_whole_dataset(self):
        self.edge_of_pg = (torch.cat([self.split_train_set.edge_of_pg, self.split_val_set.edge_of_pg,
                                      self.split_test_set.edge_of_pg], dim=-1).t()).numpy()
        self.num_nodes = self.split_train_set.num_nodes + self.split_val_set.num_nodes + self.split_test_set.num_nodes
        self.num_edges = self.split_train_set.num_edges + self.split_val_set.num_edges + self.split_test_set.num_edges
        self.num_of_class = self.split_train_set.num_of_class
        self.feature_dim = self.split_train_set.feature_dim
        self.feature_of_pg = (torch.cat([self.split_train_set.feature_of_pg, self.split_val_set.feature_of_pg,
                                        self.split_test_set.feature_of_pg], dim=0)).numpy()
        self.label_of_pg = (torch.cat([self.split_train_set.label_of_pg, self.split_val_set.label_of_pg,
                                      self.split_test_set.label_of_pg], dim=0)).numpy()
        return

    # 产生mask
    def data_partition_node(self):
        train_num = self.split_train_set.num_nodes
        val_num = self.split_val_set.num_nodes
        test_num = self.split_test_set.num_nodes
        train_mask = torch.arange(0, train_num)
        val_mask = torch.arange(train_num, train_num+val_num)
        test_mask = torch.arange(train_num+val_num, train_num+val_num+test_num)
        return train_mask, val_mask, test_mask

    # 打印信息
    def print(self):
        print(f"num_nodes of set is {self.num_nodes}")
        print(f"num_edges of dataset is {self.num_edges}")
        print(f"num_of_class of dataset is {self.num_of_class}")
        print(f"feature_dim of data is {self.feature_dim}")
        print(f"shape of nodes feature is {self.feature_of_pg.shape}")
        print(f"shape of edge_index is {self.edge_of_pg.shape}")
        print(f"shape of label of nodes is {self.label_of_pg.shape}")
        print(f"the edge_index is undirected: {self.undirected}")
        return


class PPIDataFromJson:
    def __init__(self, path_of_ppi):
        self.ppi_path = path_of_ppi
        self.feature_of_pg = None
        self.label_of_pg = None

        self.edge_of_pg = []
        self.train_mask = []
        self.test_mask = []
        self.val_mask = []
        self.num_nodes = 0
        self.num_edges = 0
        self.num_of_class = 0
        self.feature_dim = 0

        self.get_node_feature()
        self.get_edge_index()
        self.get_node_label()

    # 读取节点处的feature信息
    def get_node_feature(self):
        path_of_feature = self.ppi_path + "/ppi-feats.npy"
        self.feature_of_pg = np.load(path_of_feature)
        self.num_nodes = len(self.feature_of_pg)
        self.feature_dim = len(self.feature_of_pg[0])
        return

    # 读取节点连接信息以及划分数据集的函数
    def get_edge_index(self):
        graph = self.ppi_path + "/ppi-G.json"
        with open(graph, 'r', encoding='utf-8') as fp:
            json_format = json.load(fp)
            for nodes in json_format['nodes']:
                test_bool = nodes['test']
                node_id = int(nodes['id'])
                val_bool = nodes['val']
                if test_bool:
                    self.test_mask.append(node_id)
                elif val_bool:
                    self.val_mask.append(node_id)
                else:
                    self.train_mask.append(node_id)

            for edges in json_format['links']:
                source = edges['source']
                target = edges['target']
                if source != target:
                    self.edge_of_pg.append([source, target])
                    self.edge_of_pg.append([target, source])
        self.num_edges = len(self.edge_of_pg)
        return

    # 读取节点处标签
    def get_node_label(self):
        labels = self.ppi_path + "/ppi-class_map.json"
        with open(labels, 'r', encoding='utf-8') as fp:
            json_format = json.load(fp)
            self.num_of_class = len(json_format['0'])
            self.label_of_pg = np.ones([len(json_format), len(json_format['0'])], dtype=float)
            for label in json_format.keys():
                key = int(label)
                self.label_of_pg[key] = np.array(json_format[label])
        return

    def data_partition_node(self):
        train_mask_tensor = torch.tensor(self.train_mask, dtype=torch.long)
        val_mask_tensor = torch.tensor(self.val_mask, dtype=torch.long)
        test_mask_tensor = torch.tensor(self.test_mask, dtype=torch.long)
        return train_mask_tensor, val_mask_tensor, test_mask_tensor


if __name__ == "__main__":
    path_of_ppi = "../ppi/ppi"
    ppi = PPIDataFromJson(path_of_ppi)
    print(1)
