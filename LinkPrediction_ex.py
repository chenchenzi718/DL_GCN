import torch

from CoraData import *
from MyGCNNet import *
from CiteseerData import CiteseerData
from NodeClassification import init_seeds, plot_loss_with_acc
from sklearn.metrics import roc_auc_score
from PPIData import PPIDataFromJson
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import Data

# 本文件利用pyg进行数据集分割，并得到最终的链路预测实验结果，与LinkPrediction中的结果做对比

# 产生tensor_adjacency
def generate_tensor_adjacency_for_link(edge_index, drop_edge=1.1):
    if drop_edge >= 1.0:
        adj = get_adjacent(edge_of_pg=edge_index, num_graph_node=num_nodes)
    else:
        adj = random_adjacent_sampler(edge_of_pg=edge_index, num_graph_node=num_nodes, drop_edge=drop_edge)

    normalize_adj = normalization(adj)

    # 准备将原来的coo_matrix转化到tensor形式
    index_of_coo_matrix = torch.from_numpy(np.asarray([normalize_adj.row,
                                                       normalize_adj.col]).astype('int64')).long()

    values_of_index_in_matrix = torch.from_numpy(normalize_adj.data.astype(np.float32))

    # 根据三元组构造稀疏矩阵张量,张量大小为是 (2708,2708)
    tensor_adjacency = torch.sparse.FloatTensor(
        index_of_coo_matrix, values_of_index_in_matrix,
        torch.Size([num_nodes, num_nodes]))
    return tensor_adjacency


if __name__ == "__main__":

    def get_link_labels_ppi(pos_edge_index, neg_edge_index):
        E = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(E, dtype=torch.float, device=device)
        link_labels[:pos_edge_index.size(1)] = 1.
        return link_labels

    def train():
        loss_list = []
        val_acc_history = []
        model.train()

        for epoch in range(epoch_num):
            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
                num_neg_samples=data.train_pos_edge_index.size(1),
                force_undirected=True,
            )
            neg_edge_index = neg_edge_index.to(device)
            out_feature = model.encode(tensor_x, train_pos_edge)

            logits = model.decode(out_feature, train_pos_edge, neg_edge_index)
            labels = get_link_labels_ppi(train_pos_edge, neg_edge_index)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播计算参数的梯度
            optimizer.step()  # 使用优化方法进行梯度更新

            # train_acc = test("train")  # 计算当前模型训练集上的准确率  调用test函数
            val_acc = test("val")  # 计算当前模型在验证集上的准确率

            # 记录训练过程中损失值和准确率的变化，用于画图
            loss_list.append(loss.item())
            val_acc_history.append(val_acc.item())
            print("Epoch {:03d}: Loss {:.4f}, ValAcc {:.4f}".format(
                epoch, loss.item(), val_acc.item()))

        return loss_list, val_acc_history

    def test(input="train"):
        model.eval()  # 表示将模型转变为evaluation（测试）模式，这样就可以排除BN和Dropout对测试的干扰

        pos_edge_index = data[f'{input}_pos_edge_index'].to(device)
        neg_edge_index = data[f'{input}_neg_edge_index'].to(device)
        with torch.no_grad():  # 显著减少显存占用
            out_feature = model.encode(tensor_x, train_pos_edge)  # (N,16)->(N,7) N节点数
            logits = model.decode(out_feature, pos_edge_index, neg_edge_index)
            labels = get_link_labels_ppi(pos_edge_index, neg_edge_index)
            accuarcy = roc_auc_score(labels.cpu(), logits.cpu())
        return accuarcy

    init_seeds()
    # 设置跑的平台是CPU还是GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 超参数设置
    learning_rate = 0.006
    epoch_num = 1000
    weight_decay = 5e-4
    hidden_layer_dim = 128
    out_feature_dim = 64
    layer_num = 2
    drop_edge = 1.1
    use_pair_norm = True

    # 读取数据
    dataset_name = "ppi"
    path_of_ppi = "../ppi/ppi"

    dataset = PPIDataFromJson(path_of_ppi)
    tensor_x = torch.tensor(dataset.feature_of_pg, dtype=torch.float)
    edge_index = torch.tensor(dataset.edge_of_pg, dtype=torch.long)
    data = Data(x=tensor_x, edge_index=edge_index.t().contiguous())

    data = train_test_split_edges(data)

    num_nodes = dataset.num_nodes
    edge_index = dataset.edge_of_pg
    train_mask, val_mask, test_mask = dataset.data_partition_node()
    num_of_class = dataset.num_of_class
    feature_dim = dataset.feature_dim


    # model = MyLinkPredictionGCN(hidden_layer_dim=hidden_layer_dim, num_of_hidden_layer=layer_num,
    #                            out_feature_dim=out_feature_dim, use_pair_norm=use_pair_norm,
    #                            input_feature_dim=feature_dim).to(device)

    model = LinkPredictionGCNFromPYG(hidden_layer_dim=hidden_layer_dim, num_of_hidden_layer=layer_num,
                                    out_feature_dim=out_feature_dim, use_pair_norm=use_pair_norm,
                                    input_feature_dim=feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    tensor_x = tensor_x.to(device)
    train_pos_edge = data.train_pos_edge_index.to(device)

    loss, val_acc = train()
    test_acc = test("test")
    print("Test accuarcy: ", test_acc.item())
    plot_loss_with_acc(loss, val_acc)
