from CoraData import *
from MyGCNNet import *
from CiteseerData import CiteseerData
from NodeClassification import init_seeds, plot_loss_with_acc
from sklearn.metrics import roc_auc_score
from PPIData import PPIDataFromJson

# 本文件不利用pyg，均用手写的函数进行链路预测结果的实现

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

    def train():
        loss_list = []
        val_acc_history = []
        model.train()

        train_pos_edge_index_cuda = torch.tensor(train_pos_edge_index, device=device, dtype=torch.long)

        for epoch in range(epoch_num):
            tensor_adjacency = generate_tensor_adjacency_for_link(train_pos_edge_index, drop_edge=drop_edge).to(device)
            negative_edge_index = negative_edge_sampling(train_neg_edge_index, train_pos_edge_index)
            negative_edge_index_cuda = torch.tensor(negative_edge_index, device=device, dtype=torch.long)
            out_feature = model.encode(tensor_x, tensor_adjacency)

            logits = model.decode(out_feature, train_pos_edge_index_cuda, negative_edge_index_cuda)
            labels = get_link_labels(train_pos_edge_index_cuda, negative_edge_index_cuda, device=device)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播计算参数的梯度
            optimizer.step()  # 使用优化方法进行梯度更新

            train_acc = test(train_pos_edge_index, train_neg_edge_index)  # 计算当前模型训练集上的准确率  调用test函数
            val_acc = test(validate_pos_edge_index, validate_neg_edge_index)  # 计算当前模型在验证集上的准确率

            # 记录训练过程中损失值和准确率的变化，用于画图
            loss_list.append(loss.item())
            val_acc_history.append(val_acc.item())
            print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
                epoch, loss.item(), train_acc.item(), val_acc.item()))

        return loss_list, val_acc_history

    def test(pos_edge_index, neg_edge_index):
        model.eval()  # 表示将模型转变为evaluation（测试）模式，这样就可以排除BN和Dropout对测试的干扰

        with torch.no_grad():  # 显著减少显存占用
            tensor_adjacency = generate_tensor_adjacency_for_link(train_pos_edge_index).to(device)
            pos_edge_index_cuda = torch.tensor(pos_edge_index, device=device, dtype=torch.long)
            neg_edge_index_cuda = torch.tensor(neg_edge_index, device=device, dtype=torch.long)

            out_feature = model.encode(tensor_x, tensor_adjacency)  # (N,16)->(N,7) N节点数
            logits = model.decode(out_feature, pos_edge_index_cuda, neg_edge_index_cuda)
            labels = get_link_labels(pos_edge_index_cuda, neg_edge_index_cuda, device=device)

            accuarcy = roc_auc_score(labels.cpu(), logits.cpu())
        return accuarcy

    init_seeds()
    # 设置跑的平台是CPU还是GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 超参数设置
    learning_rate = 0.01
    epoch_num = 100
    weight_decay = 5e-4
    hidden_layer_dim = 128
    out_feature_dim = 64
    layer_num = 2
    drop_edge = 1.1
    use_pair_norm = False

    # 读取数据
    dataset_name = "citeseer"
    path_of_cora = "../cora/cora"
    path_of_citeseer = "../citeseer/citeseer"
    path_of_ppi = "../ppi/ppi"

    dataset = None
    if dataset_name == "cora":
        dataset = CoraData(path_of_cora)
    elif dataset_name == "citeseer":
        dataset = CiteseerData(path_of_citeseer)
    elif dataset_name == "ppi":
        dataset = PPIDataFromJson(path_of_ppi)

    num_nodes = dataset.num_nodes
    edge_index = dataset.edge_of_pg
    train_mask, val_mask, test_mask = dataset.data_partition_node()
    num_of_class = dataset.num_of_class
    feature_dim = dataset.feature_dim

    model = MyLinkPredictionGCN(hidden_layer_dim=hidden_layer_dim, num_of_hidden_layer=layer_num,
                                out_feature_dim=out_feature_dim, use_pair_norm=use_pair_norm,
                                input_feature_dim=feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 边集分割
    train_pos_edge_index, validate_pos_edge_index, test_pos_edge_index, train_neg_edge_index, \
    validate_neg_edge_index, test_neg_edge_index = data_partition_edge(dataset.edge_of_pg, num_nodes)

    tensor_x = torch.tensor(dataset.feature_of_pg, device=device, dtype=torch.float)

    loss, val_acc = train()
    test_acc = test(test_pos_edge_index, test_neg_edge_index)
    print("Test accuarcy: ", test_acc.item())
    plot_loss_with_acc(loss, val_acc)
