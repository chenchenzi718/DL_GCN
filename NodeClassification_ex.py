import matplotlib.pyplot as plt
import torch

from CoraData import *
from MyGCNNet import *
from CiteseerData import CiteseerData
from PPIData import PPIDataFromJson


# 本文件利用pyg进行节点分类实验，用以和NodeClassification中结果做对比


# 产生tensor_adjacency
def generate_tensor_adjacency_for_classify(edge_index, drop_edge=1.1):
    if drop_edge >= 1.0:
        adj = get_adjacent(edge_of_pg=edge_index, num_graph_node=num_nodes, symmetric_of_edge=True)
    else:
        adj = random_adjacent_sampler(edge_of_pg=edge_index, num_graph_node=num_nodes, symmetric_of_edge=True,
                                      drop_edge=drop_edge)

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


# 设置随机数种子
def init_seeds(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 设置模型保存路径
def set_model_save_path(learning_rate, epoch_num, hidden_size, layer_num, dataset_name='cora'):

    path = '../cora_model_result/' + dataset_name + '_rate' + str(learning_rate) + '_epoch' + str(epoch_num) \
            + '_hidden' + str(hidden_size) + '_layer' + str(layer_num) + '.pth'
    return path


# 设置图片保存路径
def set_pic_save_path(learning_rate, epoch_num, hidden_size, layer_num, dataset_name='cora'):

    path = '../cora_model_result/' + dataset_name + '_rate' + str(learning_rate) + '_epoch' + str(epoch_num) \
           + '_hidden' + str(hidden_size) + '_layer' + str(layer_num) + '.png'
    return path


# 画图
def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    # 坐标系ax1画曲线1
    ax1 = fig.add_subplot(111)  # 指的是将plot界面分成1行1列，此子图占据从左到右从上到下的1位置
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)  # c为颜色
    plt.ylabel('Loss')

    # 坐标系ax2画曲线2
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)  # 其本质就是添加坐标系，设置共享ax1的x轴，ax2背景透明
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()  # 开启右边的y坐标

    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()


if __name__ == '__main__':
    # 训练模型
    def train():
        loss_list = []
        val_acc_history = []
        model.train()

        train_y = tensor_y[train_mask]  # shape=（140，）不是（2708，）了
        # 共进行200次训练
        for epoch in range(epoch_num):
            tensor_adjacency = torch.tensor(edge_index, device=device, dtype=torch.long).t()
            logits = model(tensor_x, tensor_adjacency)  # 前向传播，认为因为声明了 model.train()，不用forward了
            train_mask_logits = logits[train_mask]  # 只选择训练节点进行监督 (140,)

            loss = criterion(train_mask_logits, train_y)  # 计算损失值

            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播计算参数的梯度
            optimizer.step()  # 使用优化方法进行梯度更新

            train_acc = test(train_mask)  # 计算当前模型训练集上的准确率  调用test函数
            val_acc = test(val_mask)  # 计算当前模型在验证集上的准确率

            # 记录训练过程中损失值和准确率的变化，用于画图
            loss_list.append(loss.item())
            val_acc_history.append(val_acc.item())
            print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
                epoch, loss.item(), train_acc.item(), val_acc.item()))

        return loss_list, val_acc_history

    # 测试模型
    def test(mask):
        model.eval()  # 表示将模型转变为evaluation（测试）模式，这样就可以排除BN和Dropout对测试的干扰

        with torch.no_grad():  # 显著减少显存占用
            tensor_adjacency = torch.tensor(edge_index, device=device, dtype=torch.long).t()
            logits = model(tensor_x, tensor_adjacency)  # (N,16)->(N,7) N节点数
            test_mask_logits = logits[mask]  # 矩阵形状和mask一样

            accuracy = micro_f1_score(test_mask_logits.cpu(), tensor_y[mask].cpu())

        return accuracy

    # 计算多分类问题的准确率
    def micro_f1_score(y_pred, y_true):
        # Convert y_pred and y_true into binary tensors
        y_pred = (y_pred > 0.5).float()
        y_true = (y_true > 0.5).float()

        tp = (y_pred * y_true).sum(dim=0)
        fp = ((1 - y_true) * y_pred).sum(dim=0)
        fn = (y_true * (1 - y_pred)).sum(dim=0)

        precision = tp / (tp + fp + 1e-16)
        recall = tp / (tp + fn + 1e-16)

        f1 = 2 * precision * recall / (precision + recall + 1e-16)
        f1 = f1.mean()

        return f1


    init_seeds()
    # 设置跑的平台是CPU还是GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 超参数设置
    learning_rate = 0.01
    epoch_num = 200
    weight_decay = 5e-4
    hidden_layer_dim = 128
    layer_num = 2
    drop_edge = 1.1
    use_pair_norm = True

    # 读取数据
    dataset_name = "ppi"
    path_of_ppi = "../ppi/ppi"

    dataset = PPIDataFromJson(path_of_ppi)

    num_nodes = dataset.num_nodes
    edge_index = dataset.edge_of_pg
    train_mask, val_mask, test_mask = dataset.data_partition_node()
    num_of_class = dataset.num_of_class
    feature_dim = dataset.feature_dim

    tensor_x = torch.tensor(dataset.feature_of_pg, device=device, dtype=torch.float)
    tensor_y = torch.tensor(dataset.label_of_pg, device=device, dtype=torch.float)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # 模型定义
    # model = MyClassificationGCN(hidden_layer_dim=hidden_layer_dim,
    #                            num_of_hidden_layer=layer_num, use_pair_norm=use_pair_norm,
    #                            num_of_class=num_of_class, input_feature_dim=feature_dim).to(device)
    model = ClassificationGCNFromPYG(hidden_layer_dim=hidden_layer_dim,
                                    num_of_hidden_layer=layer_num, use_pair_norm=use_pair_norm,
                                    num_of_class=num_of_class, input_feature_dim=feature_dim).to(device)

    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss, val_acc = train()
    test_acc = test(test_mask)
    print("Test accuarcy: ", test_acc.item())
    plot_loss_with_acc(loss, val_acc)
