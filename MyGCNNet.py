import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# 本文件中记录了我自己搭建的卷积核以及GCN网络，以及利用PYG搭建的GCN网络以验证我的卷积网络正确性


# 自己搭建的GCN卷积模型
class GraphConvolution(nn.Module):
    def __init__(self, in_features_dim, out_features_dim, use_bias=True):
        # 此部分要计算的是卷积部分D^-1/2 A D^-1/2 * X * W , X为feature，W为参数
        super(GraphConvolution, self).__init__()

        self.in_features_dim = in_features_dim
        self.out_features_dim = out_features_dim
        self.use_bias = use_bias

        # 定义GCN层的 W 权重形状
        self.weight = nn.Parameter(torch.Tensor(in_features_dim, out_features_dim))

        # 定义GCN层的 b 权重矩阵
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 声明初始化 nn.Module 类里面的W,b参数
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        # init.xavier_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adj, in_feature):
        # 输入的为稀疏矩阵adj
        support = torch.mm(in_feature, self.weight)  # X*W
        output = torch.sparse.mm(adj, support)  # A*X*W
        if self.use_bias:
            output += self.bias  # 添加偏置项
        return output


class MyMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layer_num, dropout=0.01):
        super(MyMLP, self).__init__()
        self.layers = nn.ModuleList()
        # 输入层
        self.layers.append(nn.Linear(input_size, hidden_size))
        # 隐藏层
        for i in range(hidden_layer_num - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        # 输出层
        self.layers.append(nn.Linear(hidden_size, output_size))
        self.dropout = nn.Dropout(p=dropout)
        self.num_of_hidden_layer = hidden_layer_num

    def forward(self, x):
        output = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                output = self.dropout(output)
            output = layer(output)
            if i != (self.num_of_hidden_layer-1):
                output = F.relu(output)
        return output


# 搭建利用我搭建的卷积网络做节点分类的GCN网络
class MyClassificationGCN(nn.Module):
    def __init__(self, hidden_layer_dim, num_of_hidden_layer,
                 num_of_class=7, input_feature_dim=1433, dropout=0.01, use_pair_norm=True):
        super(MyClassificationGCN, self).__init__()
        self.layers = nn.ModuleList()
        # 输入层
        self.layers.append(GraphConvolution(input_feature_dim, hidden_layer_dim))
        # 隐藏层
        for i in range(num_of_hidden_layer-2):
            self.layers.append(GraphConvolution(hidden_layer_dim, hidden_layer_dim))
        # 输出层
        self.layers.append(GraphConvolution(hidden_layer_dim, num_of_class))
        self.dropout = nn.Dropout(p=dropout)
        self.use_pair_norm = use_pair_norm
        self.num_of_hidden_layer = num_of_hidden_layer
        self.mlp = MyMLP(input_size=hidden_layer_dim, hidden_size=hidden_layer_dim,
                         output_size=num_of_class, hidden_layer_num=5)

    @staticmethod
    def PairNorm(x_feature):
        mode = 'PN-SI'
        scale = 1
        col_mean = x_feature.mean(dim=0)
        if mode == 'PN':
            x_feature = x_feature - col_mean
            row_norm_mean = (1e-6 + x_feature.pow(2).sum(dim=1).mean()).sqrt()
            x_feature = scale * x_feature / row_norm_mean

        if mode == 'PN-SI':
            x_feature = x_feature - col_mean
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual

        if mode == 'PN-SCS':
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual - col_mean

        return x_feature

    def forward(self, x_feature, adj):
        output = x_feature
        for i, layer in enumerate(self.layers):
            if i != 0:
                output = self.dropout(output)

            output = layer(adj, output)
            if i != (self.num_of_hidden_layer-1):
                output = F.relu(output)
            if self.use_pair_norm:
                output = self.PairNorm(output)

        # output = self.mlp(output)
        output = torch.sigmoid(output)
        return output


# 搭建链路链接的网络
class MyLinkPredictionGCN(nn.Module):
    def __init__(self, hidden_layer_dim, num_of_hidden_layer, out_feature_dim,
                 input_feature_dim=1433, dropout=0.1, use_pair_norm=True):
        super(MyLinkPredictionGCN, self).__init__()
        self.layers = nn.ModuleList()
        # 输入层
        self.layers.append(GraphConvolution(input_feature_dim, hidden_layer_dim))
        # 隐藏层
        for i in range(num_of_hidden_layer - 1):
            self.layers.append(GraphConvolution(hidden_layer_dim, hidden_layer_dim))
        # 输出层
        self.layers.append(GraphConvolution(hidden_layer_dim, out_feature_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.use_pair_norm = use_pair_norm

    @staticmethod
    def PairNorm(x_feature):
        mode = 'PN-SI'
        scale = 1
        col_mean = x_feature.mean(dim=0)
        if mode == 'PN':
            x_feature = x_feature - col_mean
            row_norm_mean = (1e-6 + x_feature.pow(2).sum(dim=1).mean()).sqrt()
            x_feature = scale * x_feature / row_norm_mean

        if mode == 'PN-SI':
            x_feature = x_feature - col_mean
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual

        if mode == 'PN-SCS':
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual - col_mean

        return x_feature

    # encode部分将num_node*num_feature的in_feature转变成了num_node*out_feature_dim的矩阵
    def encode(self, in_feature, adj):
        output = in_feature
        for i, layer in enumerate(self.layers):
            if i != 0:
                output = self.dropout(output)
            output = layer(adj, output)
            output = F.relu(output)
            if self.use_pair_norm:
                output = self.PairNorm(output)
        return output

    # 这里的out_feature是encode部分输出的量，edge_index为一个2E*2的张量，下面需要将一个edge上的两个点处的out_feature做点积
    # pos_edge_index为E*2，neg_edge_index为E*2, logits为一个2E*1的量， 表示那些边的属性
    @staticmethod
    def decode(out_feature, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=0)
        logits = (out_feature[edge_index[:, 0]] * out_feature[edge_index[:, 1]]).sum(dim=-1)
        logits = torch.sigmoid(logits)
        return logits


# 使用了pyg中GCNConv的类
class ClassificationGCNFromPYG(nn.Module):
    def __init__(self, hidden_layer_dim, num_of_hidden_layer,
                 num_of_class=7, input_feature_dim=1433, dropout=0.01, use_pair_norm=True):
        super(ClassificationGCNFromPYG, self).__init__()
        self.layers = nn.ModuleList()
        # 输入层
        self.layers.append(GCNConv(input_feature_dim, hidden_layer_dim))
        # 隐藏层
        for i in range(num_of_hidden_layer-2):
            self.layers.append(GCNConv(hidden_layer_dim, hidden_layer_dim))
        # 输出层
        self.layers.append(GCNConv(hidden_layer_dim, num_of_class))
        self.dropout = nn.Dropout(p=dropout)
        self.use_pair_norm = use_pair_norm
        self.num_of_hidden_layer = num_of_hidden_layer

    @staticmethod
    def PairNorm(x_feature):
        mode = 'PN-SI'
        scale = 1
        col_mean = x_feature.mean(dim=0)
        if mode == 'PN':
            x_feature = x_feature - col_mean
            row_norm_mean = (1e-6 + x_feature.pow(2).sum(dim=1).mean()).sqrt()
            x_feature = scale * x_feature / row_norm_mean

        if mode == 'PN-SI':
            x_feature = x_feature - col_mean
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual

        if mode == 'PN-SCS':
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual - col_mean

        return x_feature

    def forward(self, x_feature, adj):
        output = x_feature
        for i, layer in enumerate(self.layers):
            if i != 0:
                output = self.dropout(output)

            output = layer(output, adj)
            if i != (self.num_of_hidden_layer-1):
                output = F.relu(output)
            if self.use_pair_norm:
                output = self.PairNorm(output)

        output = torch.sigmoid(output)
        return output


# 利用pyg中的GCNConv搭建链路链接的网络
class LinkPredictionGCNFromPYG(nn.Module):
    def __init__(self, hidden_layer_dim, num_of_hidden_layer, out_feature_dim,
                 input_feature_dim=1433, dropout=0.1, use_pair_norm=True):
        super(LinkPredictionGCNFromPYG, self).__init__()
        self.layers = nn.ModuleList()
        # 输入层
        self.layers.append(GCNConv(input_feature_dim, hidden_layer_dim))
        # 隐藏层
        for i in range(num_of_hidden_layer - 2):
            self.layers.append(GCNConv(hidden_layer_dim, hidden_layer_dim))
        # 输出层
        self.layers.append(GCNConv(hidden_layer_dim, out_feature_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.use_pair_norm = use_pair_norm
        self.num_of_hidden_layer = num_of_hidden_layer

    @staticmethod
    def PairNorm(x_feature):
        mode = 'PN-SI'
        scale = 1
        col_mean = x_feature.mean(dim=0)
        if mode == 'PN':
            x_feature = x_feature - col_mean
            row_norm_mean = (1e-6 + x_feature.pow(2).sum(dim=1).mean()).sqrt()
            x_feature = scale * x_feature / row_norm_mean

        if mode == 'PN-SI':
            x_feature = x_feature - col_mean
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual

        if mode == 'PN-SCS':
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual - col_mean

        return x_feature

    # encode部分将num_node*num_feature的in_feature转变成了num_node*out_feature_dim的矩阵
    def encode(self, in_feature, adj):
        output = in_feature
        for i, layer in enumerate(self.layers):
            if i != 0:
                output = self.dropout(output)
            output = layer(output, adj)

            if i != (self.num_of_hidden_layer - 1):
                output = F.relu(output)
            if self.use_pair_norm:
                output = self.PairNorm(output)
        return output

    # 这里的out_feature是encode部分输出的量，edge_index为一个2E*2的张量，下面需要将一个edge上的两个点处的out_feature做点积
    # pos_edge_index为E*2，neg_edge_index为E*2, logits为一个2E*1的量， 表示那些边的属性
    @staticmethod
    def decode(out_feature, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (out_feature[edge_index[0]] * out_feature[edge_index[1]]).sum(dim=-1)
        logits = torch.sigmoid(logits)
        return logits
