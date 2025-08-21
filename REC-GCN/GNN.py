import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GNNLayer(Module):    #特征数量和DNN编码器数量一直，保证学得的信息能线性运算
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features   #FloatTensor类型转换, 将list ,numpy转化为tensor
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)#一个服从均匀分布的Glorot初始化器

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight) #矩阵a和b矩阵相乘
        output = torch.spmm(adj, support)    #矩阵乘法
        if active:
            output = F.relu(output)
        return output

