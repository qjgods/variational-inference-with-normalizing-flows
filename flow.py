import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from utils import safe_log


class NormalizingFlow(nn.Module):

    def __init__(self, dim, flow_length):
        super().__init__()

        self.transforms = nn.Sequential(*(
            PlanarFlow(dim) for _ in range(flow_length) #产生flow_length个PlanarFlow
        ))

        self.log_jacobians = nn.Sequential(*(
            PlanarFlowLogDetJacobian(t) for t in self.transforms #对每个PlanarFlow产生PlanarFlowLogDetJacobian
        ))

    def forward(self, z):

        log_jacobians = [] # 这个跟self.log_jacobians是不同的不要混淆

        for transform, log_jacobian in zip(self.transforms, self.log_jacobians): #遍历每个flow的函数，其变换函数为transform,其导数函数为log_jacobian
            log_jacobians.append(log_jacobian(z)) # 计算z的导数
            z = transform(z)

        zk = z # 记录最后flow的z

        return zk, log_jacobians # 输出z 以及所有层的jacobian


class PlanarFlow(nn.Module): # f(z)=z+scale*tang(z*weight+bias)

    def __init__(self, dim):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.tanh = nn.Tanh()

        self.reset_parameters() # 初始化参数，所有从均匀分布中随机抽取

    def reset_parameters(self):

        self.weight.data.uniform_(-0.01, 0.01)
        self.scale.data.uniform_(-0.01, 0.01)
        self.bias.data.uniform_(-0.01, 0.01)

    def forward(self, z):

        activation = F.linear(z, self.weight, self.bias) # activation= z*weight+bias
        return z + self.scale * self.tanh(activation) # =z+scale*tang(z*weight+bias)


class PlanarFlowLogDetJacobian(nn.Module):
    """A helper class to compute the determinant of the gradient of
    the planar flow transformation."""

    def __init__(self, affine): # 输入是一个PlanarFlow
        super().__init__()

        self.weight = affine.weight # 读取PlanarFlow中的参数
        self.bias = affine.bias
        self.scale = affine.scale
        self.tanh = affine.tanh

    def forward(self, z):

        activation = F.linear(z, self.weight, self.bias) # 应该是这里z只有一维所以才可以这么写？，否则zw应该是一个外积才对。
        psi = (1 - self.tanh(activation) ** 2) * self.weight # 1+h′(zw^T+b)w 
        det_grad = 1 + torch.mm(psi, self.scale.t()) # 1+h′(zw^T+b)wu^T 
        return safe_log(det_grad.abs())
