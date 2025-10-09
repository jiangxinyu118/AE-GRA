from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import global_mean_pool

import utils
from copy import deepcopy
from sklearn.metrics import f1_score
# from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset
# from opacus.accountants.rdp import RDPAccountant
import numpy as np

class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # self.weight.data.fill_(1)
        # if self.bias is not None:
        #     self.bias.data.fill_(1)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

#nfeat=features.shape[1], nhid=16, device=device
class embedding_GCN(nn.Module):
    def __init__(self, nfeat, nhid, with_bias=True, device=None):
        super(embedding_GCN, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.with_bias = with_bias

    # def forward(self, x, adj):
    #     x = self.gc1(x, adj)
    #     # x = F.relu(self.gc1(x, adj))
    #     return x

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        # x = F.relu(self.gc1(x, adj))
        laplace = torch.distributions.Laplace(loc=0.0, scale=10)
        noise = laplace.sample(x.shape).to(x.device)

        x = x + noise
        return x

    def initialize(self):
        self.gc1.reset_parameters()

class GCN(nn.Module):
    """ 2 Layer Graph Convolutional Network.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN. When `with_relu` is True, `weight_decay` will be set to 0.
    with_relu : bool
        whether to use relu activation function. If False, GCN will be linearized.
    with_bias: bool
        whether to include bias term in GCN weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train GCN.

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    >>> gcn = gcn.to('cpu')
    >>> gcn.fit(features, adj, labels, idx_train) # train without earlystopping
    >>> gcn.fit(features, adj, labels, idx_train, idx_val, patience=30) # train with earlystopping

    """
    # nfeat = features.shape[1], nclass = labels.max().item() + 1, nhid = 16,
    # dropout = 0.5, weight_decay = 5e-4, device = device)
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, device=None):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None



    def forward(self, x, adj):
        if self.with_relu:  # true
            x = F.relu(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)# x = F.softmax(x, dim=1)
        return x     #常用

    # def forward(self, x, adj):
    #     if self.with_relu:  # true
    #         x = F.relu(self.gc1(x, adj))
    #     else:
    #         x = self.gc1(x, adj)
    #
    #     # === 在隐藏层节点嵌入上添加 Laplace(β=0.1) 扰动 ===
    #     laplace = torch.distributions.Laplace(loc=0.0, scale=10)
    #     noise = laplace.sample(x.shape).to(x.device)
    #     x = x + noise
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = self.gc2(x, adj)  # 最后一层输出
    #     return x   #节点嵌入扰动机制


    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

#features, adj, labels, idx_train, idx_val
    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=True, normalize=True, patience=500, **kwargs):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        normalize : bool
            whether to normalize the input adjacency matrix.
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """

        self.device = self.gc1.weight.device
        if initialize:#True
            self.initialize()#重新初始化两层图卷积的参数

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:#True
            if utils.is_sparse_tensor(adj):#判断是否是稀疏矩阵
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)#对稀疏邻接矩阵进行归一化处理
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels

        if idx_val is None:#idx_val有值
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:#patience=500  train_iters=200
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:#训练
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)
                # self._train_with_val_df(labels, idx_train, idx_val, train_iters, verbose)

                # self._train_with_val1(labels, idx_train, idx_val, train_iters, verbose)#原始攻击

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
#
            saved_var = dict()
            for tensor_name, tensor in self.named_parameters():
                saved_var[tensor_name] = torch.zeros_like(tensor)
#
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            for tensor_name, tensor in self.named_parameters():
                new_grad = tensor.grad
                saved_var[tensor_name].add_(new_grad)

            for tensor_name, tensor in self.named_parameters():
                if self.device.type == 'cuda':
                    noise = torch.cuda.FloatTensor(tensor.grad.shape).normal_(0, 0.05)
                else:
                    noise = torch.FloatTensor(tensor.grad.shape).normal_(0, 0.05)
                saved_var[tensor_name].add_(noise)
                tensor.grad = saved_var[tensor_name]

            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

#opacue实现DP-报错
# labels, idx_train, idx_val, train_iters=200, verbose=true
#     def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose, noise_multiplier=1.0, max_grad_norm=1.0,
#                         target_epsilon=10.0, target_delta=1e-5):
#         if verbose:
#             print('=== 使用差分隐私训练 GCN 模型，控制 epsilon 和 delta ===')
#
#         # 初始化优化器
#         optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#
#         # 初始化 PrivacyEngine，设置噪声乘数和梯度裁剪
#         privacy_engine = PrivacyEngine()
#
#         # 使用 make_private 替代 attach
#         self, optimizer = privacy_engine.make_private(
#             module=self,
#             optimizer=optimizer,
#             noise_multiplier=noise_multiplier,
#             max_grad_norm=max_grad_norm
#         )
#
#         best_acc_val = 0
#         criterion = nn.CrossEntropyLoss()
#         weights = None
#
#         # 训练过程
#         for i in range(train_iters):  # 200 次迭代
#             self.train()
#             optimizer.zero_grad()
#
#             # 正向传播
#             output = self.forward(self.features, self.adj_norm)
#             loss_train = criterion(output[idx_train], labels[idx_train])
#
#             # 反向传播（应用差分隐私机制）
#             loss_train.backward()
#
#             # 梯度剪切并添加噪声
#             optimizer.step()
#             privacy_engine.step()  # 应用差分隐私步骤
#
#             self.eval()
#             output = self.forward(self.features, self.adj_norm)
#             acc_val = utils.accuracy(output[idx_val], labels[idx_val])
#
#             # 每10个迭代打印一次进度
#             if verbose and i % 10 == 0:
#                 print(f'第 {i} 轮，训练损失: {loss_train.item()}, 验证准确率: {acc_val}')
#
#             # 输出当前的 epsilon 和 delta
#             current_epsilon, current_delta = privacy_engine.get_privacy_spent(delta=target_delta)
#             print(f"训练 {i} 轮后，当前 epsilon: {current_epsilon:.2f}, delta: {current_delta:.2e}")
#
#             # 控制 epsilon 和 delta，不超过目标值
#             if current_epsilon > target_epsilon:
#                 print(f"训练超出了目标 epsilon {target_epsilon}，停止训练。")
#                 break
#
#             if acc_val > best_acc_val:
#                 best_acc_val = acc_val
#                 print(f"在第 {i} 轮保存了最好的验证准确率: {acc_val}")
#                 self.output = output
#                 weights = deepcopy(self.state_dict())
#
#         if weights is not None:
#             torch.save(weights, '/home2/jxy/GraphMI-main/DP_defense/dp_cora_targetmodel.pth')
#             print("成功保存模型权重。")


#手动实现DP-SGD，但是效果不理想
    # def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
    #     r=2000
    #     # r = 20000
    #     epsilon = 10*r
    #     delta = 1e-5
    #     max_grad_norm = 1.0
    #     noise_scale = 0.1
    #
    #     if verbose:
    #         print('=== Training GCN with DP-SGD ===')
    #
    #     optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #     criterion = nn.CrossEntropyLoss()
    #     best_acc_val = 0
    #     weights = None
    #
    #     # 存储 RDP 计算参数
    #     alphas = list(range(2, 100))  # RDP 计算的 alpha 范围
    #     # q = len(idx_train) / self.features.shape[0]  # 采样率
    #     q=0.1
    #     # **初始化 RDPAccountant**
    #     accountant = RDPAccountant()
    #
    #     for i in range(train_iters):
    #         self.train()
    #         optimizer.zero_grad()
    #
    #         # 前向传播
    #         output = self.forward(self.features, self.adj_norm)
    #         loss_train = criterion(output[idx_train], labels[idx_train])
    #
    #         # 计算梯度
    #         loss_train.backward()
    #
    #         # **梯度裁剪**
    #         torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
    #
    #         # **添加高斯噪声**
    #         for param in self.parameters():
    #             if param.grad is not None:
    #                 noise = torch.normal(0, noise_scale, size=param.grad.shape, device=param.grad.device)
    #                 param.grad += noise
    #
    #         # **优化器更新**
    #         optimizer.step()
    #
    #         # **更新隐私预算**更新隐私损失的累积量
    #         accountant.step(noise_multiplier=noise_scale, sample_rate=q)
    #
    #         # 计算当前 ε
    #         epsilon_t, _ = accountant.get_privacy_spent(delta=delta,alphas=alphas)
    #
    #         self.eval()
    #         output = self.forward(self.features, self.adj_norm)
    #         acc_val = utils.accuracy(output[idx_val], labels[idx_val])
    #
    #         if verbose and i % 10 == 0:
    #             print(f'Epoch {i}, Training Loss: {loss_train.item()}, Validation Acc: {acc_val}, ε = {epsilon_t:.2f}')
    #
    #         if acc_val > best_acc_val:
    #             best_acc_val = acc_val
    #             print(f"保存模型字典时的验证 acc: {acc_val}")
    #             self.output = output
    #             weights = deepcopy(self.state_dict())
    #
    #         # **检查是否超出隐私预算**
    #         if epsilon_t > epsilon:
    #             print(f"训练提前终止，已达到隐私预算 ε = {epsilon_t:.2f}")
    #             break
    #
    #     # **保存模型**
    #     if weights is not None:
    #         torch.save(weights, '/hy-tmp/GraphMI-main/DP_defense/AIDS/AIDS_10_targetmodel.pth')
    #         print("成功保存字典")


    # ----不用DP的训练 - ---
    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):

        if verbose:  # true
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0
        # 定义交叉熵损失函数
        criterion = nn.CrossEntropyLoss()
        weights = None

        for i in range(train_iters):  # 200
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = criterion(output[idx_train], labels[idx_train])

            loss_train.backward()
            optimizer.step()
            self.eval()
            output = self.forward(self.features, self.adj_norm)
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}, gcn验证节点预测 val acc: {}'.format(i, loss_train.item(), acc_val))

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                print("保存模型字典时的验证acc是" + str(acc_val))
                self.output = output
                weights = deepcopy(self.state_dict())

        if weights is not None:
            torch.save(weights,
                       ' ')

    def _train_with_val_df(self, labels, idx_train, idx_val, train_iters, verbose):

        if verbose:  # true
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0
        # 定义交叉熵损失函数
        criterion = nn.CrossEntropyLoss()
        weights = None

        for i in range(train_iters):  # 200
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = criterion(output[idx_train], labels[idx_train])

            loss_train.backward()
            optimizer.step()
            self.eval()
            output = self.forward(self.features, self.adj_norm)
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}, gcn验证节点预测 val acc: {}'.format(i, loss_train.item(), acc_val))

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if weights is not None:
            torch.save(weights,
                       ' ')


    def _train_with_val1(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}, val acc: {}'.format(i, loss_train.item(), acc_val))

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            # def eval_class(output, labels):
            #     preds = output.max(1)[1].type_as(labels)
            #     return f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='micro') + \
            #         f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

            # perf_sum = eval_class(output[idx_val], labels[idx_val])
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self, idx_test):
        """Evaluate GCN performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test


    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized data

        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)

