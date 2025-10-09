import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm
import scipy.sparse as sp
import utils
from base_attack import BaseAttack
from utils import *



#model=victim_model, embedding=embedding, nnodes=adj.shape[0], loss_type='CE', device=device
class PGDAttack(BaseAttack):
#是BaseAttack类的子类
    def __init__(self, model=None, embedding=None, nnodes=None, loss_type='CE', feature_shape=None,
                 attack_structure=True, attack_features=False, device='cpu'):
        super(PGDAttack, self).__init__(model, nnodes, attack_structure, attack_features, device)
        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.loss_type = loss_type
        self.modified_adj = None
        self.modified_features = None
        self.edge_select = None
        self.complementary = None
        self.embedding = embedding
        if attack_structure:#True nnodes * (nnodes - 1) / 2 这个公式计算的是下三角区域的元素数目
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(torch.FloatTensor(int(nnodes * (nnodes - 1) / 2)))
            self.adj_changes.data.fill_(0)

        if attack_features:#False
            assert True, 'Topology Attack does not support attack feature'


    def sample_negative_edges(self,adj, num_neg_samples):
        N = adj.shape[0]
        with torch.no_grad():
            mask = (adj == 0).triu(1)  # 只考虑上三角避免重复
            neg_indices = mask.nonzero(as_tuple=False)
            rand_idx = torch.randperm(neg_indices.shape[0])[:num_neg_samples]
            sampled_neg_edges = neg_indices[rand_idx]
        return sampled_neg_edges  # Tensor of shape [num_samples, 2]

    def embeddingattackGCN(self, adj,ori_features, ori_adj, labels, idx_random, num_edges,
               epochs=200, sample=False, **kwargs):
        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        embedding=self.embedding
        victim_model = self.surrogate  # 父类BaseAttack的surrogate属性
        victim_model.eval()
        self.embedding.eval()
        ori_adj_norm = utils.normalize_adj_tensor1(adj)
        em_target=embedding(ori_features, ori_adj_norm)
        N = em_target.shape[0]
        torch.cuda.empty_cache()
        for t in tqdm(range(200)):
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            em_est = embedding(ori_features, adj_norm)
            em_est_norm = F.normalize(em_est, p=2, dim=1)
            em_target_norm = F.normalize(em_target, p=2, dim=1)
            loss_mse_norm = F.mse_loss(em_est_norm, em_target_norm)
            score_matrix = torch.matmul(em_target, em_target.T)  # [N, N]
            prob_matrix = torch.sigmoid(score_matrix)
            epsilon = 1e-8  # 防止 log(0)+lossother+ alpha * loss_kl
            loss_kl = - (modified_adj * torch.log(prob_matrix + epsilon)).sum()
            alpha = 0.7
            loss = loss_mse_norm+alpha*loss_kl
            print("loss")
            print(loss)
            adj_grad = -torch.autograd.grad(loss, self.adj_changes)[0]
            lr = 0.1
            self.adj_changes.data.add_(lr * adj_grad)
            self.projection(num_edges)
            with torch.no_grad():
                self.adj_changes.copy_(torch.clamp(self.adj_changes, min=0, max=1))

        #Encoder
        em = self.embedding(ori_features, adj_norm)  # x = F.relu(self.gc1(x, adj))
        #Decoder
        self.adj_changes.data = self.dot_product_decode(em)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()
        return em_est.detach()

    def embeddingattackGAT(self, edge_index, adj,ori_features, ori_adj, labels, idx_random, num_edges,
               epochs=200, sample=False, **kwargs):
        self.sparse_features = sp.issparse(ori_features)  # 检查是否为稀疏矩阵，赋值为True或False
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        embedding=self.embedding
        victim_model = self.surrogate  # 父类BaseAttack的surrogate属性
        victim_model.eval()
        self.embedding.eval()
        ori_adj_norm = utils.normalize_adj_tensor1(adj)
        em_target=embedding(ori_features, edge_index)
        N = em_target.shape[0]
        torch.cuda.empty_cache()
        norm_loss_list = []
        kl_loss_list = []
        loss_list = []
        edge_index_change = torch.empty((2, 0), dtype=torch.long)
        for t in tqdm(range(30)):  # 200
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)  # 归一化，值在0到1之间
            edge_index_change, edge_weight = dense_to_sparse(adj_norm)
            em_est = embedding(ori_features, edge_index_change)
            em_est_norm = F.normalize(em_est, p=2, dim=1)
            em_target_norm = F.normalize(em_target, p=2, dim=1)
            loss_mse_norm = F.mse_loss(em_est_norm, em_target_norm)
            score_matrix = torch.matmul(em_target, em_target.T)  # [N, N]
            prob_matrix = torch.sigmoid(score_matrix)
            epsilon = 1e-8  # 防止 log(0)+lossother+ alpha * loss_kl
            loss_kl = - (modified_adj * torch.log(prob_matrix + epsilon)).sum()
            alpha = 0.5
            loss = loss_mse_norm+alpha*loss_kl
            norm_loss_list.append(loss_mse_norm.item())
            kl_loss_list.append(loss_kl.item())
            loss_list.append(loss.item())
            print(loss)
            adj_grad = -torch.autograd.grad(loss, self.adj_changes)[0]
            lr = 0.1
            self.adj_changes.data.add_(lr * adj_grad)
            self.projection(num_edges)
            with torch.no_grad():
                self.adj_changes.copy_(torch.clamp(self.adj_changes, min=0, max=1))
        #Encoder
        em = self.embedding(ori_features, edge_index_change)  # x = F.relu(self.gc1(x, adj))
        #Decoder
        self.adj_changes.data = self.dot_product_decode(em)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()
        return em_est.detach()

    def embeddingattackGCN_GAT(self,device, em_target, edge_index, adj,ori_features, ori_adj, labels, idx_random, num_edges,
               epochs=200, sample=False, **kwargs):

        self.sparse_features = sp.issparse(ori_features)  # 检查是否为稀疏矩阵，赋值为True或False
        ori_adj = torch.FloatTensor(ori_adj).to(device)
        ori_features = ori_features.detach().to(device).float()
        embedding=self.embedding
        victim_model = self.surrogate  # 父类BaseAttack的surrogate属性
        victim_model.eval()
        self.embedding.eval()
        N = em_target.shape[0]
        torch.cuda.empty_cache()
        edge_index_change = torch.empty((2, 0), dtype=torch.long)
        for t in tqdm(range(30)):  # 200
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)  # 归一化，值在0到1之间
            edge_index_change, edge_weight = dense_to_sparse(adj_norm)
            em_est = embedding(ori_features, edge_index_change)
            em_est_norm = F.normalize(em_est, p=2, dim=1)
            em_target_norm = F.normalize(em_target, p=2, dim=1)
            loss_mse_norm = F.mse_loss(em_est_norm, em_target_norm)
            score_matrix = torch.matmul(em_target, em_target.T)  # [N, N]
            prob_matrix = torch.sigmoid(score_matrix)
            epsilon = 1e-8  # 防止 log(0)+lossother+ alpha * loss_kl
            loss_kl = - (modified_adj * torch.log(prob_matrix + epsilon)).sum()
            alpha = 0.5
            loss = loss_mse_norm+alpha*loss_kl
            print("loss")
            print(loss)
            adj_grad = -torch.autograd.grad(loss, self.adj_changes)[0]
            lr = 0.1
            self.adj_changes.data.add_(lr * adj_grad)
            self.projection(num_edges)
            with torch.no_grad():
                self.adj_changes.copy_(torch.clamp(self.adj_changes, min=0, max=1))

        #Encoder
        em = self.embedding(ori_features, edge_index_change)  # x = F.relu(self.gc1(x, adj))
        #Decoder
        self.adj_changes.data = self.dot_product_decode(em)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()
        return em_est.detach()



    def gan_build_symmetric_matrix(self, lower_tri_elements, node_features_dim):
        batch_size = lower_tri_elements.size(0)
        # 初始化对称矩阵
        adj_matrix = torch.zeros(batch_size, node_features_dim, node_features_dim, device=lower_tri_elements.device)

        # 获取下三角部分的索引
        tril_indices = torch.tril_indices(row=node_features_dim, col=node_features_dim, offset=0)

        # 填充下三角部分的元素
        adj_matrix[:, tril_indices[0], tril_indices[1]] = lower_tri_elements

        # 对称化矩阵
        adj_matrix = adj_matrix + adj_matrix.transpose(1, 2) - torch.diag_embed(torch.diagonal(adj_matrix, dim1=1, dim2=2))
        return adj_matrix


#攻击conv代理模型


    def create_edge_index_from_modified_adj(self,modified_adj, threshold):
        num_nodes = modified_adj.shape[0]
        edge_index = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if modified_adj[i][j] > threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()



    def matrixFactorizationattack(self,cosine_similarity_matrix,target_model,node_num,features, init_adj, labels, idx_attack, num_edges, epochs=200,
                                  device=None, **kwargs):
        self.sparse_features = sp.issparse(features)  # 检查是否为稀疏矩阵，赋值为True或False
        ori_adj = init_adj
        tensorfeatures = features.to(device)
        cosine_similarity_matrix = cosine_similarity_matrix.to(device)
        target_model.eval()
        self.embedding.eval()
        modified_adj = self.proxy_get_modified_adj(ori_adj, device)
        adj_norm = utils.normalize_adj_tensor(modified_adj)
        output = target_model(tensorfeatures, adj_norm)#logit值
        # probabilities = F.softmax(output, dim=1)
        logits =output[idx_attack]
        reconstructed_A = torch.matmul(logits, logits.T)
        sym_A = 0.5 * (reconstructed_A + reconstructed_A.T)
        tau = 0  # 你可以调整这个阈值
        thresholded_A = (sym_A >= tau).float()
        print("thresholded_A:")
        print(thresholded_A)
        return thresholded_A

    def kmeans(self):
        center = np.random.choice(len(self.adj_changes), 2, replace=False)
        center = self.adj_changes[center]
        label = torch.zeros_like(self.adj_changes)
        for i in range(20):
            tmp0 = (self.adj_changes-center[0])**2
            tmp1 = (self.adj_changes-center[1])**2
            label = torch.min(torch.cat((tmp0.unsqueeze(0), tmp1.unsqueeze(0)), 0), 0)[1]
            label = label.float()
            tmp = torch.dot((torch.ones_like(label) - label), self.adj_changes)/(torch.ones_like(label) - label).sum()
            if torch.abs(tmp - center[0]) < 1e-5:
                print("stop early! ", i)
                break

            center[0] = tmp
            center[1] = torch.dot(label, self.adj_changes) / label.sum()

        if center[0] > center[1]:
            label = torch.ones_like(label) - label
        print(center[0], center[1])
        return label

    def random_sample(self, ori_adj, ori_features, labels, idx_attack):
        K = 20
        best_loss = 1000
        victim_model = self.surrogate
        with torch.no_grad():
            ori_s = self.adj_changes.cpu().detach().numpy()
            s = ori_s / ori_s.sum()
            for _ in range(K):
                sampled = np.random.choice(len(s), 5000, replace=False, p=s)
                self.adj_changes.data.copy_(torch.zeros_like(torch.tensor(s)))
                for k in sampled:
                    self.adj_changes[k] = 1.0

                modified_adj = self.get_modified_adj(ori_adj)
                adj_norm = utils.normalize_adj_tensor(modified_adj)
                output = victim_model(ori_features, adj_norm)
                loss_smooth_feat = self.feature_smoothing(modified_adj, ori_features)
                loss = self._loss(output[idx_attack], labels[idx_attack]) + torch.norm(self.adj_changes,
                                                                                             p=2) * 0.001 + 5e-7 * loss_smooth_feat
                test_acc = utils.accuracy(output[idx_attack], labels[idx_attack])
                print("loss= {:.4f}".format(loss.item()), "test_accuracy= {:.4f}".format(test_acc.item()))
                if best_loss > loss:
                    best_loss = loss
                    best_s = sampled

            self.adj_changes.data.copy_(torch.zeros_like(torch.tensor(s)))
            #self.adj_changes.data.copy_(torch.tensor(ori_s))
            for k in best_s:
                self.adj_changes[k] = 1.0

    def _loss(self, output, labels):
        if self.loss_type == "CE":
            loss = F.nll_loss(output, labels)
        if self.loss_type == "CW":
            onehot = utils.tensor2onehot(labels)
            best_second_class = (output - 1000 * onehot).argmax(1)
            margin = output[np.arange(len(output)), labels] - \
                     output[np.arange(len(output)), best_second_class]
            k = 0
            loss = -torch.clamp(margin, min=k).mean()
            # loss = torch.clamp(margin.sum()+50, min=k)
        return loss


    def feature_smoothing(self, adj, X):
        rowsum = adj.sum(1)  # 计算邻接矩阵每行的和（度数）
        r_inv = rowsum.flatten()  # 将每行和展平成1维张量
        D = torch.diag(r_inv)  # 构建度矩阵 D
        L = D - adj  # 拉普拉斯矩阵 L = D - A

        r_inv = r_inv + 1e-3  # 为避免数值不稳定，给 r_inv 添加一个很小的数
        r_inv = r_inv.pow(-1 / 2).flatten()  # 对 r_inv 的每个元素取 -1/2 次幂，得到 D^(-1/2)
        r_inv[torch.isinf(r_inv)] = 0.  # 避免出现无穷大的情况，替换为 0
        r_mat_inv = torch.diag(r_inv)  # 构建对角矩阵 D^(-1/2)

        L = torch.matmul(torch.matmul(r_mat_inv, L), r_mat_inv)  # 计算归一化的拉普拉斯矩阵

        # 确保 X 形状是 (batch_size, num_nodes, num_features)
        if len(X.shape) == 2:
            X = X.unsqueeze(0)  # 添加批次维度

        # 计算 XLXT，X 形状应为 (batch_size, num_nodes, num_features)
        XLXT = torch.matmul(torch.matmul(X.permute(0, 2, 1), L), X)  # 结果形状为 (batch_size, num_features, num_features)

        # 如果 XLXT 的形状是 (batch_size, num_features, num_features)，则计算每个矩阵的 trace
        if XLXT.dim() == 3:  # batch_size, num_features, num_features
            loss_smooth_feat = torch.sum(torch.diagonal(XLXT, dim1=1, dim2=2))  # 对每个批次的对角线求和
        else:
            raise ValueError("Expected XLXT to be a 3D tensor but got shape: {}".format(XLXT.shape))

        return loss_smooth_feat

    def projection(self, num_edges):
        if torch.clamp(self.adj_changes, 0, 1).sum() > num_edges:#将所有修改的边进行求和，当前修改的边数是否超过数量
            #print('high')通过二分法调整邻接矩阵的修改量
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, num_edges, epsilon=1e-5)
            #从 self.adj_changes.data 中减去之前通过二分法找到的 miu 值，调整邻接矩阵的变化量。使用 clamp 限制修改量，确保所有值都在 [0, 1] 之间。
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
        else:
            #对其进行限制操作，不需要使用二分法。
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))

    def get_modified_adj2(self):

        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes
        m = m + m.t()

        return m

    def get_modified_adj(self, ori_adj):
        #生成全为1的补集矩阵，形状与原来邻接矩阵一样，减去一个对角线为1的单位矩阵
        if self.complementary is None:#none
            self.complementary = torch.ones_like(ori_adj) - torch.eye(self.nnodes).to(self.device)
        #全 0 的矩阵，大小为 nnodes x nnodes
        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        #生成下三角矩阵的索引（不包括对角线）
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        #矩阵 m 的下三角部分被填充为修改后的边
        m[tril_indices[0], tril_indices[1]] = self.adj_changes##############adj_changes初始为0
        #将 m 的转置加回到 m 上。这样 m 变成了一个对称矩阵，表示无向图中的边
        m = m + m.t()
        #self.complementary * m（逐元素乘积） 补集矩阵控制修改边的区域，只修改非自环的部分，+ ori_adj 表示在保留原始图结构的基础上添加这些修改
        modified_adj = self.complementary * m + ori_adj

        return modified_adj
    def proxy_get_modified_adj(self, ori_adj,device):
        #生成全为1的补集矩阵，形状与原来邻接矩阵一样，减去一个对角线为1的单位矩阵
        if self.complementary is None:
            self.complementary = (torch.ones_like(ori_adj) - torch.eye(self.nnodes)).to(device)
        # print("------------------查询调整函数中的内容------------------------------")
        # print("complementary={}".format(self.complementary))
        # 全 0 的矩阵
        m = torch.zeros((self.nnodes, self.nnodes), device=device)
        # print("m={}".format(m))
        # 生成下三角矩阵的索引
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1, device=device)
        # print("tril_indices={}".format(tril_indices))
        # 矩阵 m 的下三角部分被填充为修改后的边
        # print("self.adj_changes的值为".format(self.adj_changes))
        m[tril_indices[0], tril_indices[1]] = self.adj_changes.to(device)
        # print("加入下三角值的m={}".format(m))

        # 将 m 的转置加回到 m 上
        m = m + m.t()
        # print("完整的m={}".format(m))

        # 确保 ori_adj 在同一设备上
        ori_adj = ori_adj.to(device)

        # 逐元素乘积并相加
        modified_adj = self.complementary * m + ori_adj
        # print("获得的modified_adj{}".format(modified_adj))
        # print("------------------打印完毕------------------------------")
        return modified_adj


    def pubmed_proxy_get_modified_adj(self, ori_adj, device):
        if self.complementary is None:
            self.complementary = (torch.ones_like(ori_adj) - torch.eye(self.nnodes)).to(device)

        m = torch.zeros((self.nnodes, self.nnodes), device=device)

        # 批量生成下三角索引
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1, device=device)

        # 检查并修正 adj_changes 的大小
        num_indices = tril_indices[0].numel()
        if self.adj_changes.numel() != num_indices:
            self.adj_changes = self.adj_changes[:num_indices]

        # 填充下三角矩阵
        m[tril_indices[0], tril_indices[1]] = self.adj_changes.to(device)

        # 添加上三角部分
        m = m + m.t()

        ori_adj = ori_adj.to(device)

        # 生成最终的修改邻接矩阵
        modified_adj = self.complementary * m + ori_adj

        return modified_adj

    def batched_tril_indices(self,nnodes, batch_size, device):
        tril_indices = []
        for start in range(0, nnodes, batch_size):
            end = min(start + batch_size, nnodes)
            tril_indices.append(
                torch.tril_indices(end - start, nnodes, offset=-1, device=device) + start
            )
        return torch.cat(tril_indices, dim=1)
    def SVD(self):
        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes.detach()
        m = m + m.t()
        U, S, V = np.linalg.svd(m.cpu().numpy())
        U, S, V = torch.FloatTensor(U).to(self.device), torch.FloatTensor(S).to(self.device), torch.FloatTensor(V).to(
            self.device)
        alpha = 0.02
        tmp = torch.zeros_like(S).to(self.device)
        diag_S = torch.diag(torch.where(S > alpha, S, tmp))
        adj = torch.matmul(torch.matmul(U, diag_S), V)
        return adj[tril_indices[0], tril_indices[1]]

    def filter(self, Z):
        A = torch.zeros(Z.size()).to(self.device)
        return torch.where(Z > 0.9, Z, A)

    def bisection(self, a, b, num_edges, epsilon):
        def func(x):
            return torch.clamp(self.adj_changes - x, 0, 1).sum() - num_edges

        miu = a
        while ((b - a) >= epsilon):
            miu = (a + b) / 2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu) * func(a) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
        return miu

#em = self.embedding(ori_features, adj_norm)
    def dot_product_decode(self, Z):#em
        Z = F.normalize(Z, p=2, dim=1)#对张量进行归一化， p=2 表示采用 L2 范数，dim=1：表示对每个节点的嵌入向量进行归一化
        # print("归一化Z={}".format(Z))

        A_pred = torch.relu(torch.matmul(Z, Z.t()))#计算矩阵 Z 的点积，这个矩阵表示每对节点之间的相似度或连接预测值
        # print("预测A_pred={}".format(A_pred))
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)#返回一个二维索引数组，表示矩阵的下三角部分的索引
        # print("下三角索引tril_indices={}".format(tril_indices))

        return A_pred[tril_indices[0], tril_indices[1]]#使用 tril_indices 索引从 A_pred 中提取下三角部分的值，包含邻接矩阵 A_pred 的下三角部分
