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
        if attack_structure:
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(torch.FloatTensor(int(nnodes * (nnodes - 1) / 2)))
            self.adj_changes.data.fill_(0)

        if attack_features:#False
            assert True, 'Topology Attack does not support attack feature'


    def sample_negative_edges(self,adj, num_neg_samples):
        N = adj.shape[0]
        with torch.no_grad():
            mask = (adj == 0).triu(1)  
            neg_indices = mask.nonzero(as_tuple=False)
            rand_idx = torch.randperm(neg_indices.shape[0])[:num_neg_samples]
            sampled_neg_edges = neg_indices[rand_idx]
        return sampled_neg_edges  # Tensor of shape [num_samples, 2]

    def embeddingattackGCN(self, adj,ori_features, ori_adj, labels, idx_random, num_edges,
               epochs=200, sample=False, **kwargs):
        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        embedding=self.embedding
        victim_model = self.surrogate 
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
        self.sparse_features = sp.issparse(ori_features)  
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        embedding=self.embedding
        victim_model = self.surrogate  
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
            adj_norm = utils.normalize_adj_tensor(modified_adj)  
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

        self.sparse_features = sp.issparse(ori_features)
        ori_adj = torch.FloatTensor(ori_adj).to(device)
        ori_features = ori_features.detach().to(device).float()
        embedding=self.embedding
        victim_model = self.surrogate  
        victim_model.eval()
        self.embedding.eval()
        N = em_target.shape[0]
        torch.cuda.empty_cache()
        edge_index_change = torch.empty((2, 0), dtype=torch.long)
        for t in tqdm(range(30)):  # 200
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
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
        adj_matrix = torch.zeros(batch_size, node_features_dim, node_features_dim, device=lower_tri_elements.device)
        tril_indices = torch.tril_indices(row=node_features_dim, col=node_features_dim, offset=0)
        adj_matrix[:, tril_indices[0], tril_indices[1]] = lower_tri_elements
        adj_matrix = adj_matrix + adj_matrix.transpose(1, 2) - torch.diag_embed(torch.diagonal(adj_matrix, dim1=1, dim2=2))
        return adj_matrix


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
        self.sparse_features = sp.issparse(features)  
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
        tau = 0 
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
        rowsum = adj.sum(1)  
        r_inv = rowsum.flatten()  
        D = torch.diag(r_inv)  
        L = D - adj 

        r_inv = r_inv + 1e-3
        r_inv = r_inv.pow(-1 / 2).flatten() 
        r_inv[torch.isinf(r_inv)] = 0.  
        r_mat_inv = torch.diag(r_inv)  

        L = torch.matmul(torch.matmul(r_mat_inv, L), r_mat_inv)

        
        if len(X.shape) == 2:
            X = X.unsqueeze(0)  

        
        XLXT = torch.matmul(torch.matmul(X.permute(0, 2, 1), L), X)  

        
        if XLXT.dim() == 3:  # batch_size, num_features, num_features
            loss_smooth_feat = torch.sum(torch.diagonal(XLXT, dim1=1, dim2=2)) 
        else:
            raise ValueError("Expected XLXT to be a 3D tensor but got shape: {}".format(XLXT.shape))

        return loss_smooth_feat

    def projection(self, num_edges):
        if torch.clamp(self.adj_changes, 0, 1).sum() > num_edges:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, num_edges, epsilon=1e-5)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))

    def get_modified_adj2(self):

        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes
        m = m + m.t()

        return m

    def get_modified_adj(self, ori_adj):
        if self.complementary is None:#none
            self.complementary = torch.ones_like(ori_adj) - torch.eye(self.nnodes).to(self.device)
        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes##############adj_changes初始为0
        m = m + m.t()
        modified_adj = self.complementary * m + ori_adj

        return modified_adj
    def proxy_get_modified_adj(self, ori_adj,device):
        if self.complementary is None:
            self.complementary = (torch.ones_like(ori_adj) - torch.eye(self.nnodes)).to(device)
        m = torch.zeros((self.nnodes, self.nnodes), device=device)
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1, device=device)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes.to(device)
        m = m + m.t()
        ori_adj = ori_adj.to(device)
        modified_adj = self.complementary * m + ori_adj
        return modified_adj


    def pubmed_proxy_get_modified_adj(self, ori_adj, device):
        if self.complementary is None:
            self.complementary = (torch.ones_like(ori_adj) - torch.eye(self.nnodes)).to(device)

        m = torch.zeros((self.nnodes, self.nnodes), device=device)

        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1, device=device)

        num_indices = tril_indices[0].numel()
        if self.adj_changes.numel() != num_indices:
            self.adj_changes = self.adj_changes[:num_indices]

        m[tril_indices[0], tril_indices[1]] = self.adj_changes.to(device)

        m = m + m.t()

        ori_adj = ori_adj.to(device)

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

    def dot_product_decode(self, Z):#em
        Z = F.normalize(Z, p=2, dim=1) 
        A_pred = torch.relu(torch.matmul(Z, Z.t())) 
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1) 

        return A_pred[tril_indices[0], tril_indices[1]]
