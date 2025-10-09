from torch_geometric.utils import from_scipy_sparse_matrix, dense_to_sparse

from models.gcn import GCN, embedding_GCN
from topology_attack import PGDAttack
from trainGATTargetEmbeddingModel import GATtargetModel, embedding_GAT
from trainGraphSAGETargetEmbeddingModel import GraphSAGEtargetModel, embedding_GraphSAGE
from utils import *
from dataset import Dataset
import argparse
import utils
from scipy.sparse import csr_matrix
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, average_precision_score,precision_score, recall_score
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.datasets import Actor

def test(adj, features, labels, victim_model):
    adj, features, labels = to_tensor(adj, features, labels, device=device)

    victim_model.eval()
    adj_norm = normalize_adj_tensor(adj)
    output = victim_model(features, adj_norm)

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:", "loss= {:.4f}".format(loss_test.item()), "gcn测试节点预测accuracy= {:.4f}".format(acc_test.item()))

    return output.detach()

def dot_product_decode(Z):
    Z = F.normalize(Z, p=2, dim=1)#Z 的每一行归一化到单位范数（即每行的L2范数为1）
    Z = torch.matmul(Z, Z.t())#计算了 Z 和 Z 的转置 Z.t() 的点积（矩阵乘法）。结果的元素表示原始 Z 矩阵中每对节点之间的相似度。
    adj = torch.relu(Z-torch.eye(Z.shape[0]))#减去单位矩阵，只保留非对角线上的值，ReLU激活函数确保输出的值是非负的，保留非负的相似度值，并将负值设为0
    return adj

def preprocess_Adj(adj, feature_adj):
    n=len(adj)
    cnt=0
    adj=adj.numpy()
    feature_adj=feature_adj.numpy()
    for i in range(n):
        for j in range(n):
            if feature_adj[i][j]>0.14 and adj[i][j]==0.0:
                adj[i][j]=1.0
                cnt+=1
    print(cnt)
    return torch.FloatTensor(adj)

def transfer_state_dict(pretrained_dict, model_dict):
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict

def metric(ori_adj, inference_adj, idx):
    real_edge = ori_adj[idx, :][:, idx].reshape(-1)
    pred_edge = inference_adj[idx, :][:, idx].reshape(-1)
    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    index = np.where(real_edge == 0)[0]
    index_delete = np.random.choice(index, size=int(len(real_edge)-2*np.sum(real_edge)), replace=False)
    real_edge = np.delete(real_edge, index_delete)
    pred_edge = np.delete(pred_edge, index_delete)
    print("Inference attack AUC: %f AP(average precision): %f" % (auc(fpr, tpr), average_precision_score(real_edge, pred_edge)))



def metric_all(ori_adj, inference_adj, idx):
    real_edge = ori_adj[idx, :][:, idx].reshape(-1)
    pred_edge = inference_adj[idx, :][:, idx].reshape(-1)
    bin_pred_edge = (pred_edge >= 0.5).astype(int)
    TP = np.sum((real_edge == 1) & (bin_pred_edge == 1))  # 预测对了的边
    FP = np.sum((real_edge == 0) & (bin_pred_edge == 1))  # 错把无边预测为有边
    FN = np.sum((real_edge == 1) & (bin_pred_edge == 0))  # 把真实的边漏掉了
    print("TP: %f, FP: %f, FN: %f" % (TP,FP,FN))
    precision = TP / (TP + FP + 1e-10)  # 避免除 0
    recall = TP / (TP + FN + 1e-10)
    print(f"Precision: {precision:.6f}  Recall: {recall:.6f}")


def metric1(ori_adj, inference_adj, idx):
    inference_adj=inference_adj[0]
    real_edge = ori_adj[idx, :][:, idx].reshape(-1)
    pred_edge = inference_adj[idx, :][:, idx].reshape(-1)
    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    index = np.where(real_edge == 0)[0]
    index_delete = np.random.choice(index, size=int(len(real_edge)-2*np.sum(real_edge)), replace=False)
    real_edge = np.delete(real_edge, index_delete)
    pred_edge = np.delete(pred_edge, index_delete)
    print("Inference attack AUC: %f AP(average precision): %f" % (auc(fpr, tpr), average_precision_score(real_edge, pred_edge)))

def Auc(ori_adj, modified_adj, idx):
    real_edge = []
    pred_edge = []
    for i in idx:
        for j in idx:
            if i != j:
                real_edge.append(ori_adj[i][j])
                pred_edge.append(modified_adj[i][j])
                #pred_edge.append(np.dot(output[idx[i]], output[idx[j]])/(np.linalg.norm(output[idx[i]])*np.linalg.norm(output[idx[j]])))
                #pred_edge.append(-np.linalg.norm(output[idx[i]]-output[idx[j]]))
                #pred_edge.append(np.dot(features[idx[i]], features[idx[j]]) / (np.linalg.norm(features[idx[i]]) * np.linalg.norm(features[idx[j]])))

    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    print(auc(fpr, tpr))
    return auc(fpr, tpr)

def create_edge_index_from_features(features, threshold):
    sim_matrix = cosine_similarity(features.cpu().numpy())

    num_nodes = features.shape[0]
    edge_index = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if sim_matrix[i][j] > threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])

    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

def pubmed_create_edge_index_from_features2(features, threshold=0.5, block_size=3000):
    """
    """
    num_nodes = features.size(0)
    device = features.device
    edge_list = []

    features = torch.nn.functional.normalize(features, p=2, dim=1)

    for start in range(0, num_nodes, block_size):
        end = min(start + block_size, num_nodes)

        sim_block = torch.mm(features[start:end], features.T)  # [block_size, num_nodes]
        rows, cols = torch.where(sim_block >= threshold)  # 筛选出满足阈值的索引
        rows += start  # 调整行索引
        edge_list.append(torch.stack([rows, cols], dim=0))

        print(f"处理块 {start}-{end} 完成，边数：{rows.size(0)}")

    edge_index = torch.cat(edge_list, dim=1)
    return edge_index.to(device)

def pubmed_create_edge_index_from_features(features, threshold=0.5):
    sim_matrix = cosine_similarity(features.cpu().numpy())
    sim_matrix_sparse = csr_matrix(sim_matrix)
    sim_matrix_sparse = sim_matrix_sparse.multiply(sim_matrix_sparse >= threshold)
    sim_matrix_sparse.eliminate_zeros()
    rows, cols = sim_matrix_sparse.nonzero()  # 提取行和列索引
    edge_index = np.vstack((rows, cols))  # 合并为一个二维 NumPy 数组
    return torch.from_numpy(edge_index).to(features.device)  # 转为 PyTorch 张量并与 features 保持一致设备




def cosine_similarity_matrix(features):
    normalized_features = F.normalize(features, p=2, dim=1)
    print("获得归一化矩阵")
    similarity_matrix = torch.mm(normalized_features, normalized_features.t())
    print("获得相似度矩阵")
    return similarity_matrix


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def find_neighbors_threshold(model, features, adj_norm, node_idx, N,epsilon=0.1, threshold=0.1):
    perturbed_features = features.clone()
    perturbed_features[node_idx] += torch.randn_like(perturbed_features[node_idx]) * epsilon
    p1 = model(features, adj_norm)
    p2 = model(perturbed_features, adj_norm)
    delta_p = (p2 - p1).abs().sum(dim=1)#其他节点类别预测变化总量
    # print(f"delta_p for node {node_idx}: {delta_p}")
    affected_nodes = torch.where(delta_p > threshold)[0]
    return affected_nodes


def find_delta_p_matrix(model, features, adj_norm, epsilon=0.1):
    N = features.shape[0]
    delta_p_matrix = torch.zeros(N, N).to(features.device)

    p1 = model(features, adj_norm)

    for node_idx in tqdm(range(N), desc="Perturbing nodes"):
        perturbed_features = features.clone()
        perturbed_features[node_idx] += torch.randn_like(perturbed_features[node_idx]) * epsilon

        p2 = model(perturbed_features, adj_norm)
        delta_p = (p2 - p1).abs().sum(dim=1)
        delta_p[node_idx] = 0  # 不考虑自己
        delta_p_matrix[node_idx] = delta_p
    torch.cuda.empty_cache()
    print("预测差值矩阵为")
    print(delta_p_matrix)
    return delta_p_matrix

def build_topk_adj_from_asymmetric_delta(delta_p_matrix, density=0.0015):
    N = delta_p_matrix.shape[0]
    K = int(density * N * (N - 1) / 2)
    print("K值为")
    print(K)
    delta_p_matrix = delta_p_matrix.clone()
    delta_p_matrix.fill_diagonal_(-1)
    torch.cuda.empty_cache()
    flat = delta_p_matrix.flatten()
    topk = torch.topk(flat, k=min(K, flat.numel()))#为了避免 K 大于【flat.numel()=flat 张量中元素的数量=N^2 】的总元素数而导致报错。
    topk_indices = topk.indices
    rows = topk_indices // N
    cols = topk_indices % N
    adj_reconstructed = torch.zeros((N, N), device=delta_p_matrix.device)
    for i, j in zip(rows, cols):
        adj_reconstructed[i, j] = 1
        adj_reconstructed[j, i] = 1  # 如果你希望是无向图（可选）

    return adj_reconstructed




def reconstruct_graph_by_gradient_diff_K(model, features, adj_norm, density=0.005, epsilon=0.1):
    N = features.shape[0]

    delta_matrix = torch.zeros((N, N), device=features.device)
    for node_idx in tqdm(range(N), desc="Perturbing nodes"):
        features_clean = features.clone().detach().requires_grad_(True)
        output_clean = model(features_clean, adj_norm)
        output_sum_clean = output_clean.sum()
        model.zero_grad()
        output_sum_clean.backward()
        grads_before = features_clean.grad.detach().clone()  # N x F
        features_perturbed = features.clone().detach()
        features_perturbed[node_idx] += torch.randn_like(features_perturbed[node_idx]) * epsilon
        features_perturbed.requires_grad_(True)

        output_perturbed = model(features_perturbed, adj_norm)

        output_sum_perturbed = output_perturbed.sum()
        model.zero_grad()
        output_sum_perturbed.backward()
        grads_after = features_perturbed.grad.detach().clone()

        grad_diff = (grads_after - grads_before).norm(p=2, dim=1)
        delta_matrix[node_idx] = grad_diff

        del features_clean, features_perturbed, grads_before, grads_after

    delta_matrix.fill_diagonal_(-1)
    flat = delta_matrix.flatten()
    K = int(density * N * (N - 1) / 2)
    K = min(K, flat.numel())
    topk = torch.topk(flat, k=K)
    topk_indices = topk.indices
    rows = topk_indices // N
    cols = topk_indices % N

    adj_reconstructed = torch.zeros((N, N), device=features.device)
    adj_reconstructed[rows, cols] = 1
    adj_reconstructed[cols, rows] = 1  # 对称

    return adj_reconstructed


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to optimize in GraphMI attack.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['cora','enzyme', 'actor','AIDS', 'usair','cora_ml', 'citeseer', 'polblogs', 'pubmed',  'brazil'], help='dataset')
parser.add_argument('--density', type=float, default=1.0, help='Edge density estimation')
parser.add_argument('--model', type=str, default='PGD', choices=['PGD', 'min-max'], help='model variant')
parser.add_argument('--nlabel', type=float, default=0.1)


args = parser.parse_args()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root=' ', name=args.dataset, setting='GCN')

adj, features, labels, init_adj = data.adj, data.features, data.labels, data.init_adj

adj_coo = adj.tocoo()

idx_train, idx_val, idx_test= data.idx_train, data.idx_val, data.idx_test
idx_random = np.array(random.sample(range(adj.shape[0]), int(adj.shape[0]*args.nlabel)))
num_edges = int(0.5 * args.density * adj.sum()/adj.shape[0]**2 * len(idx_random)**2)
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, onehot_feature=False)
init_adj = torch.FloatTensor(init_adj.todense())
init_adj.to(device)
 

victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
                   dropout=0.5, weight_decay=5e-4, device=device)
best_victim_params = torch.load(' ')
victim_model = victim_model.to(device)
victim_model.load_state_dict(best_victim_params)
embedding = embedding_GCN(nfeat=features.shape[1], nhid=16, device=device)
embedding.load_state_dict(transfer_state_dict(victim_model.state_dict(), embedding.state_dict()))
model = PGDAttack(model=victim_model, embedding=embedding, nnodes=adj.shape[0], loss_type='CE', device=device)
model = model.to(device)



def main():
    model.embeddingattackGCN(adj, features, init_adj, labels, idx_train, num_edges, epochs=args.epochs)
    adj_reconstructed = model.modified_adj.cpu()
    print('attack cora calculating edge inference AUC&AP ===')
    for i in range(5):  # 评估 5 次
        idx_random = np.array(random.sample(range(adj.shape[0]), int(adj.shape[0] * args.nlabel)))
        print(f"\n=== Evaluation Round {i + 1} ===")
        metric(adj.numpy(), adj_reconstructed.numpy(), idx_random)


if __name__ == '__main__':
    main()
