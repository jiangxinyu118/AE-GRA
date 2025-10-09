import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import argparse
from dataset import Dataset
import utils
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models.gcn import GCN
from trainGATTargetEmbeddingModel import GATtargetModel
from utils import *
from torch_geometric.utils import from_scipy_sparse_matrix, dense_to_sparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GraphSAGEtargetModel(nn.Module):
    def __init__(self, num_features, nclass, threshold=0.5):
        super(GraphSAGEtargetModel, self).__init__()
        self.conv1 = SAGEConv(num_features, 8)
        self.conv2 = SAGEConv(8, nclass)
        self.threshold = threshold
        self.nclass = nclass
        self.nfeat = None
        self.hidden_sizes = None

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class embedding_GraphSAGE(nn.Module):
    def __init__(self, num_features, nclass, threshold=0.5):
        super(embedding_GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, 8)
        self.threshold = threshold
        self.nclass = nclass
        self.nfeat = None
        self.hidden_sizes = None

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return x



def create_edge_index_from_features(features, threshold):
    sim_matrix = cosine_similarity(features.cpu().numpy())
    num_nodes = features.shape[0]
    edge_index = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if sim_matrix[i][j] > threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])

    return torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)

def pubmed_create_edge_index_from_features2(features, threshold=0.5, block_size=500):#1000
    """
    分块计算余弦相似度，仅保留大于阈值的边。
    """
    num_nodes = features.size(0)
    device = features.device
    edge_list = []

    # 归一化特征矩阵
    features = torch.nn.functional.normalize(features, p=2, dim=1)

    for start in range(0, num_nodes, block_size):
        torch.cuda.empty_cache()
        end = min(start + block_size, num_nodes)

        # 计算分块与全体的相似度
        sim_block = torch.mm(features[start:end], features.T)  # [block_size, num_nodes]
        rows, cols = torch.where(sim_block >= threshold)  # 筛选出满足阈值的索引
        rows += start  # 调整行索引
        edge_list.append(torch.stack([rows, cols], dim=0))

        print(f"处理块 {start}-{end} 完成，边数：{rows.size(0)}")

    # 合并所有边索引
    edge_index = torch.cat(edge_list, dim=1)
    return edge_index.to(device)



def main():
    print("训练graphsage机制目标模型")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='usair',
                        choices=['cora', 'cora_ml', 'citeseer','enzyme', 'polblogs', 'pubmed', 'AIDS', 'usair', 'brazil'],
                        help='dataset')
    args = parser.parse_args()
    data = Dataset(root=' ', name=args.dataset, setting='GCN')

    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_train = torch.tensor(idx_train).to(device)
    idx_val = torch.tensor(idx_val).to(device)
    idx_test = torch.tensor(idx_test).to(device)

    labels = torch.LongTensor(labels).to(device)
    features = torch.FloatTensor(np.array(features.todense())).to(device)
    if sp.issparse(adj):
        print("邻接矩阵是稀疏矩阵")
        edge_index, _ = from_scipy_sparse_matrix(adj)  # 直接用 scipy.sparse 格式
    else:
        print("邻接矩阵是稠密矩阵")
        edge_index, _ = dense_to_sparse(torch.FloatTensor(adj))


    edge_index = edge_index.to(device)
    print(edge_index)
    torch.cuda.empty_cache()

    target_model = GraphSAGEtargetModel(num_features=features.shape[1], nclass=labels.max().item() + 1).to(device)
    optimizer = torch.optim.Adam(target_model.parameters(), lr=0.01)
    patience = 10  # 如果验证集损失连续patience个epoch没有改善，则停止训练
    best_acc_val = 0
    best_model_params = None
    early_stopping_counter = 0
    criterion = nn.CrossEntropyLoss()

    for epoch in range(200):
        target_model.train()
        optimizer.zero_grad()
        output = target_model(features, edge_index)
        loss = criterion(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, train_loss: {loss.item()}')

        if (epoch + 1) % 10 == 0:
            target_model.eval()
            with torch.no_grad():
                val_output = target_model(features, edge_index)
                val_loss = F.cross_entropy(val_output[idx_val], labels[idx_val])
                val_accuracy = utils.accuracy(val_output[idx_val], labels[idx_val])
                print(f'val_loss : {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
                # 如果验证集损失有改善，保存当前模型参数
                if val_accuracy > best_acc_val:
                    best_acc_val = val_accuracy
                    print("保存模型字典时的验证acc是" + str(val_accuracy))
                    best_model_params = target_model.state_dict().copy()
                    early_stopping_counter = 0  # 重置早停计数器
                else:
                    early_stopping_counter += 1

        # 检查是否需要早
        if early_stopping_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break
        # 加载性能最好的模型参数
    if best_model_params is not None:
        target_model.load_state_dict(best_model_params)
        torch.save(best_model_params,
                   ' ')

    target_model.eval()
    with torch.no_grad():
        test_output = target_model(features, edge_index)
        test_accuracy = utils.accuracy(test_output[idx_test], labels[idx_test])
        print(f'Test Accuracy: {test_accuracy:.4f}')


if __name__ == '__main__':
    main()
