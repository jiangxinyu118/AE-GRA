import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import argparse
from dataset import Dataset
from scipy.sparse import csr_matrix
import utils
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from models.gcn import GCN, embedding_GCN
from utils import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch_geometric.utils import from_scipy_sparse_matrix, dense_to_sparse
class GATtargetModel(nn.Module):
    def __init__(self, num_features, nclass):
        super(GATtargetModel, self).__init__()
        self.gat1 = GATConv(num_features, 8, heads=8)
        self.gat2 = GATConv(8 * 8, nclass)
        self.nclass = nclass
        self.nfeat = None
        self.hidden_sizes = None

    def forward(self, x, edge_index):
        if edge_index.size(1) == 0:
            raise ValueError("edge_index is empty.")
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

class embedding_GAT(nn.Module):
    def __init__(self, num_features, nclass):
        super(embedding_GAT, self).__init__()
        self.gat1 = GATConv(num_features, 8, heads=8) 
        self.nclass = nclass
        self.nfeat = None
        self.hidden_sizes = None

    def forward(self, x, edge_index):
        if edge_index.size(1) == 0:
            raise ValueError("edge_index is empty.")
        x = F.elu(self.gat1(x, edge_index))
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
    """
    num_nodes = features.size(0)
    device = features.device
    edge_list = []

    features = torch.nn.functional.normalize(features, p=2, dim=1)

    for start in range(0, num_nodes, block_size):
        torch.cuda.empty_cache()
        end = min(start + block_size, num_nodes)

        sim_block = torch.mm(features[start:end], features.T)  # [block_size, num_nodes]
        rows, cols = torch.where(sim_block >= threshold)   
        rows += start   
        edge_list.append(torch.stack([rows, cols], dim=0))

        print(f"处理块 {start}-{end} 完成，边数：{rows.size(0)}")

    edge_index = torch.cat(edge_list, dim=1)
    return edge_index.to(device)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='usair',
                        choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'enzyme','pubmed', 'AIDS', 'usair', 'brazil'],
                        help='dataset')
    args = parser.parse_args()
    data = Dataset(root=' ', name=args.dataset, setting='GCN')

    adj, features, labels= data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_train = torch.tensor(idx_train).to(device)
    idx_val = torch.tensor(idx_val).to(device)
    idx_test = torch.tensor(idx_test).to(device)

    labels = torch.LongTensor(labels).to(device)
    features = torch.FloatTensor(np.array(features.todense())).to(device)
    if sp.issparse(adj):
        edge_index, _ = from_scipy_sparse_matrix(adj)   
    else:
        edge_index, _ = dense_to_sparse(torch.FloatTensor(adj))

    edge_index = edge_index.to(device)
    torch.cuda.empty_cache()
    target_model = GATtargetModel(num_features=features.shape[1], nclass=4).to(device)
    optimizer = torch.optim.Adam(target_model.parameters(), lr=0.01)
    patience = 10  
    best_acc_val = 0
    best_model_params = None
    early_stopping_counter = 0
    criterion = nn.CrossEntropyLoss()
    for epoch in range(200):
        torch.cuda.empty_cache()
        target_model.train()
        optimizer.zero_grad()
        output = target_model(features, edge_index)
        loss= criterion(output[idx_train], labels[idx_train])

        torch.cuda.empty_cache()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, train_loss: {loss.item()}')

        if (epoch + 1) % 10 == 0:
            target_model.eval()
            with torch.no_grad():
                val_output = target_model(features, edge_index)
                val_loss=F.cross_entropy(val_output[idx_val], labels[idx_val])
                val_accuracy = utils.accuracy(val_output[idx_val], labels[idx_val])
                print(f'val_loss : {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
                if val_accuracy > best_acc_val:
                    best_acc_val = val_accuracy
                    best_model_params = target_model.state_dict().copy()
                    early_stopping_counter = 0   
                else:
                    early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break
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

