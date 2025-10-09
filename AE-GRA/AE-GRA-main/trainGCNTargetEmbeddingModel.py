import sys

from models.gcn import GCN, embedding_GCN
from topology_attack import PGDAttack
from utils import *
from dataset import Dataset
import argparse
from sklearn.metrics import roc_curve, auc, average_precision_score
import random
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

#victim_model.state_dict(), embedding.state_dict()
def transfer_state_dict(pretrained_dict, model_dict):
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict

#adj.numpy(), inference_adj.numpy(), idx_attack
def metric(ori_adj, inference_adj, idx):
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

    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    print(auc(fpr, tpr))
    return auc(fpr, tpr)



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
parser.add_argument('--dataset', type=str, default='usair',
                    choices=['cora', 'actor','cora_ml', 'citeseer', 'polblogs', 'enzyme','pubmed', 'AIDS', 'usair', 'brazil'], help='dataset')
parser.add_argument('--density', type=float, default=1.0, help='Edge density estimation')
parser.add_argument('--model', type=str, default='PGD', choices=['PGD', 'min-max'], help='model variant')
parser.add_argument('--nlabel', type=float, default=0.1)

args = parser.parse_args() 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("-----------正在用"+str(device)+"训练-----------")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root=' ', name=args.dataset, setting='GCN')


adj, features, labels, init_adj = data.adj, data.features, data.labels, data.init_adj
idx_train, idx_val, idx_test= data.idx_train, data.idx_val, data.idx_test
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, onehot_feature=False)

feature_adj = dot_product_decode(features)#获得归一化相似度矩阵


init_adj = torch.FloatTensor(init_adj.todense())#大小是节点数X节点数；将稀疏矩阵转换为密集矩阵，将密集矩阵转换为 PyTorch 张量

# Setup Victim Model
victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=128,
                   dropout=0.5, weight_decay=5e-4, device=device)

victim_model = victim_model.to(device)
victim_model.fit(features, adj, labels, idx_train, idx_val)

best_model_params = torch.load(' ')
victim_model.load_state_dict(best_model_params)
test(adj, features, labels, victim_model)





