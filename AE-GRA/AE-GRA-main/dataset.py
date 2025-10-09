import numpy as np
import random
import scipy.sparse as sp
import os.path as osp
import os
import urllib.request
import sys
import pickle as pkl
import networkx as nx
from utils import get_train_val_test, get_train_val_test_gcn_more_train_node, get_train_val_test_gcn, \
    get_train_val_test_gcn_different_distribution, \
    get_train_val_test_gcn_80_train_node
import scipy.io as sio

class Dataset():
    """Dataset class contains four citation network datasets "cora", "cora-ml", "citeseer" and "pubmed",
    and one blog dataset "Polblogs".
    The 'cora', 'cora-ml', 'poblogs' and 'citeseer' are downloaded from https://github.com/danielzuegner/gnn-meta-attack/tree/master/data, and 'pubmed' is from https://github.com/tkipf/gcn/tree/master/gcn/data.

    Parameters
    ----------
    root :
        root directory where the dataset should be saved.
    name :
        dataset name, it can be choosen from ['cora', 'citeseer', 'cora_ml', 'polblogs', 'pubmed']
    setting :
        there are two data splits settings. The 'nettack' setting follows nettack paper where they select the largest connected components of the graph and use 10%/10%/80% nodes for training/validation/test . The 'gcn' setting follows gcn paper where they use 20 samples in each class for traing, 500 nodes for validation, and 1000 nodes for test. (Note here 'gcn' setting is not a fixed split, i.e., different random seed would return different data splits)
    seed :
        random seed for splitting training/validation/test.
    require_mask :
        setting require_mask True to get training, validation and test mask (self.train_mask, self.val_mask, self.test_mask)

    Examples
    --------
	We can first create an instance of the Dataset class and then take out its attributes.

	>>> from deeprobust.graph.data import Dataset
	>>> data = Dataset(root='/tmp/', name='cora')
	>>> adj, features, labels = data.adj, data.features, data.labels
	>>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    """

    # root='', name=args.dataset=cora, setting='GCN'
    def __init__(self, root, name, setting='nettack', seed=None, require_mask=False):
        self.name = name.lower()#cora
        self.setting = setting.lower()#GCN

        assert self.name in ['cora', 'actor','citeseer', 'cora_ml', 'polblogs', 'pubmed', 'aids', 'enzyme', 'usair', 'brazil',\
                             'blogcatalog','europe'], \
            'Currently only support cora, citeseer, cora_ml, polblogs, pubmed, aids, enzyme, usair, brazil, europe'
        assert self.setting in ['gcn', 'nettack'], 'Settings should be gcn or nettack'

        self.seed = seed
        # self.url =  'https://raw.githubusercontent.com/danielzuegner/nettack/master/data/%s.npz' % self.name
        self.url =  'https://raw.githubusercontent.com/danielzuegner/gnn-meta-attack/master/data/%s.npz' % self.name
        self.root = osp.expanduser(osp.normpath(root))
        self.data_folder = osp.join(root, self.name)# #/home2/jxy/GraphMI-main/data/cora
        self.data_filename = self.data_folder + '.npz'#/home2/jxy/GraphMI-main/data/cora.npz
        self.require_mask = require_mask

        self.require_lcc = True if setting == 'nettack' else False
        self.adj, self.features, self.labels = self.load_data()

        self.init_adj = self.init_matrix(self.adj)

        self.idx_train, self.idx_val, self.idx_test= self.get_train_val_test()
        if self.require_mask:#False
            self.get_mask()

    def get_train_val_test(self):
        """Get training, validation, test splits according to self.setting (either 'nettack' or 'gcn').
        """
        if self.setting == 'nettack':#测试集占比大
            return get_train_val_test(nnodes=self.adj.shape[0], val_size=0.1, test_size=0.8, stratify=self.labels, seed=self.seed)
        if self.setting == 'gcn':#划分更均匀
            # return get_train_val_test_gcn(self.labels, seed=self.seed)
            # return get_train_val_test_gcn_more_train_node(self.labels, seed=self.seed)
            return get_train_val_test_gcn_80_train_node(self.labels, seed=self.seed)

            # return get_train_val_test_gcn_different_distribution(self.labels, seed=self.seed)


    def load_data(self):
        print('Loading {} dataset...'.format(self.name))

        if self.name == 'actor':
            return self.load_Actor()

        if self.name == 'pubmed':
            return self.load_pubmed()

        if self.name == 'aids':
            return self.load_AIDS()

        if self.name == 'usair':
            return self.load_usair()

        if self.name == 'brazil':
            return self.load_brazil()

        if self.name == 'europe':
            return self.load_europe()

        if self.name == 'enzyme':
            return self.load_enzyme()

        if self.name == 'blogcatalog':
            return self.load_blog()

        if not osp.exists(self.data_filename):#数据文件不存在，我才去下载，如果存在了，我就直接加载
            self.download_npz()

        adj, features, labels = self.get_adj()#比较特殊的数据集的加载都列到前面了，有.npz后缀的文件加载统一在这里实现
        return adj, features, labels

    def download_npz(self):
        """Download adjacen matrix npz file from self.url.
        """
        print('Dowloading from {} to {}'.format(self.url, self.data_filename))
        try:
            urllib.request.urlretrieve(self.url, self.data_filename)
        except:
            raise Exception('''Download failed! Make sure you have stable Internet connection and enter the right name''')

    def download_pubmed(self, name):
        url = 'https://raw.githubusercontent.com/tkipf/gcn/master/gcn/data/'
        try:
            urllib.request.urlretrieve(url + name, osp.join(self.root, name))
        except:
            raise Exception('''Download failed! Make sure you have stable Internet connection and enter the right name''')

    def load_Actor(self):
        data_dir='/hy-tmp/GraphMI-main/data/Actor'
        edge_path = os.path.join(data_dir, 'out1_graph_edges.txt')
        feat_label_path = os.path.join(data_dir, 'out1_node_feature_label.txt')

        ### === Step 1: Load edges ===
        edge_list = []
        with open(edge_path, 'r') as f:
            next(f)  # skip header
            for line in f:
                src, dst = map(int, line.strip().split('\t'))
                edge_list.append((src, dst))
                edge_list.append((dst, src))

        edge_array = np.array(edge_list, dtype=np.int32)
        num_nodes = edge_array.max() + 1

        adj = sp.coo_matrix(
            (np.ones(len(edge_array)), (edge_array[:, 0], edge_array[:, 1])),
            shape=(num_nodes, num_nodes),
            dtype=np.float32
        )
        adj = adj.tocsr()

        ### === Step 2: Load features and labels ===
        features = sp.lil_matrix((num_nodes, 932), dtype=np.float32)
        labels = np.full((num_nodes,), -1, dtype=np.int64)  # 初始化为-1表示无标签

        with open(feat_label_path, 'r') as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue
                node_id = int(parts[0])
                feat_indices = list(map(int, parts[1].split(',')))
                label = int(parts[2])

                for idx in feat_indices:
                    features[node_id, idx] = 1.0
                labels[node_id] = label

        features = features.tocsr()

        return adj, features, labels
    def load_pubmed(self):
        dataset = 'pubmed'
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            name = "ind.{}.{}".format(dataset, names[i])
            data_filename = osp.join(self.root, name)

            if not osp.exists(data_filename):
                self.download_pubmed(name)

            with open(data_filename, 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)


        test_idx_file = "ind.{}.test.index".format(dataset)
        if not osp.exists(osp.join(self.root, test_idx_file)):
            self.download_pubmed(test_idx_file)

        test_idx_reorder = parse_index_file(osp.join(self.root, test_idx_file))
        test_idx_range = np.sort(test_idx_reorder)

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        labels = np.where(labels)[1]
        return adj, features, labels

    def load_AIDS(self):
        dataset = 'AIDS'
        g = np.zeros((1429, 1429))
        with open('%s/%s_A.txt' % (dataset, dataset), 'r') as f:
            for _ in range(2948):
                row = f.readline().strip().replace(',',' ').split()
                i, j = [int(w) for w in row]
                g[i-1][j-1] = 1
        adj = sp.csr_matrix(g)

        features = []
        with open('%s/%s_node_attributes.txt' % (dataset, dataset), 'r') as f:
            for _ in range(1429):
                row = f.readline().strip().replace(',',' ').split()
                features.append([float(w) for w in row])
        features = sp.csr_matrix(features)

        labels = np.loadtxt('%s/%s_node_labels.txt' % (dataset, dataset))
        labels = np.array(labels, dtype='int8')[:1429]

        return adj, features, labels

    def load_enzyme(self):
        dataset = 'ENZYMES'
        g = np.zeros((6254, 6254))
        with open('%s/%s_A.txt' % (dataset, dataset), 'r') as f:
            for _ in range(23914):
                row = f.readline().strip().replace(',',' ').split()
                i, j = [int(w) for w in row]
                g[i-1][j-1] = 1
        adj = sp.csr_matrix(g)

        features = []
        with open('%s/%s_node_attributes.txt' % (dataset, dataset), 'r') as f:
            for _ in range(6254):
                row = f.readline().strip().replace(',',' ').split()
                features.append([float(w) for w in row])
        features = sp.csr_matrix(features)

        labels = np.loadtxt('%s/%s_node_labels.txt' % (dataset, dataset))
        labels = np.array(labels, dtype='int8')[:6254]
        labels = labels - 1 #使类别索引从0开始
        return adj, features, labels

    def load_usair(self):
        dataset = 'usair'
        node_dict = {}
        f = np.loadtxt('%s/%s_lable.txt' % (dataset, dataset))
        id = f[:, 0]
        labels = f[:, 1]
        for i in range(len(id)):
            node_dict[id[i]] = i
        g = np.zeros((1190, 1190))
        with open('%s/%s_A.txt' % (dataset, dataset), 'r') as f:
            for _ in range(13582):
                row = f.readline().strip().split()
                i, j = [int(w) for w in row]
                g[node_dict[i]][node_dict[j]] = 1
                g[node_dict[j]][node_dict[i]] = 1
        adj = sp.csr_matrix(g)

        features = np.identity(1190)
        features = sp.csr_matrix(features)

        labels = np.array(labels, dtype='int8')

        return adj, features, labels

    def load_europe(self):
        dataset = 'europe'
        node_dict = {}
        f = np.loadtxt('%s/%s_lable.txt' % (dataset, dataset))
        id = f[:, 0]
        labels = f[:, 1]
        for i in range(len(id)):
            node_dict[id[i]] = i
        g = np.zeros((399, 399))
        with open('%s/%s_A.txt' % (dataset, dataset), 'r') as f:
            for _ in range(5995):
                row = f.readline().strip().split()
                i, j = [int(w) for w in row]
                g[node_dict[i]][node_dict[j]] = 1
                g[node_dict[j]][node_dict[i]] = 1
        adj = sp.csr_matrix(g)

        features = np.identity(399)
        features = sp.csr_matrix(features)

        labels = np.array(labels, dtype='int8')

        return adj, features, labels

    def load_brazil(self):
        dataset = 'brazil'
        node_dict = {}
        f = np.loadtxt('%s/%s_lable.txt' % (dataset, dataset))
        id = f[:, 0]
        labels = f[:, 1]
        for i in range(len(id)):
            node_dict[id[i]] = i
        g = np.zeros((131, 131))
        with open('%s/%s_A.txt' % (dataset, dataset), 'r') as f:
            for _ in range(1074):
                row = f.readline().strip().split()
                i, j = [int(w) for w in row]
                g[node_dict[i]][node_dict[j]] = 1
                g[node_dict[j]][node_dict[i]] = 1
        adj = sp.csr_matrix(g)

        features = np.identity(131)
        features = sp.csr_matrix(features)

        labels = np.array(labels, dtype='int8')

        return adj, features, labels

    def load_blog(self):
        data = sio.loadmat('blogcatalog.mat')
        adj = data['network']
        lable = data['group'].todense()
        labels = np.array(np.argmax(lable, 1).squeeze(1)).squeeze()

        features = np.identity(10312)
        features = sp.csr_matrix(features)

        labels = np.array(labels, dtype='int8')

        return adj, features, labels

    def get_adj(self):
        adj, features, labels = self.load_npz(self.data_filename)
        adj = adj + adj.T
        adj = adj.tolil()
        adj[adj > 1] = 1

        if self.require_lcc:
            lcc = self.largest_connected_components(adj)
            adj = adj[lcc][:, lcc]
            features = features[lcc]
            labels = labels[lcc]
            assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

        # whether to set diag=0?
        adj.setdiag(0)
        adj = adj.astype("float32").tocsr()
        adj.eliminate_zeros()

        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

        return adj, features, labels

    def load_npz(self, file_name, is_sparse=True):
        with np.load(file_name) as loader:
            # loader = dict(loader)
            if is_sparse:
                adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                            loader['adj_indptr']), shape=loader['adj_shape'])
                if 'attr_data' in loader:
                    features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                 loader['attr_indptr']), shape=loader['attr_shape'])
                else:
                    features = None
                labels = loader.get('labels')
            else:
                adj = loader['adj_data']
                if 'attr_data' in loader:
                    features = loader['attr_data']
                else:
                    features = None
                labels = loader.get('labels')
        if features is None:
            features = np.eye(adj.shape[0])
        features = sp.csr_matrix(features, dtype=np.float32)
        return adj, features, labels

    def largest_connected_components(self, adj, n_components=1):
        """Select k largest connected components.

		Parameters
		----------
		adj : scipy.sparse.csr_matrix
			input adjacency matrix
		n_components : int
			n largest connected components we want to select
		"""

        _, component_indices = sp.csgraph.connected_components(adj)
        component_sizes = np.bincount(component_indices)
        components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
        nodes_to_keep = [
            idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
        print("Selecting {0} largest connected components".format(n_components))
        return nodes_to_keep

    def __repr__(self):
        return '{0}(adj_shape={1}, feature_shape={2})'.format(self.name, self.adj.shape, self.features.shape)

    def get_mask(self):
        idx_train, idx_val, idx_test = self.idx_train, self.idx_val, self.idx_test
        labels = self.onehot(self.labels)

        def get_mask(idx):
            mask = np.zeros(labels.shape[0], dtype=np.bool)
            mask[idx] = 1
            return mask

        def get_y(idx):
            mx = np.zeros(labels.shape)
            mx[idx] = labels[idx]
            return mx

        self.train_mask = get_mask(self.idx_train)
        self.val_mask = get_mask(self.idx_val)
        self.test_mask = get_mask(self.idx_test)
        self.y_train, self.y_val, self.y_test = get_y(idx_train), get_y(idx_val), get_y(idx_test)

    def onehot(self, labels):
        eye = np.identity(labels.max() + 1)
        onehot_mx = eye[labels]
        return onehot_mx

    def init_matrix(self, adj):
        n = adj.shape[0]
        result = np.zeros((n, n))
        return sp.csr_matrix(result)#将全零的result数组转换为一个压缩稀疏行（CSR）矩阵，只存储非0元素


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

