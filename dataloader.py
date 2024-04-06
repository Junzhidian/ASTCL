import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from fastdtw import fastdtw
from utils import normalize_distance
from utils import gen_adj
import torch

# Sensor
class Node():
    def __init__(self, id, dist, target_node):
        self.id = id
        self.dist = dist
        self.target_node = target_node

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def __lt__(self, other):
        return self.dist < other.dist

    def __le__(self, other):
        return self.dist <= other.dist

    def __gt__(self, other):
        return self.dist > other.dist

    def __ge__(self, other):
        return self.dist >= other.dist

    def __repr__(self):
        return f'Node id: {self.id} connectivaty with Node {self.target_node}: {self.dist:.4f}'


class CrossDataset(Dataset):
    # Downsample timesteps for training
    def __init__(self, samples_path, targets_path, graph_path):

        self._samples = np.load(samples_path)
        self._targets = np.load(targets_path)
        self._adj = np.load(graph_path)
        assert self._samples.shape[0] == self._targets.shape[0] == self._adj.shape[0]

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self._samples[idx], self._targets[idx], self._adj[idx]



def load_data(data_path, sensor_percent):
    adj_mat_path = (os.path.join(data_path, r"adj_mat.npy"))
    feature_path = (os.path.join(data_path, r"node_values.npy"))

    A = np.load(os.path.normpath(adj_mat_path))
    X = np.load(os.path.normpath(feature_path))

    #(N,N)
    A = A.astype(np.float32)
    #(N,T,C)
    X = X.astype(np.float32)

    if sensor_percent !=1:
        number_sensors = X.shape[0]
        partial_sensors = int(number_sensors * sensor_percent)
        selected_sensors = np.random.choice(number_sensors, size=partial_sensors, replace=False)
        X_percent = X[selected_sensors,::]
    else:
        X_percent = X
    return A, X, X_percent


def find_similar_nodes(target_node, X, num_nodes):
    #X[N,t_in,F]
    node_dists = []
    for i,node in enumerate(X):
        dist, path = fastdtw(node,target_node)   
        node_dists.append([i,dist])
    #(N,2)
    node_dists = normalize_distance(np.array(node_dists))

    sorted_idx = np.argsort(node_dists[:, 1])[::-1]
    node_dists = node_dists[sorted_idx]
    node_dists = node_dists[:num_nodes,:]
    node_ids = [int(each[0]) for each in node_dists]
    node_connectivity = [each[1] for each in node_dists]
    
    return node_ids, node_connectivity

def prepare_samples_targets_list_speed(A, X, num_nodes, t_in, t_out, keep_days=7, interval=288, debug_flag=False,
                                        target_nodes='all', grid=False):
    traffic_feature_idx = 0  

    all_nodes = X.shape[0]
    num_times = X.shape[1]      
    num_feats = X.shape[2] 

    if target_nodes != 'all':
        target_node_list = target_nodes
    else:
        target_node_list = tqdm(range(all_nodes))  # all nodes

    samples, targets, graphs = [], [], []
    for i,node in enumerate(target_node_list):
        target_node = node
        # Downsampling in time dimension if necessary
        if keep_days != 0:
            time_stamps = int(keep_days * interval) + (t_in + t_out)
            all_times = range(num_times - (t_in + t_out))
            
            downsample_size = min(int(time_stamps), len(all_times))
            # choosen = range(downsample_size - (t_in + t_out))

            choosen = range(num_times-time_stamps,num_times- (t_in + t_out))
            indices = [(i, i + (t_in + t_out)) for i in choosen]
        else:  # keep all data
            indices = [(i, i + (t_in + t_out)) for i in range(num_times - (t_in + t_out))]

        # Convert data into training samples and targets
        for i, j in indices:
            sample_distant = np.zeros((num_nodes, t_in, num_feats))

            node_ids, node_connectivity = find_similar_nodes(X[target_node, i:i+t_in,:], X[:, i:i+t_in,:], num_nodes)
            sample_distant[:, :, :] = X[node_ids, i: i + t_in, :]
            sample = sample_distant
            graph = gen_adj(sample)
            #print(sample_distant.shape)
            target = X[target_node, i + t_in: j, traffic_feature_idx]

            graphs.append(graph)
            samples.append(sample)
            targets.append(target)
        
    return samples, targets, graphs

def prepare_samples_targets_list_flow(A, X, num_nodes, t_in, t_out, keep_days=7, interval=288, debug_flag=False,
                                        target_nodes='all', grid=False):
    traffic_feature_idx = 0  # speed must be the first feature

    all_nodes = X.shape[0]
    num_times = X.shape[1]      
    num_feats = X.shape[2]

    if target_nodes != 'all':
        target_node_list = target_nodes
    else:
        target_node_list = tqdm(range(all_nodes))  # all nodes

    samples, targets, graphs = [], [], []
    for node in target_node_list:
        target_node = node
        # Flatten ids and distances

        # Downsampling in time dimension if necessary

        if keep_days != 0:
            time_stamps = keep_days * interval + (t_in + t_out)
            all_times = range(num_times - (t_in + t_out))
            
            downsample_size = min(int(time_stamps), len(all_times))
            choosen = range(downsample_size - (t_in + t_out))
            indices = [(i, i + (t_in + t_out)) for i in choosen]
        else:  # keep all data
            indices = [(i, i + (t_in + t_out)) for i in range(num_times - (t_in + t_out))]

        # Convert data into training samples and targets
        for i, j in indices:
            sample_distant = np.zeros((num_nodes, t_in, num_feats))

            node_distant_ids, node_distant_connectivity = find_similar_nodes(X[target_node, i:i+t_in,:], X[:, i:i+t_in,:],  num_nodes)
            sample_distant[:, :, :] = X[node_distant_ids, i: i + t_in, :]

            sample = sample_distant
            graph = gen_adj(sample)
            target = X[target_node, i + t_in: j, traffic_feature_idx]
    
            samples.append(sample)
            targets.append(target)
            graphs.append(graph)
        
    return samples, targets, graphs


# %% Convert X,A to a collection of samples, targets
def preprocess_dataset(data, t_in=12, t_out=3, num_nodes=15, keep_days=7, percent=1, interval=288,
                       train=True, val=False, test=False, debug=False, target_nodes='all', test_flag=False, 
                        type='speed'):
    # 70% train, 20% validation, 10% test
    assert isinstance(data, str)
    if target_nodes != 'all':  # Predict on node-of-interest only
        train_samples_path = os.path.join(data, rf'train_samples_{t_in}_{t_out}_{keep_days}_{percent}_{num_nodes}_vtar{target_nodes}.npy')
        train_targets_path = os.path.join(data, rf'train_targets_{t_in}_{t_out}_{keep_days}_{percent}_{num_nodes}_vtar{target_nodes}.npy')
        val_samples_path = os.path.join(data, rf'val_samples_{t_in}_{t_out}_{num_nodes}_vtar{target_nodes}.npy')
        val_targets_path = os.path.join(data, rf'val_targets_{t_in}_{t_out}_{num_nodes}_vtar{target_nodes}.npy')
        test_samples_path = os.path.join(data, rf'test_samples_{t_in}_{t_out}_{num_nodes}_vtar{target_nodes}.npy')
        test_targets_path = os.path.join(data, rf'test_targets_{t_in}_{t_out}_{num_nodes}_vtar{target_nodes}.npy')
    else:  # Predict on all nodes
        train_samples_path = os.path.join(data, rf'train_samples_{t_in}_{t_out}_{keep_days}_{percent}_{num_nodes}.npy')
        train_targets_path = os.path.join(data, rf'train_targets_{t_in}_{t_out}_{keep_days}_{percent}_{num_nodes}.npy')
        val_samples_path = os.path.join(data, rf'val_samples_{t_in}_{t_out}_{num_nodes}.npy')
        val_targets_path = os.path.join(data, rf'val_targets_{t_in}_{t_out}_{num_nodes}.npy')
        test_samples_path = os.path.join(data, rf'test_samples_{t_in}_{t_out}_{num_nodes}.npy')
        test_targets_path = os.path.join(data, rf'test_targets_{t_in}_{t_out}_{num_nodes}.npy')
        train_graph_path = os.path.join(data, rf'train_graph_{keep_days}_{percent}_{num_nodes}.npy')
        val_graph_path = os.path.join(data, rf'val_graph_{num_nodes}.npy')
        test_graph_path = os.path.join(data, rf'test_graph_{num_nodes}.npy')

    # Check if data already exists
    if os.path.isfile(train_samples_path) and os.path.isfile(train_targets_path) and os.path.isfile(train_graph_path) \
            and os.path.isfile(val_samples_path) and os.path.isfile(val_targets_path) and os.path.isfile(val_graph_path) \
            and os.path.isfile(test_samples_path) and os.path.isfile(test_targets_path) and os.path.isfile(test_graph_path):
        return train_samples_path, train_targets_path, train_graph_path, \
               val_samples_path, val_targets_path, val_graph_path,\
               test_samples_path, test_targets_path, test_graph_path
               

    if test_flag and os.path.isfile(test_samples_path) and os.path.isfile(test_targets_path):
        return train_samples_path, train_targets_path, train_graph_path, \
               val_samples_path, val_targets_path, val_graph_path,\
               test_samples_path, test_targets_path, test_graph_path

    print("Cannot find the data, will prepare and save the data in proper format...")

    A, X, X_percent = load_data(data, percent)
    #(N,N)
    print('A shape:', A.shape)
    #(N,T,C)  (207,34272,2)
    print('X shape:', X.shape)

    if type=='speed':
        prepare_samples_targets_list = prepare_samples_targets_list_speed
    elif type=='flow':
        prepare_samples_targets_list = prepare_samples_targets_list_flow
    else:
        raise Exception('Unsupport adj matrix shape')

    # Split train/val/test speed  7:1:2   flow 6:2:2
    if type == 'speed':
        cut_point1 = int(X.shape[1] * 0.7)
    else:
        cut_point1 = int(X.shape[1] * 0.6)
    cut_point2 = int(X.shape[1] * 0.8)

    train_X = np.expand_dims(X_percent[:, :cut_point1,0],axis=-1)
    val_X = np.expand_dims(X[:, cut_point1:cut_point2,0],axis=-1)
    test_X = np.expand_dims(X[:, cut_point2:,0],axis=-1)

    if train:
        print('Prepare train dataset')
        train_samples, train_targets, train_graphs = prepare_samples_targets_list(A, train_X, num_nodes, t_in, t_out,
                                                                keep_days=keep_days, interval=interval,
                                                                debug_flag=debug, target_nodes=target_nodes)
        print('Saving train set to disk...')
        np.save(train_samples_path, np.array(train_samples))
        np.save(train_targets_path, np.array(train_targets))
        np.save(train_graph_path, np.array(train_graphs))
    if val:
        print('Prepare val dataset')
        val_samples, val_targets, val_graphs = prepare_samples_targets_list(A, val_X, num_nodes, t_in, t_out,
                                                            keep_days=0, interval=interval,
                                                            debug_flag=debug, target_nodes=target_nodes)
        print('Saving val set to disk...')
        np.save(val_samples_path, np.array(val_samples))
        np.save(val_targets_path, np.array(val_targets))
        np.save(val_graph_path, np.array(val_graphs))
    if test:
        print('Prepare test dataset')
        test_samples, test_targets, test_graphs = prepare_samples_targets_list(A, test_X, num_nodes, t_in, t_out,
                                                              keep_days=0, interval=interval,
                                                              debug_flag=debug, target_nodes=target_nodes)
        print('Saving test set to disk...')
        np.save(test_samples_path, np.array(test_samples))
        np.save(test_targets_path, np.array(test_targets))
        np.save(test_graph_path, np.array(test_graphs))

    return train_samples_path, train_targets_path, train_graph_path, \
            val_samples_path, val_targets_path, val_graph_path,\
            test_samples_path, test_targets_path, test_graph_path


