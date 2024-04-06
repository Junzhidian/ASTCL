import math
import pickle
import time
import metric
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import linalg
from fastdtw import fastdtw

import numpy as np
from prettytable import PrettyTable


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self):
        pass
    
    def fit(self, X):
        
        self._mean = X.mean()
        self._std = X.std()

    def transform(self, X):
        return (X - self._mean) / self._std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        return (X * self._std) + self._mean
    
class MinMaxScaler():
    def __init__(self):
        pass

    def fit(self, X):
        self.mins = X.min(axis=tuple(range(len(X.shape)-1)))
        self.maxs = X.max(axis=tuple(range(len(X.shape)-1)))
        self._min = self.mins[0]
        self._max = self.maxs[0]
        #print("min:", self._min, "max:", self._max, "mean", self._mean)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def get_normalized_features(X):
    means = np.mean(X, axis=(0, 1))  
    X = X - means.reshape((1, 1, -1))
    stds = np.std(X, axis=(0, 1))  
    X = X / stds.reshape((1, 1, -1))
    return X, means, stds

def normalize_distance(X):
    temp = X[:,1]
    std = np.square(np.std(temp))
    #print(std)
    if std==0:
        normallize = 1
    else:
        normallize = np.exp(-(np.square(temp) / std))
    X[:,1] = normallize
    return X

def normalize_conectivity(X,threshold):
    X = X.astype(float)
    temp = X
    std = np.square(np.std(temp),)
    #print(std)
    if std==0:
        normallize = 1
    else:
        normallize = np.exp(-(np.square(temp) / std))
    X = normallize
    X[X < threshold] = 0
    return X

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def masked_MAE(pred, target):
    valid_indices = np.nonzero(target)
    pred = pred[valid_indices]
    target = target[valid_indices]
    return np.mean(np.absolute(pred - target))


def masked_MSE(pred, target):
    valid_indices = np.nonzero(target)
    pred = pred[valid_indices]
    target = target[valid_indices]
    return np.mean((pred - target) ** 2)


def masked_RMSE(pred, target):
    valid_indices = np.nonzero(target)
    pred = pred[valid_indices]
    target = target[valid_indices]
    return np.sqrt(np.mean((pred - target) ** 2))


def masked_MAPE(pred, target):
    valid_indices = np.nonzero(target)
    pred = pred[valid_indices]
    target = target[valid_indices]
    return np.mean(np.absolute((pred - target) / (target + 1e-5))) * 100

def masked_mae_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class MaskedMAELoss:
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=0.0):
        return masked_mae_loss(preds, labels, null_val)



def elapsed_time_format(total_time):
    hour = 60 * 60
    minute = 60
    if total_time < 60:
        return f"{math.ceil(total_time):d} secs"
    elif total_time > hour:
        hours = divmod(total_time, hour)
        return f"{int(hours[0]):d} hours, {elapsed_time_format(hours[1])}"
    else:
        minutes = divmod(total_time, minute)
        return f"{int(minutes[0]):d} mins, {elapsed_time_format(minutes[1])}"


def model_summary(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    return table, total_params


def fit_delimiter(string='', length=80, delimiter="="):
    result_len = length - len(string)
    half_len = math.floor(result_len / 2)
    result = delimiter * half_len + string + delimiter * half_len
    return result


class Speedometer:
    def __init__(self, metric_names, metric_indexes):

        self._metric_names = metric_names
        self._metric_indexes = metric_indexes
        self.reset()

    def reset(self):
        self._metrics = metric.Metrics(self._metric_names, self._metric_indexes)
        self._start = time.time()
        self._tic = time.time()
        self._counter = 0

    def update(self, preds, labels, step_size=1):
        self._metrics.update(preds, labels)
        self._counter += step_size
        time_spent = time.time() - self._tic
        return self._metrics.get_value()



    def finish(self):
        out_str = str(self._metrics)
        
        return self._metrics.get_value()

def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

class DualBranchContrast(torch.nn.Module):
    def __init__(self, loss, mode, intraview_negs=False, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None):
        l1 = self.loss(anchor=h1, sample=h2)
        l2 = self.loss(anchor=h2, sample=h1)
        return (l1 + l2) * 0.5


class InfoNCE(object):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob.diag()
        return -loss.mean()

    def __call__(self, anchor, sample) -> torch.FloatTensor:
        loss = self.compute(anchor, sample)
        return loss
    

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def gen_adj(sample_sequence):
    node = sample_sequence.shape[0]
    graph=np.ones((node,node))
    for j in range(node-1):
        ds = []
        for m in range(j+1,node):
            d,_ = fastdtw(sample_sequence[j],sample_sequence[m])
            ds.append(d)
        graph[j,j+1:] = ds
        graph[j+1:,j] = ds
    adj = asym_adj(normalize_conectivity(graph,0.05))

    return adj

def load_adj(adj_mx, adjtype):
    adj = []
    if adjtype == "scalap":
        for i in range(adj_mx.shape[0]):
            adj.append(calculate_scaled_laplacian(adj_mx[i]))
    elif adjtype == "normlap":
        for i in range(adj_mx.shape[0]):
            adj.append(calculate_normalized_laplacian(adj_mx[i]).astype(np.float32).todense())
    elif adjtype == "symnadj":
        for i in range(adj_mx.shape[0]):
            adj.append(sym_adj(adj_mx[i]))
    elif adjtype == "transition":
        for i in range(adj_mx.shape[0]):
            adj.append(asym_adj(adj_mx[i]))
    elif adjtype == "doubletransition":
        for i in range(adj_mx.shape[0]):
            adj.append(asym_adj(adj_mx[i]), asym_adj(np.transpose(adj_mx[i])))
    elif adjtype == "identity":
        for i in range(adj_mx.shape[0]):
            adj.append(np.diag(np.ones(adj_mx.shape[1])).astype(np.float32))
    else:
        error = 0
        assert error, "adj type not defined"
    return adj
