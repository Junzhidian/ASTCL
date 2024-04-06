import argparse
import configparser
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import preprocess_dataset, CrossDataset #,preprocess_datasets
from models.model import ASTCL
from utils import masked_MAE, masked_MAPE, masked_RMSE

torch.set_num_threads(1)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpu_available = torch.cuda.is_available()
if gpu_available:
    device = torch.device("cuda")
else:
    device = 'cpu'
print(device)

DATASET = 'PEMS08'  
parser = argparse.ArgumentParser()
config_file = 'config/{}.conf'.format(DATASET)
print('Read configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)

parser.add_argument('--seed', type=int, default=config['model']['seed'], help='random seed')  #seed set 1 or 7
parser.add_argument('--device', type=str, default=device, help='device')
parser.add_argument('--gpu_id', type=str, default='2', help='gpu id')
parser.add_argument('--data', type=str, nargs='+', default='data/PEMS08', help='data path')
parser.add_argument('--keep_days', type=float, default=config['model']['keep_days'],
                        help='number of the days to training the model')
parser.add_argument('--percent', type=float, default=config['model']['percent'],
                        help='percent of traffic sensors to training the model')
parser.add_argument('--interval', type=float, default=config['model']['interval'],
                        help='timestamp interval')
parser.add_argument('--type', type=str, default=config['model']['type'], help='prediction target')
parser.add_argument('--num_nodes', type=int, default=config['model']['num_nodes'], help='number of nodes in traffic graph')
parser.add_argument('--num_features', type=int, default=config['model']['num_features'], help='traffic event: task, timestamp, conectivity')
parser.add_argument('--num_heads', type=str, default=config['model']['num_heads'], help='number of attention heads')
parser.add_argument('--partial', type=int, default=config['model']['partial'], help='partial of channels')
parser.add_argument('--channels', type=str, default=config['model']['channels'], help='feature dimension in model')
parser.add_argument('--depth', type=int, default=config['model']['depth'], help='depth of model')
parser.add_argument('--t_history', type=int, default=config['model']['t_history'], help='T_h')
parser.add_argument('--t_pred', type=int, default=config['model']['t_pred'], help='T_r')
parser.add_argument('--target_node', type=int, default=0, help='target node to predict')
parser.add_argument('--epochs', type=int, default=config['model']['epochs'], help='number of epochs')
parser.add_argument('--lr', type=float, default=config['model']['lr'], help='learning rate')
parser.add_argument('--weight_decay', type=float, default=config['model']['weight_decay'], help='weight decay rate')
parser.add_argument('--batch_size', type=int, default=config['model']['batch_size'], help='batch size')
parser.add_argument('--dropout', type=float, default=config['model']['dropout'], help='dropout rate')
parser.add_argument('--revin', type=eval, default=config['model']['revin'], help='revin normalization')
parser.add_argument('--graph', type=eval, default=config['model']['graph'], help='sequence graph')
parser.add_argument('--train', type=eval, default=config['model']['train'], help='whether genetate train dataset')
parser.add_argument('--val', type=eval, default=config['model']['val'], help='whether genetate val dataset')
parser.add_argument('--test', type=eval, default=config['model']['test'], help='whether genetate test dataset')
parser.add_argument('--save', action='store_true', default=True, help='whether save model')
parser.add_argument('--debug', action='store_true', default=False, help='debug mode, faster')
parser.add_argument('--warmstart', type=str, default='weights/metr_1_1.pt',help='innitial weight')
args = parser.parse_args()


print('Prepareing dataset...')
train_samples_path, train_targets_path, train_graph_path,\
        val_samples_path, val_targets_path, val_graph_path,\
        test_samples_path, test_targets_path, test_graph_path = \
    preprocess_dataset(args.data, t_in=args.t_history, t_out=args.t_pred,
                               num_nodes=args.num_nodes, keep_days=args.keep_days,
                               percent=args.percent, interval=args.interval,
                               train = args.train, val=args.val, test = args.test,
                               debug=args.debug, type=args.type)
test_set = CrossDataset(test_samples_path, test_targets_path, test_graph_path)
test_dataloader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, drop_last=False, num_workers=0)

def test():
    num_heads = [int(i) for i in list(args.num_heads.split(','))]
    channels = [int(i) for i in list(args.channels.split(','))]
    model = ASTCL(channels, num_heads, args.depth, args.partial, args.num_features,
                     args.t_history, args.t_pred, node_num=args.num_nodes, dropout=args.dropout,
                     revin=args.revin, graph=args.graph)
    checkpoint = torch.load('weights/metr_1_0.2.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    loop = tqdm(test_dataloader, ncols=110)
    batches_test_metrics = {'MAEs': [], 'RMSEs': [], 'MAPEs': []}
    y = []
    out = []
    start = time.time()
    for data,target,graph in loop:
        x_batch = data.to(device=args.device, dtype=torch.float)
        y_batch = target.to(device=args.device, dtype=torch.float)
        g_batch = graph.to(device=args.device, dtype=torch.float)
        out_batch = model(x_batch,g_batch)

                # Metrics
        out_batch = out_batch.detach().cpu().numpy()
        y_batch = y_batch.detach().cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

        mae = masked_MAE(out_batch, y_batch)
        rmse = masked_RMSE(out_batch, y_batch)
        mape = masked_MAPE(out_batch, y_batch)
        if not (np.isnan(mae) or np.isnan(rmse) or np.isnan(mape)):
            batches_test_metrics['MAEs'].append(mae)
            batches_test_metrics['RMSEs'].append(rmse)
            batches_test_metrics['MAPEs'].append(mape)

            loop.set_description(f'Test')
            move_mae = np.mean(np.array(batches_test_metrics['MAEs']))
            move_rmse = np.mean(np.array(batches_test_metrics['RMSEs']))
            move_mape = np.mean(np.array(batches_test_metrics['MAPEs']))
            loop.set_postfix(MAE=move_mae, RMSE=move_rmse, MAPE=move_mape)
    end = time.time()
    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()
    mae_all = masked_MAE(out, y)
    rmse_all = masked_RMSE(out, y)
    mape_all = masked_MAPE(out, y)

    print(f"All Steps MAE = {mae_all:.4f}, RMSE =  {rmse_all:.4f}, MAPE = {mape_all:.4f}")
    out_steps = y.shape[1]
    for i in range(out_steps):
        mae = masked_MAE(out[:,i],y[:,i])
        rmse = masked_RMSE(out[:,i],y[:,i])
        mape = masked_MAPE(out[:,i],y[:,i])
        print(f"Step {i+1} MAE = {mae:.4f}, RMSE =  {rmse:.4f}, MAPE = {mape:.4f}")
    print(f"Inference time: {(end-start):.2f} s")

if __name__ == '__main__':
    test()