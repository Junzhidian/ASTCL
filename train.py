import argparse
import configparser
import datetime
import logging
import os
import time
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

from dataloader import preprocess_dataset, CrossDataset #,preprocess_datasets
from models.model import ASTCL
from utils import fit_delimiter
from utils import masked_MAE, masked_MAPE, masked_RMSE
from utils import MaskedMAELoss
from utils import model_summary

torch.set_num_threads(1)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
gpu_available = torch.cuda.is_available()
if gpu_available:
    device = torch.device("cuda")
else:
    device = 'cpu'
print(device)

def initialization(args):
    #Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    # Create log dir
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(f'runs/{args.data}', f'exp {timestamp}')
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    args.log_dir = log_dir

    # Initialize logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='',
                        filename=os.path.join(log_dir, f"log_training.txt"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    args.log_plotting = os.path.join(log_dir, f"log_plotting.txt")

    # Save hyper-parameters
    logging.info(fit_delimiter('Hyper-parameters', 80))
    for arg in vars(args):
        logging.info(f'{arg}={getattr(args, arg)}')

    return logging


# Tool function
def prepare_dataloaders(args):
    print('Transform Dataset...')
    # Convert data to the sub-spacetime format
    if isinstance(args.data, str):
        train_samples_path, train_targets_path, train_graph_path,\
        val_samples_path, val_targets_path, val_graph_path,\
        test_samples_path, test_targets_path, test_graph_path = \
            preprocess_dataset(args.data, t_in=args.t_history, t_out=args.t_pred,
                               num_nodes=args.num_nodes, keep_days=args.keep_days,
                               percent=args.percent, interval=args.interval,
                               train = args.train, val=args.val, test = args.test,
                               debug=args.debug, type=args.type)
    else:
        raise Exception('Check args.data!')

    if 'test_samples_path' in args and 'test_targets_path' in args:
        test_samples_path = args.test_samples_path
        test_targets_path = args.test_targets_path

    print('Construct DataLoader...')
    # Training set loader
    train_set = CrossDataset(train_samples_path, train_targets_path, train_graph_path)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True, num_workers=0)
    # Validation set loader
    val_set = CrossDataset(val_samples_path, val_targets_path, val_graph_path)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size,
                                shuffle=False, drop_last=False, num_workers=0)

    # Test set loader
    test_set = CrossDataset(test_samples_path, test_targets_path, test_graph_path)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size,
                                 shuffle=False, drop_last=False, num_workers=0)

    return train_dataloader, val_dataloader, test_dataloader


def train_batch(model, x, y, g, optimizer, criterions, device):
    x = x.to(device=device, dtype=torch.float)
    y = y.to(device=device, dtype=torch.float)
    g = g.to(device=device, dtype=torch.float)

    optimizer.zero_grad()
    output = model(x,g)
    #print(output)
    loss = compute_loss(criterions, output, y)
    loss.backward()
    optimizer.step()

    return loss, output

def compute_loss( criterions, pred, true):
    loss = criterions[0](pred, true)

    return loss

def train(args, logging, train_dataloader, val_dataloader):
    # Define model
    num_heads = [int(i) for i in list(args.num_heads.split(','))]
    channels = [int(i) for i in list(args.channels.split(','))]
    model = ASTCL(channels, num_heads, args.depth, args.partial, args.num_features,
                     args.t_history, args.t_pred, node_num=args.num_nodes, dropout=args.dropout,
                     revin=args.revin, graph=args.graph)
    
    model = model.to(device=args.device)

    # Warm start
    if 'warmstart' in args:
        print('Fine tune:',args.warmstart)
        checkpoint = torch.load(args.warmstart)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Model summary
    table, total_params = model_summary(model)
    # pa = count_parameters(model)
    logging.info(f'{table}')
    logging.info(f'Total Trainable Params: {total_params}')

    # Define loss and optimizer
    loss_criterions = []
    loss1 = nn.L1Loss()
    loss_mask = MaskedMAELoss()
    loss_criterions.append(loss1)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20,40,60,80,100], gamma=0.8)  # 0.5,0.7.0.9

    training_losses = []
    validation_losses = []
    validation_metrics = {'MAEs': [], 'RMSEs': [], 'MAPEs': []}
    time_all = 0
    early_stop = 15
    for epoch in range(args.epochs):
        t = time.time()
        logging.info(f"------------- Epoch: {epoch:03d} -----------")
        logging.info(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                     f"train batches: {len(train_dataloader)}, "
                     f"val batches: {len(val_dataloader)}, ")

        # Train
        print('Training')
        batches_train_loss = []
        loop = tqdm(train_dataloader, ncols=100)
        for data, target, graph in loop:
            model.train()
            x = data.to(device=args.device, dtype=torch.float)
            y = target.to(device=args.device, dtype=torch.float)
            g = graph.to(device=args.device, dtype=torch.float)
            loss, out = train_batch(model, x, y, g, optimizer, loss_criterions, args.device)
            batches_train_loss.append(loss.detach().cpu().numpy())

            loop.set_description(f'Train {epoch + 1}/{args.epochs}')
            loop.set_postfix(loss=np.mean(np.array(batches_train_loss)))
        training_losses.append(np.mean(np.array(batches_train_loss)))
        scheduler.step()
        t = time.time() - t
        time_all +=t
        if epoch % 10 == 0:
            print('learning rate:', optimizer.param_groups[0]['lr'])
        # Validation
        print('Validation')
        batches_val_loss = []
        batches_val_metrics = {'MAEs': [], 'RMSEs': [], 'MAPEs': []}
        with torch.no_grad():
            loop = tqdm(val_dataloader, ncols=100)
            for data,target,graph in loop:
                model.eval()
                x_val = data.to(device=args.device, dtype=torch.float)
                y_val = target.to(device=args.device, dtype=torch.float)
                g_val = graph.to(device=args.device, dtype=torch.float)
                out = model(x_val,g_val)
                val_loss = compute_loss(loss_criterions ,out, y_val).to(device="cpu")
                batches_val_loss.append((val_loss.detach().numpy()).item())

                # Metrics
                out_denormalized = out.detach().cpu().numpy().flatten()
                target_denormalized = y_val.detach().cpu().numpy().flatten()
                mae = masked_MAE(out_denormalized, target_denormalized)
                rmse = masked_RMSE(out_denormalized, target_denormalized)
                mape = masked_MAPE(out_denormalized, target_denormalized)
                if not (np.isnan(mae) or np.isnan(rmse) or np.isnan(mape)):
                    batches_val_metrics['MAEs'].append(mae)
                    batches_val_metrics['RMSEs'].append(rmse)
                    batches_val_metrics['MAPEs'].append(mape)

                loop.set_description(f'Val {epoch + 1}/{args.epochs}')
                move_mae = np.mean(np.array(batches_val_metrics['MAEs']))
                move_rmse = np.mean(np.array(batches_val_metrics['RMSEs']))
                move_mape = np.mean(np.array(batches_val_metrics['MAPEs']))
                loop.set_postfix(MAE=move_mae, RMSE=move_rmse, MAPE=move_mape)

            assert np.mean(np.array(batches_val_loss)) == sum(batches_val_loss) / len(batches_val_loss)
            epoch_val_loss = np.mean(np.array(batches_val_loss))
            validation_losses.append(epoch_val_loss)
            validation_metrics['MAEs'].append(np.mean(np.array(batches_val_metrics['MAEs'])))
            validation_metrics['RMSEs'].append(np.mean(np.array(batches_val_metrics['RMSEs'])))
            validation_metrics['MAPEs'].append(np.mean(np.array(batches_val_metrics['MAPEs'])))

            # Save model based on val loss
            if args.save:
                if epoch_val_loss == min(validation_losses):
                    best_time = 0
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()
                                }, os.path.join(args.log_dir, f'epoch{epoch}_checkpoint.pt'))
                    best_state_dict = copy.deepcopy(model.state_dict())
                else:
                    best_time +=1
                    print('poor performance', best_time)
                    if best_time >= early_stop:
                        logging.info(f"Early stop at epoch {epoch}, best results at epoch {epoch-early_stop}!")
                        logging.info(f"Training time: {datetime.timedelta(seconds=time_all)}\n")
                        break

        # Print epoch results
        logging.info(f"Pred {args.t_pred} steps - Training loss:   {training_losses[-1]:.8f}")
        logging.info(f"Pred {args.t_pred} steps - Validation loss: {validation_losses[-1]:.8f}")
        logging.info(f"Pred {args.t_pred} steps - Validation MAE:  {validation_metrics['MAEs'][-1]:.4f}")
        logging.info(f"Pred {args.t_pred} steps - Validation RMSE: {validation_metrics['RMSEs'][-1]:.4f}")
        logging.info(f"Pred {args.t_pred} steps - Validation MAPE: {validation_metrics['MAPEs'][-1]:.4f}")
        logging.info(f"Training time: {datetime.timedelta(seconds=time_all)}\n")

        with open(args.log_plotting, 'w') as f:
            print(f"Training loss={training_losses}", file=f)
            print(f"Validation loss={validation_losses}", file=f)
            print(f"Validation MAE={validation_metrics['MAEs']}", file=f)
            print(f"Validation RMSE={validation_metrics['RMSEs']}", file=f)
            print(f"Validation MAPE={validation_metrics['MAPEs']}", file=f)


    model.load_state_dict(best_state_dict)

    return training_losses, validation_losses, validation_metrics, model

def test(args, logging, model, test_dataloader):
    logging.info(f"Testing:")
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

    logging.info(f"All Steps MAE = {mae_all:.4f}, RMSE =  {rmse_all:.4f}, MAPE = {mape_all:.4f}")
    out_steps = y.shape[1]
    for i in range(out_steps):
        mae = masked_MAE(out[:,i],y[:,i])
        rmse = masked_RMSE(out[:,i],y[:,i])
        mape = masked_MAPE(out[:,i],y[:,i])
        logging.info(f"Step {i+1} MAE = {mae:.4f}, RMSE =  {rmse:.4f}, MAPE = {mape:.4f}")
    logging.info(f"Inference time: {(end-start):.2f} s")



# Main function
def main(args):
    # Initializing

    logging = initialization(args)
        
    # Prepare train/va/test dataloader
    train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(args)
    # assert next(iter(train_dataloader))[0].shape[-1] == 3

    # Training
    training_losses, validation_losses, validation_metrics, model = train(args, logging,
                                                                                    train_dataloader,
                                                                                    val_dataloader)
    test(args, logging, model, test_dataloader)



if __name__ == '__main__':
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
    # parser.add_argument('--warmstart', type=str, default='weights/metr_1_1.pt',help='innitial weight')
    args = parser.parse_args()

    main(args)
