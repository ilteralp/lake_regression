#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:56:33 2021

@author: melike
"""
import torch
from torch.utils.data import Dataset, random_split, Subset, DataLoader
from torch.optim import RMSprop
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import h5py
import os
import os.path as osp
from sklearn.model_selection import KFold
from datetime import datetime
import constants as C
from datasets import Lake2dDataset
from metrics import Metrics
from models import DandadaDAN

"""
Takes a dataset and splits it into train, test and validation sets. 
"""
def split_dataset(dataset, test_ratio, val_ratio=None):
    test_len = int(np.round(len(dataset) * test_ratio))
    tr_len = len(dataset) - test_len
    if val_ratio is not None:
        val_len = int(np.round(len(dataset) * val_ratio))
        tr_len = tr_len - val_len
        tr_set, val_set, test_set = random_split(dataset, [tr_len, val_len, test_len], 
                                                 generator=torch.Generator().manual_seed(42))
        return tr_set, val_set, test_set
    else:
        tr_set, test_set = random_split(dataset, [tr_len, test_len], 
                                        generator=torch.Generator().manual_seed(42))
        return tr_set, test_set
    
"""
Takes a train set loader and args. Calculates mean and std of patches and
regression values. Adapted from 
https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/2
Do it like https://www.youtube.com/watch?v=y6IEcEBRZks
"""
def calc_mean_std(train_loader):
    patches_mean, patches_std = 0., 0.
    regs_mean, regs_std = 0., 0.
    num_samples = 0.
    for batch_id, data in enumerate(train_loader):
        patches, date_type, reg_vals, (img_idxs, pxs, pys) = data
        batch_samples = patches.size(0)
        patches = patches.view(batch_samples, patches.size(1), -1)              # (num_samples, 12, 3, 3) -> (num_samples, 12, 9)
        patches_mean += patches.mean(2).sum(0)                                  # 12 means and 12 stds
        patches_std += patches.std(2).sum(0)
        regs_mean += reg_vals.mean()
        regs_std += reg_vals.std()
        num_samples += batch_samples
    patches_mean /= num_samples
    patches_std /= num_samples
    regs_mean /= (batch_id + 1)
    regs_std /= (batch_id + 1)
    return patches_mean, patches_std, regs_mean, regs_std
    
"""
Takes a model, resets layer weights
https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/7
"""
def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()
        
"""
Returns verbose message with loss and score.
"""
def get_msg(loss, score, e, dataset):
    msg = "Epoch #{}, Losses (R+C): {}, {:.2f} = {:.2f}".format(
        e, dataset, np.mean(loss[e]['total']), np.mean(loss[e]['l_reg_loss']))
    if 'l_class_loss' in loss[e]:
        msg += " + {:.2f}".format(np.mean(loss[e]['l_class_loss']))
    # msg += "\t Scores, MAE: {:.2f}, R2: {:.2f}, RMSE: {:.2f}".format(
        # np.mean(score[e]['mae']), np.mean(score[e]['r2']), np.mean(score[e]['rmse']))
    return msg

"""
Checks given args
"""
def verify_args(args):
    if args['num_folds'] is not None and args['num_folds'] < 3:
        raise Exception('Number of folds should be at least 3 since validation and test set have the same size.')
    if args['test_per'] >= 0.5:
        raise Exception('Test percent should be less than 0.5 since validation set has the same length with it.')

"""
Plots loss and scores to Tensorboard
"""
def plot(writer, tr_loss, val_loss, tr_scores, val_scores, epoch):
    """ Losses """
    writer.add_scalar('1_Loss/train (total)', np.mean(tr_loss[epoch]['total']))
    writer.add_scalar('1_Loss/val (total)', np.mean(val_loss[epoch]['total']))
    writer.add_scalar('2_Loss/train (labeled_reg)', np.mean(tr_loss[epoch]['l_reg_loss']))
    writer.add_scalar('2_Loss/val (labeled_reg)', np.mean(val_loss[epoch]['l_reg_loss']))
    writer.add_scalar('3_Loss/train (labeled_class)', np.mean(tr_loss[epoch]['l_class_loss']))
    writer.add_scalar('3_Loss/val (labeled_class)', np.mean(val_loss[epoch]['l_class_loss']))
    writer.add_scalar('4_Loss/train (unlabeled_class)', np.mean(tr_loss[epoch]['u_class_loss']))
    
    """ Scores """
    writer.add_scalar("5_MAE/Train", np.mean(tr_scores[epoch]['mae']))
    writer.add_scalar("5_MAE/Val", np.mean(val_scores[epoch]['mae']))
    writer.add_scalar("6_RMSE/Train", np.mean(tr_scores[epoch]['rmse']))
    writer.add_scalar("6_RMSE/Val", np.mean(val_scores[epoch]['rmse']))
    writer.add_scalar("7_MAE/Train", np.mean(tr_scores[epoch]['r2']))
    writer.add_scalar("7_MAE/Val", np.mean(val_scores[epoch]['r2']))

"""
Takes model and validation set. Calculates metrics on validation set. 
Runs for each epoch. 
"""
def _validate(model, val_loader, metrics, args, loss_fn_reg, loss_fn_class, val_loss, val_scores, epoch):
    model.eval()
    
    with torch.no_grad():
        for batch_id, val_data in enumerate(val_loader):
            v_patches, v_date_types, v_reg_vals, (v_img_idxs, v_pxs, v_pys) = val_data
            v_patches, v_date_types, v_reg_vals = v_patches.to(args['device']), v_date_types.to(args['device']), v_reg_vals.to(args['device'])
            
            """" Calculate loss """ 
            v_reg_preds, v_class_preds = model(v_patches)
            reg_loss_val = loss_fn_reg(input=v_reg_preds, target=v_reg_vals)
            reg_loss_class = loss_fn_class(input=v_class_preds, target=v_date_types)
            loss = reg_loss_val + reg_loss_class
            
            """ Keep loss """
            val_loss[epoch]['l_reg_loss'].append(reg_loss_val.item())
            val_loss[epoch]['l_class_loss'].append(reg_loss_class.item())
            val_loss[epoch]['total'].append(loss.item())
            
            """ Calculate & keep score """
            score = metrics.eval_reg_batch_metrics(preds=v_reg_preds, targets=v_reg_vals)
            val_scores[epoch]['r2'].append(score['r2'])
            val_scores[epoch]['mae'].append(score['mae'])
            val_scores[epoch]['rmse'].append(score['rmse'])
            
"""
Trains model with labeled data only. 
"""
def _train_labeled_only(model, train_loader, args, metrics, loss_fn_reg, loss_fn_class, fold, run_name, val_loader=None):
    model.apply(weight_reset)                                                   # Or save weights of the model first & load them.
    model.train()
    optimizer = RMSprop(params=model.parameters(), lr=args['lr'])               # EA uses RMSprop with lr=0.0001, I can try SGD or Adam as in [1, 2] or [3].
    tr_loss = [{'l_reg_loss': [], 'l_class_loss' : []} for e in range(args['max_epoch'])]
    val_loss = [{'l_reg_loss': [], 'l_class_loss' : []} for e in range(args['max_epoch'])]
    best_val_loss = float('inf')
    fold_path = osp.join(C.MODEL_DIR_PATH, run_name, 'fold_' + str(fold))
    os.mkdir(fold_path)
    
    for e in range(args['max_epoch']):
        len_loader = len(train_loader)
        labeled_iter = iter(train_loader)
        
        batch_id = 0
        while batch_id < len_loader:
            optimizer.zero_grad()
            
            """ Labeled data only """
            labeled_data = next(labeled_iter)
            l_patches, l_date_types, l_reg_vals, (l_img_idxs, l_pxs, l_pys) = labeled_data
            l_patches, l_reg_vals, l_date_types = l_patches.to(args['device']), l_reg_vals.to(args['device']), l_date_types.to(args['device'])
            
            l_reg_preds, l_class_preds = model(l_patches)
            reg_loss_labeled = loss_fn_reg(input=l_reg_preds, target=l_reg_vals)
            class_loss_labeled = loss_fn_class(input=l_class_preds, target=l_date_types)
            
            """ Calculate loss """
            loss = reg_loss_labeled + class_loss_labeled
            loss.backward()
            optimizer.step()
            print('Losses, Labeled, reg: {}, class: {}'.format(reg_loss_labeled.item(), class_loss_labeled.item()))
            
            """ Keep losses for plotting """
            tr_loss[e]['l_reg_loss'].append(reg_loss_labeled.item())
            tr_loss[e]['l_class_loss'].append(class_loss_labeled.item())
            tr_loss[e]['total'].append(loss.item())
            batch_id += 1
        
        print(get_msg(tr_loss, e))                                              # Print epoch loss on train set. 
        
        """ Validation """
        if val_loader is not None:
            _validate(model, val_loader, metrics, args, loss_fn_reg, loss_fn_class, val_loss, e)
            if np.mean(val_loss['total'] < best_val_loss):
                best_val_loss = np.mean(val_loss['total'])
                torch.save(model.state_dict(), fold_path + 'best_val_loss.pth')
            print(get_msg(val_loss, e))                                         # Print epoch loss on validation set. 
        
    torch.save(model.state_dict(), fold_path + 'model_last_epoch.pth')          # Save model of last epoch.
            
"""
Trains model with labeled and unlabeled data. 
"""
def _train(model, train_loader, unlabeled_loader, args, metrics, loss_fn_reg, loss_fn_class, fold, run_name, writer, val_loader=None):
    model.apply(weight_reset)                                                   # Or save weights of the model first & load them.
    model.train()
    optimizer = RMSprop(params=model.parameters(), lr=args['lr'])               # EA uses RMSprop with lr=0.0001, I can try SGD or Adam as in [1, 2] or [3].
    tr_loss = [{'l_reg_loss': [], 'l_class_loss' : [], 'u_class_loss' : [], 'total' : []} for e in range(args['max_epoch'])]
    val_loss = [{'l_reg_loss': [], 'l_class_loss' : [], 'total' : []} for e in range(args['max_epoch'])]
    tr_scores = [{'r2' : [], 'mae' : [], 'rmse' : []} for e in range(args['max_epoch'])]
    val_scores = [{'r2' : [], 'mae' : [], 'rmse' : []} for e in range(args['max_epoch'])]
    best_val_loss = float('inf')
    model_dir_path = osp.join(C.MODEL_DIR_PATH, run_name, 'fold_' + str(fold))
    os.mkdir(model_dir_path)
    
    for e in range(args['max_epoch']):
        len_loader = min(len(train_loader), len(unlabeled_loader))              # Update unlabeled batch size to use all its samples. 
        labeled_iter = iter(train_loader)
        unlabeled_iter = iter(unlabeled_loader)
    
        """ Train """
        batch_id = 0
        # while batch_id < len_loader:
        optimizer.zero_grad()
        
        """ Labeled data """
        labeled_data = next(labeled_iter)
        l_patches, l_date_types, l_reg_vals, (l_img_idxs, l_pxs, l_pys) = labeled_data
        l_patches, l_reg_vals, l_date_types = l_patches.to(args['device']), l_reg_vals.to(args['device']), l_date_types.to(args['device'])
        
        """ Prediction on labeled data """
        l_reg_preds, l_class_preds = model(l_patches)
        reg_loss_labeled = loss_fn_reg(input=l_reg_preds, target=l_reg_vals)
        class_loss_labeled = loss_fn_class(input=l_class_preds, target=l_date_types)
        
        """ Unlabeled data """
        unlabeled_data = next(unlabeled_iter)
        u_patches, u_date_types, _, (u_img_idxs, u_pxs, u_pys) = unlabeled_data
        u_patches, u_date_types = u_patches.to(args['device']), u_date_types.to(args['device'])
        
        """ Prediction on unlabeled data """
        _, u_class_preds = model(u_patches)
        class_loss_unlabeled = loss_fn_class(input=u_class_preds, target=u_date_types)

        """ Calculate loss """
        loss = reg_loss_labeled + class_loss_labeled + class_loss_unlabeled
        loss.backward()
        optimizer.step()
        
        """ Keep losses for plotting """
        tr_loss[e]['l_reg_loss'].append(reg_loss_labeled.item())
        tr_loss[e]['l_class_loss'].append(class_loss_labeled.item())
        tr_loss[e]['u_class_loss'].append(class_loss_unlabeled.item())
        tr_loss[e]['total'].append(loss.item())
        
        """ Keep scores for plotting """
        score = metrics.eval_reg_batch_metrics(preds=l_reg_preds, targets=l_reg_vals)
        tr_scores[e]['r2'].append(score['r2'])
        tr_scores[e]['mae'].append(score['mae'])
        tr_scores[e]['rmse'].append(score['rmse'])
        batch_id += 1
            
        print(get_msg(tr_loss, tr_scores, e, dataset='train'))                  # Print train set epoch loss & score. 
            
        """ Validation """
        if val_loader is not None:
            _validate(model, val_loader, metrics, args, loss_fn_reg, loss_fn_class, val_loss, val_scores, e)
            if np.mean(val_loss[e]['total']) < best_val_loss:
                best_val_loss = np.mean(val_loss[e]['total'])
                torch.save(model.state_dict(), model_dir_path + 'best_val_loss.pth')
            # print(get_msg(val_loss, val_scores, e, dataset='val'))              # Print validation set epoch loss & score.
            
        """ Plot loss & scores """
        plot(writer=writer, tr_loss=tr_loss, val_loss=val_loss, tr_scores=tr_scores, val_scores=val_scores, epoch=e)
        
    torch.save(model.state_dict(), model_dir_path + 'model_last_epoch.pth')     # Save model of last epoch.
    
    
    # X Re-init model as in https://discuss.pytorch.org/t/reinitializing-the-weights-after-each-cross-validation-fold/11034
    # X Create a new optimizer. 
    # Normalize data ? 
    # X One folder per model, can be 'fold_1' or fold_1_val=13'. 
    # Use all unlabeled samples. 
    # X Save each fold's model to its folder under 'models\{run_name}'.
    # X Save train result to metrics. 
    # Test'te sadece regresyon sonucunu al, DAN oyle yapiyor. 

"""
Takes a labeled dataset, a train function and arguments. 
Creates dataset's folds and applies train function.
"""
def train_on_folds(model, dataset, unlabeled_dataset, train_fn, loss_fn_class, loss_fn_reg, args):
    unlabeled_loader = DataLoader(unlabeled_dataset, **args['unlabeled'])
    metrics = Metrics(num_folds=args['num_folds'], device=args['device'].type)
    indices = [*range(len(dataset))]                                            # Sample indices
    run_name = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    os.mkdir(osp.join(C.MODEL_DIR_PATH, run_name))                              # Create run folder.

    """ Train & test with cross-validation """
    if args['num_folds'] is not None:
        kf = KFold(n_splits=args['num_folds'], shuffle=True, random_state=args['seed'])
        for fold, (tr_index, test_index) in enumerate(kf.split(indices)):
            np.random.shuffle(tr_index)                                         # kfold does not shuffle samples in splits.     
            val_loader = None
            if args['create_val']:                                              # Create validation set
                val_len = len(test_index)
                tr_index, val_index = tr_index[:-val_len], tr_index[-val_len:]
                val_set = Subset(dataset, indices=val_index)
                val_loader = DataLoader(val_set, **args['val'])
            tr_set = Subset(dataset, indices=tr_index)
            tr_loader = DataLoader(tr_set, **args['tr'])
            writer = SummaryWriter(osp.join('runs', run_name, 'fold_{}'.format(fold)))
            
            # patches_mean, patches_std, regs_mean, regs_std = _calc_mean_std(train_loader=tr_loader, args=args['tr'])
            
            """ Train """
            train_fn(model=model, train_loader=tr_loader, val_loader=val_loader, args=args, metrics=metrics, 
                     unlabeled_loader=unlabeled_loader, loss_fn_reg=loss_fn_reg, loss_fn_class=loss_fn_class, 
                     fold=fold, run_name=run_name, writer=writer)
            
            """ Test """
            
            test_set = Subset(dataset, indices=test_index)
            # # test_loader = DataLoader(test_set, **args['test'])
            print('lens, tr: {}, val: {}, test: {}'.format(len(tr_set), len(val_set), len(test_set)))
            writer.close()
            
    # Train and test without cross-validation
    else:
        """ Create train, val and test sets """
        np.random.shuffle(indices)
        len_test = int(len(indices) * args['test_per'])
        tr_index = indices[0:-2*len_test]
        val_index = indices[-2*len_test:-len_test]
        test_index = indices[-len_test:]
        tr_set, val_set, test_set = Subset(dataset, tr_index), Subset(dataset, val_index), Subset(dataset, test_index)
        
        """ Normalize patches on all datasets """
        if args['patch_norm']:
            patches_mean, patches_std, _, _ = calc_mean_std(DataLoader(tr_set, **args['tr']))   # Calculate mean and std of train set. 
            dataset.set_mean_std(means=patches_mean, stds=patches_std)                          # Set train's mean and std as dataset's. 
        
        tr_loader = DataLoader(tr_set, **args['tr'])
        val_loader = DataLoader(val_set, **args['val'])
        writer = SummaryWriter('runs/' + run_name)
        
        """ Train """
        train_fn(model=model, train_loader=tr_loader, unlabeled_loader=unlabeled_loader, val_loader=val_loader, 
                 args=args, metrics=metrics, loss_fn_reg=loss_fn_reg, loss_fn_class=loss_fn_class, fold=1,
                 run_name=run_name, writer=writer)
        
        """ Test """
        
        writer.close()
        

        
if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     # Use GPU if available
    args = {'num_folds': None,
            'max_epoch': 5,
            'device': device,
            'seed': 42,
            'create_val': True,                                                 # Creates validation set
            'test_per': 0.1,
            'lr': 0.0001,                                                       # From EA's model, default is 1e-2.
            'patch_norm': True,                                                 # Normalizes patches
            
            'tr': {'batch_size': C.BATCH_SIZE, 'shuffle': True, 'num_workers': 4},
            'val': {'batch_size': C.BATCH_SIZE, 'shuffle': False, 'num_workers': 4},
            'unlabeled': {'batch_size': C.BATCH_SIZE, 'shuffle': True, 'num_workers': 4}}
    verify_args(args)
    
    """ Create labeled and unlabeled datasets. """
    labeled_set = Lake2dDataset(learning='labeled', date_type='month')
    unlabeled_set = Lake2dDataset(learning='unlabeled', date_type='month')
    
    """ Create model, regression and classification losses  """
    in_channels, num_classes = labeled_set[0][0].shape[0], C.NUM_CLASSES[labeled_set.date_type]
    model = DandadaDAN(in_channels=in_channels, num_classes=num_classes)
    model.to(args['device'])
    loss_fn_reg = torch.nn.MSELoss().to(args['device'])                        # Regression loss function
    loss_fn_class = torch.nn.CrossEntropyLoss().to(args['device'])             # Classification loss function 
    
    """ Train """
    train_on_folds(model=model, dataset=labeled_set, unlabeled_dataset=unlabeled_set, train_fn=_train, 
                   loss_fn_reg=loss_fn_reg, loss_fn_class=loss_fn_class, args=args)

"""
Ideas:
1. Normalize patch values
2. Change normalization to https://www.youtube.com/watch?v=y6IEcEBRZks
3. Change metrics calculation from mean of batches to all samples. 
4. Try with one-batch learning

References
[1]: https://medium.com/@benjamin.phillips22/simple-regression-with-neural-networks-in-pytorch-313f06910379
[2]: https://github.com/VICO-UoE/L2I/blob/master/Regression/train-MT.py
[3]: https://github.com/fungtion/DANN/blob/master/train/main.py
"""