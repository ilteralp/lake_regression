#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:56:33 2021

@author: melike
"""
import torch
from torch.utils.data import Dataset, random_split, Subset, DataLoader
from torch.optim import RMSprop
import numpy as np
import h5py
import os
import os.path as osp
from sklearn.model_selection import KFold
from datetime import datetime
import constants as C
from datasets import Lake2dDataset
from metrics import Metrics

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
"""
def _calc_mean_std(train_loader, args):
    patches_mean, patches_std = 0., 0.
    regs_mean, regs_std = 0., 0.
    num_samples = 0.
    for batch_id, data in enumerate(train_loader):
        patches, monthes, seasons, years, reg_vals, (img_idxs, pxs, pys) = data
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
Returns verbose message
"""
def get_msg(loss, e):
    msg = "Epoch #{}, Losses, Train, total: {:.4f} labeled_reg_loss: {:.4f}".format(
        e, np.mean(loss[e]['total']), np.mean(loss[e]['l_reg_loss']))
    if 'l_class_loss' in loss[e]:
        msg += ", labeled_class_loss: {:.4f}".format(np.mean(loss[e]['l_class_loss']))
        
"""
Takes model and validation set. Calculates metrics on validation set. 
Runs for each epoch. 
"""
def _validate(model, val_loader, metrics, args, loss_fn_reg, loss_fn_class, val_loss, epoch):
    model.eval()
    
    with torch.no_grad:
        for batch_id, val_data in enumerate(val_loader):
            v_patches, v_date_types, v_reg_vals, (v_img_idxs, v_pxs, v_pys) = val_data
            v_patches, v_date_types, v_reg_vals = v_patches.to(args['device']), v_date_types.to(args['device']), v_reg_vals.to(args['device'])
            
            """" Calculate loss """ 
            v_reg_preds, v_class_preds = model(v_patches)
            reg_loss_val = loss_fn_reg(input=v_reg_preds, target=v_reg_vals)
            reg_loss_class = loss_fn_class(input=v_class_preds, target=v_date_types)
            loss = reg_loss_val + reg_loss_class
            
            val_loss[epoch]['l_reg_loss'].append(reg_loss_val.item())
            val_loss[epoch]['l_class_loss'].append(reg_loss_class.item())
            val_loss[epoch]['total'].append(loss.item())
            
"""
Trains model with labeled data only. 
"""
def _train_labeled_only(model, train_loader, args, metrics, loss_fn_reg, fold, run_name, val_loader=None):
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
def _train(model, train_loader, unlabeled_loader, args, metrics, loss_fn_reg, loss_fn_class, fold, run_name, val_loader=None):
    model.apply(weight_reset)                                                   # Or save weights of the model first & load them.
    model.train()
    optimizer = RMSprop(params=model.parameters(), lr=args['lr'])               # EA uses RMSprop with lr=0.0001, I can try SGD or Adam as in [1, 2] or [3].
    tr_loss = [{'l_reg_loss': [], 'l_class_loss' : [], 'u_class_loss' : []} for e in range(args['max_epoch'])]
    val_loss = [{'l_reg_loss': [], 'l_class_loss' : []} for e in range(args['max_epoch'])]
    best_val_loss = float('inf')
    fold_path = osp.join(C.MODEL_DIR_PATH, run_name, 'fold_' + str(fold))
    os.mkdir(fold_path)
    
    for e in range(args['max_epoch']):
        len_loader = min(len(train_loader), len(unlabeled_loader))              # Update unlabeled batch size to use all its samples. 
        labeled_iter = iter(train_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        batch_id = 0
        while batch_id < len_loader:
            optimizer.zero_grad()
            
            """ Labeled data """
            labeled_data = next(labeled_iter)
            l_patches, l_date_types, l_reg_vals, (l_img_idxs, l_pxs, l_pys) = labeled_data
            l_patches, l_reg_vals, l_date_types = l_patches.to(args['device']), l_reg_vals.to(args['device']), l_date_types.to(args['device'])
            
            l_reg_preds, l_class_preds = model(l_patches)
            reg_loss_labeled = loss_fn_reg(input=l_reg_preds, target=l_reg_vals)
            class_loss_labeled = loss_fn_class(input=l_class_preds, target=l_date_types)
            
            """ Unlabeled data """
            unlabeled_data = next(unlabeled_iter)
            u_patches, u_date_types, _, (u_img_idxs, u_pxs, u_pys) = unlabeled_data
            u_patches, u_date_types = u_patches.to(args['device']), u_date_types.to(args['device'])
            
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
            batch_id += 1
            
        print(get_msg(tr_loss, e))                                              # Print train set epoch loss
            
        """ Validation """
        if val_loader is not None:
            _validate(model, val_loader, metrics, args, loss_fn_reg, loss_fn_class, val_loss, e)
            if np.mean(val_loss['total'] < best_val_loss):
                best_val_loss = np.mean(val_loss['total'])
                torch.save(model.state_dict(), fold_path + 'best_val_loss.pth')
            print(get_msg(val_loss, e))                                         # Print validation set epoch loss
        
    torch.save(model.state_dict(), fold_path + 'model_last_epoch.pth')          # Save model of last epoch.
    
    # X Re-init model as in https://discuss.pytorch.org/t/reinitializing-the-weights-after-each-cross-validation-fold/11034
    # X Create a new optimizer. 
    # Normalize data ? 
    # One folder per model, can be 'fold_1' or fold_1_val=13'. 
    # Use all unlabeled samples. 
    # Save each model to its folder under 'models\'.
    # Save train result to metrics. 
    # Test'te sadece regresyon sonucunu al, DAN oyle yapiyor. 

"""
Takes a labeled dataset, a train function and arguments. 
Creates dataset's folds and applies train function.
"""
def train_on_folds(model, dataset, unlabeled_dataset, train_fn, loss_fn_class, loss_fn_reg, args):
    kf = KFold(n_splits=args['num_folds'], shuffle=False, random_state=args['seed'])
    unlabeled_loader = DataLoader(unlabeled_dataset, **args['unlabeled'])
    metrics = Metrics(num_folds=args['num_folds'])
    indices = [*range(len(dataset))]                                            # Sample indices
    run_name = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    os.mkdir(osp.join(C.MODEL_DIR_PATH, run_name))                              # Create run folder.
    
    for fold, (tr_index, test_index) in enumerate(kf.split(indices)):
        tr_index = np.random.shuffle(tr_index)                                  # kfold does not shuffle samples in splits.     
        val_loader = None
        if args['create_val']:                                                  # Create validation set
            val_len = len(test_index)
            tr_index, val_index = tr_index[:-val_len], tr_index[-val_len:]
            val_set = Subset(dataset, indices=val_index)
            val_loader = DataLoader(val_set, **args['val'])
        tr_set = Subset(dataset, indices=tr_index)
        tr_loader = DataLoader(tr_set, **args['tr'])
        
        """ Train """
        train_fn(model=model, train_loader=tr_loader, val_loader=val_loader, args=args, metrics=metrics, 
                 unlabeled_loader=unlabeled_loader, loss_fn_reg=loss_fn_reg, loss_fn_class=loss_fn_class, 
                 fold=fold, run_name=run_name)
        
        """ Test """
        
        test_set = Subset(dataset, indices=test_index)
        # # test_loader = DataLoader(test_set, **args['test'])
        print('lens, tr: {}, val: {}, test: {}'.format(len(tr_set), len(val_set), len(test_set)))
            
        
if __name__ == "__main__":
    labeled_set = Lake2dDataset(learning='labeled')
    unlabeled_set = Lake2dDataset(learning='unlabeled')
    # sup_tr_set, sup_val_set, sup_test_set = split_dataset(labeled_set, test_ratio=0.1, val_ratio=0.1)
    # print('lens, tr: {}, val: {}, test: {}'.format(len(sup_tr_set), len(sup_val_set), len(sup_test_set)))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     # Use GPU if available
    args = {'num_folds': None,
            'max_epoch': None,
            'device': device,
            'metrics': None,
            'seed':  42,
            'create_val': True,                                                 # Creates validation set
            'lr': 0.0001,                                                       # From EA's model, default is 1e-2. 
            
            'tr': {'batch_size': C.BATCH_SIZE, 'shuffle': True, 'num_workers': 4},
            'val': {'batch_size': C.BATCH_SIZE, 'shuffle': False, 'num_workers': 4},
            'unlabeled': {'batch_size': C.BATCH_SIZE, 'shuffle': True, 'num_workers': 4}}
    
    model = None
    model.to(args['device'])
    loss_fn_reg = torch.nn.MSELoss().to(args['device'])                         # Regression loss function
    loss_fn_class = torch.nn.CrossEntropyLoss().to(args['device'])             # Classification loss function 

    """ Getting normalization values """
    # patches_mean, patches_std, regs_mean, regs_std = _calc_mean_std(train_loader=labeled_loader, args=args['tr'])
    
"""
References
[1]: https://medium.com/@benjamin.phillips22/simple-regression-with-neural-networks-in-pytorch-313f06910379
[2]: https://github.com/VICO-UoE/L2I/blob/master/Regression/train-MT.py
[3]: https://github.com/fungtion/DANN/blob/master/train/main.py
"""