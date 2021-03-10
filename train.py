#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:56:33 2021

@author: melike
"""
import torch
from torch.utils.data import Dataset, random_split, Subset, DataLoader
from torch.optim import RMSprop
import torch.nn as nn
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
from models import DandadaDAN, EANet

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
    # regs_mean, regs_std = 0., 0.
    # regs_sum, regs_squared_sum, num_batches = 0, 0, 0
    num_samples = 0.
    # max_r, min_r = 0.0, float('inf')
    for data in train_loader:
        patches, date_type, reg_vals, (img_idxs, pxs, pys) = data
        batch_samples = patches.size(0)
        patches = patches.view(batch_samples, patches.size(1), -1)              # (num_samples, 12, 3, 3) -> (num_samples, 12, 9)
        patches_mean += patches.mean(2).sum(0)                                  # 12 means and 12 stds
        patches_std += patches.std(2).sum(0)
        # regs_mean += reg_vals.mean()
        # regs_std += reg_vals.std()
        # regs_sum += torch.mean(reg_vals)
        # regs_squared_sum += torch.mean(reg_vals ** 2)
        num_samples += batch_samples
        # num_batches += 1
        # if max_r < torch.max(reg_vals):
        #     max_r = torch.max(reg_vals)
        # if min_r > torch.min(reg_vals):
        #     min_r = torch.min(reg_vals)
    patches_mean /= num_samples
    patches_std /= num_samples
    # # regs_mean /= (batch_id + 1)
    # # regs_std /= (batch_id + 1)
    # regs_mean = regs_sum / num_batches
    # regs_std = (regs_squared_sum / num_batches - regs_mean ** 2) ** 0.5
    return patches_mean, patches_std

"""
Returns min and max of given train set regression values.
"""
def get_reg_min_max(tr_reg_vals):
    reg_min = np.min(tr_reg_vals)
    reg_max = np.max(tr_reg_vals)
    if reg_min == reg_max:
        raise Exception('Given constant regression values with train set!')
    else:
        return reg_min, reg_max

"""
Takes a model, resets layer weights
https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/7
"""
def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()
        
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):                    # glorot_uniform in Keras is xavier_uniform_  
        nn.init.xavier_uniform_(m.weight)
        
def reset_model(m, args):
    if args['model'] == 'eanet':
        return m.apply(weights_init)
    
    elif args['model'] == 'dandadadan':
        return m.apply(weight_reset)
        
"""
Returns verbose message with loss and score.
"""
def get_msg(loss, score, e, dataset):
    sum_str = '(R+C_L+C_U)' if 'u_class_loss' in loss[e] else '(R+C)'
    start_str = '' if dataset == 'test' else 'Epoch #{}, '.format(e) 
    msg = "{}Losses {}: {}, {:.2f} = {:.2f}".format(
        start_str, sum_str, dataset, np.mean(loss[e]['total']), np.mean(loss[e]['l_reg_loss']))
    if 'l_class_loss' in loss[e]:
        msg += " + {:.2f}".format(np.mean(loss[e]['l_class_loss']))
    if 'u_class_loss' in loss[e]:
        msg += " + {:.2f}".format(np.mean(loss[e]['u_class_loss']))
    if dataset == 'test':
        msg += "\t Scores, MAE: {:.2f}, R2: {:.2f}, RMSE: {:.2f}".format(
            np.mean(score[e]['mae']), np.mean(score[e]['r2']), np.mean(score[e]['rmse']))
    return msg

"""
Checks given args
"""
def verify_args(args):
    if args['num_folds'] is not None and args['num_folds'] < 2:
        raise Exception('Number of folds should be at least 2')
    if args['test_per'] >= 0.5:
        raise Exception('Test percent should be less than 0.5 since validation set has the same length with it.')
    if args['pred_type'] not in ['reg', 'class', 'reg+class']:
        raise Exception('Expected prediction type to be one of [\'reg\', \'reg+class\']')
    if args['model'] not in ['dandadadan', 'eanet']:
        raise Exception('Model can be one of [\'dandadadan\', \'eanet\']')
    if args['use_unlabeled_samples'] and args['pred_type'] == 'reg':
        raise Exception('Unlabeled samples cannot be used with regression. They can only be used for reg+class.')

"""
Plots loss and scores to Tensorboard
"""
def plot(writer, tr_loss, val_loss, tr_scores, val_scores, e):
    
    """ Losses """
    writer.add_scalar('1_Loss/train (total)', np.mean(tr_loss[e]['total']), e)
    writer.add_scalar('1_Loss/val (total)', np.mean(val_loss[e]['total']), e)
    writer.add_scalar('2_Loss/train (labeled_reg)', np.mean(tr_loss[e]['l_reg_loss']), e)
    writer.add_scalar('2_Loss/val (labeled_reg)', np.mean(val_loss[e]['l_reg_loss']), e)
    if 'l_class_loss' in tr_loss[e]:
        writer.add_scalar('3_Loss/train (labeled_class)', np.mean(tr_loss[e]['l_class_loss']), e)
    if 'l_class_loss' in val_loss[e]:
        writer.add_scalar('3_Loss/val (labeled_class)', np.mean(val_loss[e]['l_class_loss']), e)
    if 'u_class_loss' in tr_loss[e]:
        writer.add_scalar('4_Loss/train (unlabeled_class)', np.mean(tr_loss[e]['u_class_loss']), e)
    
    """ Scores """
    writer.add_scalar("5_MAE/Train", np.mean(tr_scores[e]['mae']), e)
    writer.add_scalar("5_MAE/Val", np.mean(val_scores[e]['mae']), e)
    writer.add_scalar("6_RMSE/Train", np.mean(tr_scores[e]['rmse']), e)
    writer.add_scalar("6_RMSE/Val", np.mean(val_scores[e]['rmse']), e)
    writer.add_scalar("7_R2/Train", np.mean(tr_scores[e]['r2']), e)
    writer.add_scalar("7_R2/Val", np.mean(val_scores[e]['r2']), e)
    
"""
Loads the model with given name and prints its results. 
"""
def _test(test_set, model_name, in_channels, num_classes, metrics, args, fold, run_name):
    print('model: {} with fold: {}'.format(model_name, str(fold)))
    test_model = DandadaDAN(in_channels=in_channels, num_classes=num_classes).to(args['device'])
    model_dir_path = osp.join(C.MODEL_DIR_PATH, run_name, 'fold_' + str(fold))
    test_model.load_state_dict(torch.load(osp.join(model_dir_path, model_name)))
    test_loader = DataLoader(test_set, **args['test'])
    if args['pred_type'] == 'reg+class':
        test_loss = [{'l_reg_loss': [], 'l_class_loss' : [], 'total' : []}]
    elif args['pred_type'] == 'reg':
        test_loss = [{'l_reg_loss': [], 'total' : []}]
    test_scores = [{'r2' : [], 'mae' : [], 'rmse' : []}]
    _validate(model=test_model, val_loader=test_loader, metrics=metrics, args=args,
              val_loss=test_loss, val_scores=test_scores, epoch=0)
    
    """ Save result to file """
    msg = get_msg(test_loss, test_scores, e=0, dataset='test')
    with open(osp.join(model_dir_path, model_name + '.res'), 'w') as f:
        f.write(msg)
    print(msg)
        
    
"""
Takes model and validation set. Calculates metrics on validation set. 
Runs for each epoch. 
"""
def _validate(model, val_loader, metrics, args, val_loss, val_scores, epoch):
    model.eval()
    
    with torch.no_grad():
        for batch_id, val_data in enumerate(val_loader):
            v_patches, v_date_types, v_reg_vals, (v_img_idxs, v_pxs, v_pys) = val_data
            v_patches, v_date_types, v_reg_vals = v_patches.to(args['device']), v_date_types.to(args['device']), v_reg_vals.to(args['device'])
            
            # """" Calculate loss """ 
            # v_reg_preds, v_class_preds = model(v_patches)
            # reg_loss_val = args['loss_fn_reg'](input=v_reg_preds, target=v_reg_vals)
            # reg_loss_class = args['loss_fn_class'](input=v_class_preds, target=v_date_types)
            # loss = reg_loss_val + reg_loss_class
            
            loss = calc_loss(model=model, patches=v_patches, args=args, loss_arr=val_loss, score_arr=val_scores, 
                             e=epoch, target_regs=v_reg_vals, target_labels=v_date_types, metrics=metrics)
            
            """ Keep loss """
            val_loss[epoch]['total'].append(loss.item())      
            
"""
Updates scores. 
"""      
def add_scores(preds, targets, score_arr, metrics, e):
    score = metrics.eval_reg_batch_metrics(preds=preds, targets=targets)
    score_arr[e]['r2'].append(score['r2'])
    score_arr[e]['mae'].append(score['mae'])
    score_arr[e]['rmse'].append(score['rmse'])

"""
Calculates loss depending on prediction type
"""
def calc_loss(model, patches, args, loss_arr, score_arr, e, target_regs, metrics, target_labels=None):
    if args['pred_type'] == 'reg':    
        reg_preds = model(patches)
        if args['model'] == 'dandadadan':
            reg_preds, _ = reg_preds
        reg_loss = args['loss_fn_reg'](input=reg_preds, target=target_regs)
        loss_arr[e]['l_reg_loss'].append(reg_loss.item())
        add_scores(preds=reg_preds, targets=target_regs, e=e, score_arr=score_arr, metrics=metrics)
        return reg_loss

    elif args['pred_type'] == 'reg+class':
        reg_preds, class_preds = model(patches)
        reg_loss = args['loss_fn_reg'](input=reg_preds, target=target_regs)
        class_loss = args['loss_fn_class'](input=class_preds, target=target_labels)
        loss_arr[e]['l_reg_loss'].append(reg_loss.item())
        loss_arr[e]['l_class_loss'].append(class_loss.item())
        add_scores(preds=reg_preds, targets=target_regs, e=e, score_arr=score_arr, metrics=metrics)
        return reg_loss + class_loss

"""
Creates arrays of losses and scores with given args. 
"""
def create_losses_scores(args):
    if args['pred_type'] == 'reg':
        losses = [{'l_reg_loss': [], 'total' : []} for e in range(args['max_epoch'])]
    elif args['pred_type'] == 'reg+class':
        if args['use_unlabeled_samples']:
           losses = [{'l_reg_loss': [], 'l_class_loss' : [], 'u_class_loss' : [], 'total' : []} for e in range(args['max_epoch'])]
        else:
            losses = [{'l_reg_loss': [], 'l_class_loss' : [], 'total' : []} for e in range(args['max_epoch'])]
    scores = [{'r2' : [], 'mae' : [], 'rmse' : []} for e in range(args['max_epoch'])]
    return losses, scores


"""
Trains model with labeled and unlabeled data. 
"""
def _train(model, train_loader, unlabeled_loader, args, metrics, fold, run_name, writer, val_loader=None):
    model.apply(weight_reset)                                                   # Or save weights of the model first & load them.
    optimizer = RMSprop(params=model.parameters(), lr=args['lr'])               # EA uses RMSprop with lr=0.0001, I can try SGD or Adam as in [1, 2] or [3].
    # tr_loss = [{'l_reg_loss': [], 'l_class_loss' : [], 'u_class_loss' : [], 'total' : []} for e in range(args['max_epoch'])]
    # val_loss = [{'l_reg_loss': [], 'l_class_loss' : [], 'total' : []} for e in range(args['max_epoch'])]
    # tr_scores = [{'r2' : [], 'mae' : [], 'rmse' : []} for e in range(args['max_epoch'])]
    # val_scores = [{'r2' : [], 'mae' : [], 'rmse' : []} for e in range(args['max_epoch'])]
    tr_loss, tr_scores = create_losses_scores(args)
    val_loss, val_scores = create_losses_scores(args)
    best_val_loss = float('inf')
    best_val_r2_score = -float('inf')
    model_dir_path = osp.join(C.MODEL_DIR_PATH, run_name, 'fold_' + str(fold))
    os.mkdir(model_dir_path)
    # print('len loaders, train: {}, unlabeled: {}'.format(len(train_loader), len(unlabeled_loader)))
    
    for e in range(args['max_epoch']):
        model.train()
        if args['use_unlabeled_samples']:
            len_loader = min(len(train_loader), len(unlabeled_loader))              # Update unlabeled batch size to use all its samples. 
            unlabeled_iter = iter(unlabeled_loader)
        else:
            len_loader = len(train_loader)
        labeled_iter = iter(train_loader)
    
        """ Train """
        batch_id = 0
        while batch_id < len_loader:
            optimizer.zero_grad()
            
            """ Labeled data """
            labeled_data = next(labeled_iter)
            l_patches, l_date_types, l_reg_vals, (l_img_idxs, l_pxs, l_pys) = labeled_data
            l_patches, l_reg_vals, l_date_types = l_patches.to(args['device']), l_reg_vals.to(args['device']), l_date_types.to(args['device'])
            
            # """ Prediction on labeled data """
            # l_reg_preds, l_class_preds = model(l_patches)
            # reg_loss_labeled = args['loss_fn_reg'](input=l_reg_preds, target=l_reg_vals)
            # class_loss_labeled = args['loss_fn_class'](input=l_class_preds, target=l_date_types)
            loss = calc_loss(model=model, patches=l_patches, args=args, loss_arr=tr_loss, score_arr=tr_scores, 
                                     e=e, target_regs=l_reg_vals, target_labels=l_date_types, metrics=metrics)
            
            # """ Unlabeled data """
            if args['use_unlabeled_samples']:
                unlabeled_data = next(unlabeled_iter)
                u_patches, u_date_types, _, (u_img_idxs, u_pxs, u_pys) = unlabeled_data
                u_patches, u_date_types = u_patches.to(args['device']), u_date_types.to(args['device'])
                
                """ Prediction on unlabeled data """
                _, u_class_preds = model(u_patches)
                class_loss_unlabeled = args['loss_fn_class'](input=u_class_preds, target=u_date_types)
                tr_loss[e]['u_class_loss'].append(class_loss_unlabeled.item())
                loss = loss + class_loss_unlabeled
    
            """ Calculate loss """
            # loss = reg_loss_labeled + class_loss_labeled + class_loss_unlabeled
            loss.backward()
            optimizer.step()
            
            """ Keep losses for plotting """
            tr_loss[e]['total'].append(loss.item())
            batch_id += 1
        
        if e % 10 == 0:
            print(get_msg(tr_loss, tr_scores, e, dataset='train'))                  # Print train set loss & score for each **epoch**. 
            
        """ Validation """
        if val_loader is not None:
            _validate(model, val_loader, metrics, args, val_loss, val_scores, e)
            if np.mean(val_loss[e]['total']) < best_val_loss:
                best_val_loss = np.mean(val_loss[e]['total'])
                torch.save(model.state_dict(), osp.join(model_dir_path, 'best_val_loss.pth'))
            if np.mean(val_scores[e]['r2']) > best_val_r2_score:
                best_val_r2_score = np.mean(val_scores[e]['r2'])
                torch.save(model.state_dict(), osp.join(model_dir_path, 'best_val_r2_score.pth'))
            # print(get_msg(val_loss, val_scores, e, dataset='val'))              # Print validation set loss & score for each **epoch**. 
          
        """ Plot loss & scores """
        plot(writer=writer, tr_loss=tr_loss, val_loss=val_loss, tr_scores=tr_scores, val_scores=val_scores, e=e)
        
    torch.save(model.state_dict(), osp.join(model_dir_path, 'model_last_epoch.pth'))     # Save model of last epoch.
    
    
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
def train_on_folds(model, dataset, unlabeled_dataset, train_fn, args):
    unlabeled_loader = DataLoader(unlabeled_dataset, **args['unlabeled']) if args['use_unlabeled_samples'] else None
    metrics = Metrics(num_folds=args['num_folds'], device=args['device'].type)
    indices = [*range(len(dataset))]                                                              # Sample indices
    np.random.shuffle(indices)
    run_name = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    os.mkdir(osp.join(C.MODEL_DIR_PATH, run_name))                                                # Create model_files\<run_name> folder.
    os.mkdir(osp.join(os.getcwd(), 'runs', run_name))                                             # Create runs\<run_name> folder.   
    print('\nRun name: {}'.format(run_name))
    with open(osp.join(C.MODEL_DIR_PATH, run_name, 'args.txt'), 'w') as f:                        # Save args  
        f.write(str(args))
    with open(osp.join(os.getcwd(), 'runs', run_name, 'args.txt'), 'w') as f:
        f.write(str(args))

    """ Train & test with cross-validation """
    if args['num_folds'] is not None:
        kf = KFold(n_splits=args['num_folds'], shuffle=True, random_state=args['seed'])
        for fold, (tr_index, test_index) in enumerate(kf.split(indices)):
            print('\nFold#{}'.format(fold))
            np.random.shuffle(tr_index)                                                           # kfold does not shuffle samples in splits.     
            val_loader = None
            
            """ Create train and validation set """ 
            if args['create_val']:                                                                # Create validation set
                val_len = len(test_index)
                tr_index, val_index = tr_index[:-val_len], tr_index[-val_len:]
                val_set = Subset(dataset, indices=val_index)
            tr_set = Subset(dataset, indices=tr_index)

            """ Normalize patches on all datasets """
            if args['patch_norm']:
                patches_mean, patches_std = calc_mean_std(DataLoader(tr_set, **args['tr']))       # Calculate patch mean and std of each channel on train set. 
                dataset.set_patch_mean_std(means=patches_mean, stds=patches_std)                  # Set train set's patch mean and std as dataset's. Updated with each new train set. 
                
            """ Normalize regression value on all datasets """
            if args['reg_norm']:
                # _, _, reg_mean, reg_std = calc_mean_std(DataLoader(tr_set, **args['tr']))         # Calculate mean and std regression values on train set. 
                # dataset.set_reg_mean_std(reg_mean=reg_mean, reg_std=reg_std)                      # Set train set's regression mean and std as dataset's. Updated with each new train set. 
                reg_min, reg_max = get_reg_min_max(dataset.reg_vals[tr_index])
                dataset.set_reg_min_max(reg_min=reg_min, reg_max=reg_max)
                
            """ Load data """
            tr_loader = DataLoader(tr_set, **args['tr'])                                          # Loaders have to be after normalization. 
            if args['create_val']:
                val_loader = DataLoader(val_set, **args['val'])
            writer = SummaryWriter(osp.join('runs', run_name, 'fold_{}'.format(fold)))
            
            """ Train & Validation """
            print('\nTrain & Validation')
            train_fn(model=model, train_loader=tr_loader, val_loader=val_loader, args=args, metrics=metrics, 
                     unlabeled_loader=unlabeled_loader, fold=fold, run_name=run_name, writer=writer)
            writer.close()

            """ Test """
            print('\nTest')
            test_set = Subset(dataset, indices=test_index)
            in_channels, num_classes = dataset[0][0].shape[0], C.NUM_CLASSES[dataset.date_type]
            for model_name in ['best_val_loss.pth', 'model_last_epoch.pth', 'best_val_r2_score.pth']:
                _test(test_set=test_set, model_name=model_name, in_channels=in_channels, num_classes=num_classes, 
                      metrics=metrics, args=args, fold=fold, run_name=run_name)
            
            
            # print('lens, tr: {}, val: {}, test: {}'.format(len(tr_set), len(val_set), len(test_set)))
            print('=' * 72)    
    # Train and test without cross-validation
    else:
        """ Create train, val and test sets """
        len_test = int(len(indices) * args['test_per'])
        tr_index = indices[0:-2*len_test]
        val_index = indices[-2*len_test:-len_test]
        test_index = indices[-len_test:]
        tr_set, val_set, test_set = Subset(dataset, tr_index), Subset(dataset, val_index), Subset(dataset, test_index)
        print('tr: {}, val: {}, test: {}'.format(tr_index[0:3], val_index[0:3], test_index[0:3]))
        
        """ Normalize patches on all datasets """
        if args['patch_norm']:
            patches_mean, patches_std = calc_mean_std(DataLoader(tr_set, **args['tr']))         # Calculate mean and std of train set. 
            dataset.set_patch_mean_std(means=patches_mean, stds=patches_std)                    # Set train's mean and std as dataset's. 
            
        """ Normalize regression value on all datasets """
        if args['reg_norm']:
            reg_min, reg_max = get_reg_min_max(dataset.reg_vals[tr_index])
            dataset.set_reg_min_max(reg_min=reg_min, reg_max=reg_max)
            # reg_mean, reg_std = np.mean(tr_reg_vals), np.std(tr_reg_vals)
            # _, _, reg_mean, reg_std = calc_mean_std(DataLoader(tr_set, **args['tr']))           # Calculate mean and std regression values on train set. 
            # dataset.set_reg_mean_std(reg_mean=reg_mean, reg_std=reg_std)                       # Set train set's regression mean and std as dataset's. Updated with each new train set. 
            # print('tr dataset, reg_mean: {:.2f}, reg_std: {:.2f}'.format(reg_mean, reg_std))
                
        """ Load data """
        tr_loader = DataLoader(tr_set, **args['tr'])
        val_loader = DataLoader(val_set, **args['val'])
        writer = SummaryWriter('runs/' + run_name)
        
        """ Train & Validation """
        print('\nTrain & Validation')
        train_fn(model=model, train_loader=tr_loader, unlabeled_loader=unlabeled_loader, val_loader=val_loader, 
                 args=args, metrics=metrics, fold=1, run_name=run_name, writer=writer)
        
        writer.close()
        
        """ Test """
        print('\nTest')
        test_set = Subset(dataset, indices=test_index)
        in_channels, num_classes = dataset[0][0].shape[0], C.NUM_CLASSES[dataset.date_type]
        for model_name in ['best_val_loss.pth', 'model_last_epoch.pth', 'best_val_r2_score.pth']:
            _test(test_set=test_set, model_name=model_name, in_channels=in_channels, num_classes=num_classes, 
                  metrics=metrics, args=args, fold=1, run_name=run_name)
            
"""
Creates model.
"""
def create_model(dataset, args):
    in_channels = dataset[0][0].shape[0]
    if args['model'] == 'dandadadan':
        num_classes = C.NUM_CLASSES[dataset.date_type]
        model = DandadaDAN(in_channels=in_channels, num_classes=num_classes)
    
    elif args['model'] == 'eanet':
        model = EANet(in_channels=in_channels)
        model.apply(weights_init)
    
    return model.to(args['device'])
        
    
"""
Runs model with given args. 
"""        
def run(args):
    """ Create labeled and unlabeled datasets. """
    labeled_set = Lake2dDataset(learning='labeled', date_type=args['date_type'])
    unlabeled_set = Lake2dDataset(learning='unlabeled', date_type=args['date_type'])
    
    """ Create model, regression and classification losses  """
    model = create_model(dataset=labeled_set, args=args)
    
    loss_fn_reg = torch.nn.MSELoss().to(args['device'])                         # Regression loss function
    args['loss_fn_reg'] = loss_fn_reg

    if args['pred_type'] == 'reg+class':
        loss_fn_class = torch.nn.CrossEntropyLoss().to(args['device'])          # Classification loss function
        args['loss_fn_class'] = loss_fn_class
    
    """ Train """
    # train_fn = _train if args['use_unlabeled_samples'] else _train_labeled_only
    train_fn = _train
    train_on_folds(model=model, dataset=labeled_set, unlabeled_dataset=unlabeled_set,  
                   train_fn=train_fn, args=args)

"""
Help with params in case you need it. 
"""
def help():
    print('\'dandadan\' works with \'use_unlabeled_samples\'=[True, False], \'pred_type\'=[\'reg\', \'reg+class\'] and \'date_type\'=[\'month\', \'season\', \'year\'].\n')
    print('\'eanet\' works with \'use_unlabeled_samples\'=False, \'pred_type\'=\'reg\' and does not take \'date_type\'.\n')
    print('With \'date_type\'=\'year\', validation set cannot be created.')
    
    
if __name__ == "__main__":
    seed = 42
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     # Use GPU if available
    args = {'num_folds': None,
            'max_epoch': 10,
            'device': device,
            'seed': seed,
            'create_val': True,                                                 # Creates validation set
            'test_per': 0.1,
            'lr': 0.0001,                                                       # From EA's model, default is 1e-2.
            'patch_norm': True,                                                 # Normalizes patches
            'reg_norm': True,                                                   # Normalize regression values
            'use_unlabeled_samples': False,
            'date_type': 'month',
            'pred_type': 'reg',                                           # Prediction type, can be {'reg', 'class', 'reg+class'}
            'model': 'dandadadan',                                              # Model name, can be {dandadadan, eanet}.
            
            'tr': {'batch_size': C.BATCH_SIZE, 'shuffle': True, 'num_workers': 4},
            'val': {'batch_size': C.BATCH_SIZE, 'shuffle': False, 'num_workers': 4},
            'unlabeled': {'batch_size': C.BATCH_SIZE, 'shuffle': True, 'num_workers': 4},
            'test': {'batch_size': C.BATCH_SIZE, 'shuffle': False, 'num_workers': 4}}
    verify_args(args)
    
    # for use_unlabeled_samples in [True, False]:
    #     args['use_unlabeled_samples'] = use_unlabeled_samples
    #     run(args)
    
    print('\nOnly regression\n')
    run(args)
    print('+' * 72)
    
    args['pred_type'] = 'reg+class'
    print('\nreg+class\n')
    for use_unlabeled_samples in [True, False]:
        args['use_unlabeled_samples'] = use_unlabeled_samples
        print('use_unlabeled_samples: {}'.format(args['use_unlabeled_samples']))
        run(args)
        print('+' * 72)
 
    # for reg_norm in [True, False]:
    #     args['reg_norm'] = reg_norm
    #     for patch_norm in [True, False]:
    #         args['patch_norm'] = patch_norm
    #         # args['use_unlabeled_samples'] = use_unlabeled_samples
    #         print('reg_norm: {}, patch_norm: {}'.format(args['reg_norm'], args['patch_norm']))
    #         run(args)
    #         print('+' * 72)


"""
Ideas:
X 1. Normalize patch values
X 2. Change normalization to https://www.youtube.com/watch?v=y6IEcEBRZks
3. Change metrics calculation from mean of batches to all samples. 
X 4. Try with one-batch learning
X 5. Do not forget to change from one-batch learning to all batches. 

References
[1]: https://medium.com/@benjamin.phillips22/simple-regression-with-neural-networks-in-pytorch-313f06910379
[2]: https://github.com/VICO-UoE/L2I/blob/master/Regression/train-MT.py
[3]: https://github.com/fungtion/DANN/blob/master/train/main.py
"""