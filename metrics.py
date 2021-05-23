
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:40:07 2021

@author: melike
"""

import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, cohen_kappa_score, f1_score, accuracy_score
import numpy as np
import collections
import constants as C

class Metrics:
    """
    Measures regression and classification metrics.
    
    Args:
        num_folds (int): Number of folds. Can be None for training without 
        cross-validation or an int >0 for training with cross-validation. 
    """
    
    def __init__(self, num_folds, device, pred_type, set_name, num_classes=None):
        self.num_folds = num_folds
        self.device = device
        self.pred_type = pred_type
        self.set_name = set_name
        self.test_score_names = self._init_test_score_names()
        self.test_scores = self._init_test_scores()
        self.num_classes = num_classes
        
    def _init_test_score_names(self):
        if self.pred_type == 'reg':
            return ['r2', 'r', 'rmse', 'mae']
        elif self.pred_type == 'class':
            return ['kappa']
        elif self.pred_type == 'reg+class':
            return ['kappa', 'r2', 'r', 'rmse', 'mae']
    
    """
    Create score dict to keep test scores of all folds.
    """
    def _init_test_scores(self):
        scores = {}
        for score_name in self.test_score_names:
            scores[score_name] = {'model_last_epoch.pth': [],                    # Last epoch model
                                  'best_{}_loss.pth'.format(self.set_name): [],  # Best validation loss model
                                  'best_{}_score.pth'.format(self.set_name): [], # Best validation score model
                                  'model_early_stopping.pth': []}                # Early stopping model               
        return scores
            
    """
    Adds one fold score to test scores. 
    """
    def add_fold_score(self, fold_score, model_name):
        for score_name in self.test_score_names:
            s = np.nanmean(fold_score[0][score_name])                           # Has result of each batch, so take mean of it. Kappa may return NaN, so nanmean is used. 
            self.test_scores[score_name][model_name].append(s)
            
    """
    Returns dict of mean and std for each model and score. 
    """
    def get_mean_std_test_results(self):
        result = collections.defaultdict(lambda : collections.defaultdict(dict))
        for score_name in self.test_score_names:
            for model_name, all_fold_scores in self.test_scores[score_name].items():
                if all_fold_scores:                                                     # Check list is empty. 
                    result[score_name][model_name]['mean'] = np.mean(all_fold_scores)
                    result[score_name][model_name]['std'] = np.std(all_fold_scores)
                else:
                    result[score_name][model_name]['mean'] = float('NaN')       # Leave them empty in the report.
                    result[score_name][model_name]['std'] = float('NaN')
        return result
                
    """
    RMSE implementation with PyTorch. 
    """
    def __root_mean_squared_error(self, y_true, y_pred):
        return torch.sqrt(torch.mean(torch.pow(y_true - y_pred, 2.0)))
    
    """
    R2 score implementation with PyTorch.
    From https://en.wikipedia.org/wiki/Coefficient_of_determination
    """
    def __r2_score(self, y_true, y_pred):
        y_true_mean = torch.mean(y_true)
        ss_tot = torch.sum((y_true - y_true_mean) ** 2)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2
    
    """
    R score
    """
    def __r_score(self, y_true, y_pred):
        r2 = self.__r2_score(y_true=y_true, y_pred=y_pred)
        r = torch.sqrt(r2) if r2 > 0 else -torch.sqrt(-r2)
        return r
    
    """
    MAE implementation with PyTorch. 
    """
    def __mean_absolute_error(self, y_true, y_pred):
        mae = torch.nn.L1Loss()
        return mae(y_pred, y_true)

    """
    Calculates confusion matrix. 
    """
    def __calc_conf_matrix(self, preds, labels):
        conf_matrix = { i : {'TP' : 0, 'FP' : 0, 'FN' : 0, 'TN' : 0 } for i in range(self.num_classes)}
        total_size = preds.nelement()
        for i in range(self.num_classes):
            TP = torch.sum((preds == i) * (labels == i)).item()
            FP = torch.sum((preds == i)).item() - TP
            FN = torch.sum((labels == i)).item() - TP
            TN = total_size - TP - FP - FN
            conf_matrix[i]['TP'] += TP
            conf_matrix[i]['FP'] += FP
            conf_matrix[i]['FN'] += FN
            conf_matrix[i]['TN'] += TN
        total_same = torch.sum(preds == labels).item()
        return conf_matrix, total_same
    
    """
    Calculates classification metrics with confusion matrix. 
    """
    def calc_class_metrics(self, preds, labels):
        conf_matrix, total_same = self.__calc_conf_matrix(preds=preds, labels=labels)
        kappa_sum, acc_sum, precision_sum, recall_sum, f1_sum = 0, 0, 0, 0, 0
        scores_per_class = {}
        for k, v in conf_matrix.items():
            total = v['TP'] + v['FP'] + v['FN'] + v['TN']
            assert total > 0, "Total of samples is 0 for class {}!".format(k)
            
            """ ================ Accuracy ================ """
            acc = (v['TP'] + v['TN']) / total
            acc_sum += acc
            
            """ ================ Precision =============== """
            precision = 0 if v['TP'] + v['FP'] == 0 else v['TP'] / (v['TP'] + v['FP'])
            precision_sum += precision
            
            """ ================== Recall ================ """
            recall = 0 if (v['TP'] + v['FN']) == 0 else v['TP'] / (v['TP'] + v['FN'])
            recall_sum += recall
            
            """ ==================== F1 ================== """
            f1 = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
            f1_sum += f1
            
            """ ================== Kappa ================= """
            p_obs = (v['TP'] + v['TN']) / total
            p_yes = ((v['TP'] + v['FP']) / total) * ((v['TP'] + v['FN']) / total)
            p_no = ((v['FN'] + v['TN']) / total) * ((v['FP'] + v['TN']) / total)
            p_est = p_yes + p_no
            kappa = -1.0 if p_est == 1.0 else (p_obs - p_est) / (1 - p_est)
            kappa_sum += kappa
            scores_per_class[k] = {'precision' : precision, 'recall' : recall, 'f1' : f1, 'kappa' : kappa}

        scikit_acc = np.round(total_same / total, decimals=4)
        scores = np.array([precision_sum, recall_sum, f1_sum, kappa_sum])
        p_ov, r_ov, f_ov, k_ov = np.round(scores / self.num_classes, decimals=4)
        scores_overall = {'accuracy' : scikit_acc, 'precision' : p_ov, 'recall' : r_ov, 'f1' : f_ov, 'kappa' : k_ov }
        return scores_overall, scores_per_class
    
    """
    Calculates kappa via overall confusion matrix, not averaging class-wise. 
    """
    def calc_kappa_v2(self, conf_matrix):
        TP, TN, FN, FP = 0, 0, 0, 0
        for k, v in conf_matrix.items():
            TN += v['TN']
            TP += v['TP']
            FN += v['FN']
            FP += v['FP']
            
        total = TP + FP + FN + TN
        obs_agr = TP + TN
        exp_agr = (((TP + FP) * (TP + FN)) + ((FN + TN) * (FP + TN))) / total
        if total == exp_agr:
            raise Exception('Total of samples is equal to expected agreement!')
        kappa = (obs_agr - exp_agr) / (total - exp_agr)
        return np.round(kappa, decimals=4)
        
    """
    Calculates regression metrics for given batch.
    """
    def eval_reg_batch_metrics(self, preds, targets):
        rmse = self.__root_mean_squared_error(y_pred=preds, y_true=targets)
        r2 = self.__r2_score(y_pred=preds, y_true=targets)
        mae = self.__mean_absolute_error(y_pred=preds, y_true=targets)
        r = self.__r_score(y_pred=preds, y_true=targets)
        # print('rmse: {:.4f}, r2: {:.4f}, mae: {:.4f}'.format(rmse, r2, mae))

        """ Scikit metrics """
        ###################################################################### 
        # if self.device != 'cpu':
        #     preds, targets = preds.cpu(), targets.cpu()
        # preds, targets = preds.detach().numpy().flatten(), targets.detach().numpy().flatten()
        # sk_rmse = mean_squared_error(y_true=targets, y_pred=preds, squared=False)
        # sk_r2 = r2_score(y_true=targets, y_pred=preds)
        # sk_mae = mean_absolute_error(y_true=targets, y_pred=preds)
        # print('scikit rmse: {:.4f}, r2: {:.4f}, mae: {:.4f}'.format(sk_rmse, sk_r2, sk_mae))
        ######################################################################
        return {'rmse' : rmse.item(), 'r2' : r2.item(), 'mae' : mae.item(), 'r' : r.item()}
    
    """
    Calculates classification metrics for given batch.
    """
    def eval_class_batch_metrics(self, preds, targets):
        preds = torch.argmax(preds, 1)                                          # No need to softmax before argmax, result is the same. 
        scores_overall, scores_per_class = self.calc_class_metrics(preds=preds, labels=targets)
        kappa, f1, acc = scores_overall['kappa'], scores_overall['f1'], scores_overall['accuracy']
        # print('kappa: {:.4f}, f1: {:.4f}, acc: {:.4f}'.format(kappa, f1, acc))
        
        """ Scikit metrics """
        ######################################################################
        # if self.device != 'cpu':
        #     preds, targets = preds.cpu(), targets.cpu()
        # preds, targets = preds.detach().numpy().flatten(), targets.detach().numpy().flatten()
        # sk_kappa = cohen_kappa_score(preds, targets)
        # sk_f1 = f1_score(y_true=targets, y_pred=preds, average='macro')
        # sk_acc = accuracy_score(y_true=targets, y_pred=preds)
        # print('scikit kappa: {:.4f}, f1: {:.4f}, acc: {:.4f}'.format(sk_kappa, sk_f1, sk_acc))
        ######################################################################
        return {'kappa' : kappa, 'f1' : f1, 'acc' : acc}
    
"""
Ideas:
X 1. Implement regression and classification scores in Pytorch. 
"""

if __name__ == '__main__':
    num_classes = 12
    num_samples = 2
    metrics = Metrics(num_folds=3, device='cpu', pred_type='reg+class', num_classes=num_classes)
    
    """ Classification """
    labels = torch.randint(0, num_classes, (num_samples,))
    preds = torch.rand(num_samples, num_classes)
    metrics.eval_class_batch_metrics(preds=preds, targets=labels)
    
    """ Regression """
    # preds = torch.rand(num_samples)
    # targets = torch.rand(num_samples)
    # metrics.eval_reg_batch_metrics(preds=preds, targets=targets)
    
    


