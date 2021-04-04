#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:15:27 2021

@author: melike
"""

import os
import os.path as osp
from xlwt import Workbook
import xlwt
import math
import constants as C

class Report:
    """
    Creates a spreadsheet for the run and writes all experiment results to it.
    """
    def __init__(self):
        self.wb = Workbook()
        self.sheet = self.wb.add_sheet('Scores')
        self.header = xlwt.easyxf('font: bold 1', 'align: vert centre, horiz centre', 'alignment: wrap True')
        self._init_report()
        
    """
    Initializes report fields. 
    """
    def _init_report(self):
        self.sheet.write(0, 0, 'RunName', self.header)
        self.sheet.write(0, 1, 'Setup', self.header)
        self.sheet.write(0, 2, 'NumFolds', self.header)
        self.sheet.write(0, 3, 'PredType', self.header)
        self.sheet.write(0, 4, 'NumEpochs', self.header)
        self.sheet.write(0, 5, 'HasValSet', self.header)
        self.sheet.write(0, 6, 'UsedUnlabeled', self.header)
        self.sheet.write(0, 7, 'NumTrainSamples (labeled)', self.header)
        self.sheet.write(0, 8, 'NumTestSamples', self.header)
        self.sheet.write(0, 9, 'NumValSamples', self.header)
        self.sheet.write(0, 10, 'NumTrainSamples (unlabeled', self.header)
        self.sheet.write(0, 11, 'Seed', self.header)
        self.sheet.write(0, 12, 'PatchNorm', self.header)
        self.sheet.write(0, 13, 'RegNorm', self.header)
        self.sheet.write(0, 14, 'DateType', self.header)                        # Only meaningful in case of class and reg+class.
        self.sheet.write(0, 15, 'Model', self.header)
        self.sheet.write(0, 16, 'ModelSplitLayer', self.header)
        self.sheet.write(0, 17, 'ScoreName', self.header)
        self.sheet.write(0, 18, 'LastEpochModelMean', self.header)              # For folded cases mean of score, o.w. score itself. 
        self.sheet.write(0, 19, 'LastEpochModelStd', self.header)               # For folded cases std of score, o.w. 0.
        self.sheet.write(0, 20, 'BestValLossModelMean', self.header)
        self.sheet.write(0, 21, 'BestValLossModelStd', self.header)
        self.sheet.write(0, 22, 'BestValScoreModelMean', self.header)
        self.sheet.write(0, 23, 'BestValScoreModelStd', self.header)
        self.sheet.write(0, 24, 'ScoreName', self.header)
        self.sheet.write(0, 25, 'LastEpochModelMean', self.header)              # For folded cases mean of score, o.w. score itself. 
        self.sheet.write(0, 26, 'LastEpochModelStd', self.header)               # For folded cases std of score, o.w. 0.
        self.sheet.write(0, 27, 'BestValLossModelMean', self.header)
        self.sheet.write(0, 28, 'BestValLossModelStd', self.header)
        self.sheet.write(0, 29, 'BestValScoreModelMean', self.header)
        self.sheet.write(0, 30, 'BestValScoreModelStd', self.header)
        # self.sheet.write(0, 25, 'Loss', self.header)                          # AWL or sum

    """
    """
    def _test_result_to_sheet(self, test_result, rid, idx):
        init_idx = idx
        for score_name in test_result:
            if len(test_result) == 1 and score_name == 'reg':                   # Make R2 score for reg and reg+class start at the same column. 
                idx = idx + 7
            self.sheet.write(rid, idx, score_name)
            idx = idx + 1
            for model_name in test_result[score_name]:
                for res in test_result[score_name][model_name]:
                    val = test_result[score_name][model_name][res]
                    if not math.isnan(val):                                     # Leave NaN values are blank in column
                        self.sheet.write(rid, idx, '{:.4f}'.format(val))
                    idx = idx + 1
            if len(test_result[score_name]) == 1:                               # Has only one model, no val set. So, skip val score columns. 
                idx = idx + 4
        return init_idx + 14                                                    # 7x2=14 columns for results.    
        
    """
    Adds given dataset, its score and model name to the report. 
    """
    def add(self, args, test_result):
        rid = len(self.sheet._Worksheet__rows)
        self.sheet.write(rid, 0, args['run_name'])
        self.sheet.write(rid, 1, args['fold_setup'])
        self.sheet.write(rid, 2, args['num_folds'])                             # Check for None. 
        self.sheet.write(rid, 3, args['pred_type'])
        self.sheet.write(rid, 4, args['max_epoch'])
        self.sheet.write(rid, 5, 'True' if args['create_val'] == 1 else 'False')
        self.sheet.write(rid, 6, 'True' if args['use_unlabeled_samples'] == 1 else 'False')
        self.sheet.write(rid, 7, args['train_size'])
        self.sheet.write(rid, 8, args['test_size'])
        self.sheet.write(rid, 9, args['val_size'] if args['val_size'] is not None else 0)
        self.sheet.write(rid, 10, args['unlabeled_size'] if args['unlabeled_size'] is not None else 0)
        self.sheet.write(rid, 11, args['seed'])
        self.sheet.write(rid, 12, 'True' if args['patch_norm'] == 1 else 'False')
        self.sheet.write(rid, 13, 'True' if args['reg_norm'] == 1 else 'False')
        self.sheet.write(rid, 14, '' if args['pred_type'] == 'reg' else args['date_type'])
        self.sheet.write(rid, 15, args['model'])
        self.sheet.write(rid, 16, args['split_layer'] if args['model'] == 'eadan' else '')
        idx = self._test_result_to_sheet(test_result=test_result, rid=rid, idx=17)
        
    """
    Returns current report's id. 
    """
    def get_report_id(self):
        report_dir_path = osp.join(os.getcwd(), 'reports')
        return len([name for name in os.listdir(report_dir_path) if osp.isfile(os.path.join(report_dir_path, name))])
    
    """
    Saves the report. 
    """
    def save(self):
        report_id = self.get_report_id()
        path = osp.join(osp.join(os.getcwd(), 'reports'), str(report_id) + '.xls')
        self.wb.save(path)
        print('Report saved to', path)
        return report_id
    
    