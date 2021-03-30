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
        self.sheet.write(0, 6, 'UseUnlabeled', self.header)
        self.sheet.write(0, 7, 'Seed', self.header)
        self.sheet.write(0, 8, 'PatchNorm', self.header)
        self.sheet.write(0, 9, 'RegNorm', self.header)
        self.sheet.write(0, 10, 'DateType', self.header)                         # Only meaningful in case of class and reg+class.
        self.sheet.write(0, 11, 'Model', self.header)
        self.sheet.write(0, 12, 'ScoreName', self.header)
        self.sheet.write(0, 13, 'LastEpochModelMean', self.header)              # For folded cases mean of score, o.w. score itself. 
        self.sheet.write(0, 14, 'LastEpochModelStd', self.header)               # For folded cases std of score, o.w. 0.
        self.sheet.write(0, 15, 'BestValLossModelMean', self.header)
        self.sheet.write(0, 16, 'BestValLossModelStd', self.header)
        self.sheet.write(0, 17, 'BestValScoreModelMean', self.header)
        self.sheet.write(0, 18, 'BestValScoreModelStd', self.header)
        self.sheet.write(0, 19, 'ScoreName', self.header)
        self.sheet.write(0, 20, 'LastEpochModelMean', self.header)              # For folded cases mean of score, o.w. score itself. 
        self.sheet.write(0, 21, 'LastEpochModelStd', self.header)               # For folded cases std of score, o.w. 0.
        self.sheet.write(0, 22, 'BestValLossModelMean', self.header)
        self.sheet.write(0, 23, 'BestValLossModelStd', self.header)
        self.sheet.write(0, 24, 'BestValScoreModelMean', self.header)
        self.sheet.write(0, 25, 'BestValScoreModelStd', self.header)
        # self.sheet.write(0, 25, 'Loss', self.header)                          # AWL or sum

        
    """
    Adds given dataset, its score and model name to the report. 
    """
    def add(self, args, metrics):
        rid = len(self.sheet._Worksheet__rows)
        test_result = metrics.get_mean_std_test_result()
        self.sheet.write(rid, 0, args['run_name'])
        self.sheet.write(rid, 1, args['fold_setup'])
        self.sheet.write(rid, 2, args['num_folds'])                             # Check for None. 
        self.sheet.write(rid, 3, args['pred_type'])
        self.sheet.write(rid, 4, args['num_epoch'])
        self.sheet.write(rid, 5, args['create_val'])
        self.sheet.write(rid, 6, args['use_unlabeled_samples'])
        self.sheet.write(rid, 7, args['seed'])
        self.sheet.write(rid, 8, args['patch_norm'])
        self.sheet.write(rid, 9, args['reg_norm'])
        self.sheet.write(rid, 10, args['date_type'])
        self.sheet.write(rid, 11, args['model'])
        # self.sheet.write(rid, 12, score_name)

    """
    Returns current report's id. 
    """
    def _get_report_id(self):
        return len([name for name in os.listdir(C.REPORT_PATH) if osp.isfile(os.path.join(C.REPORT_PATH, name))])
    
    """
    Saves the report. 
    """
    def save(self):
        path = osp.join(C.REPORT_PATH, str(self._get_report_id()) + '.xls')
        self.wb.save(path)
        print('Report saved to', path)
    
    