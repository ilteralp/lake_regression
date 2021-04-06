#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:15:27 2021

@author: melike
"""

import os
import os.path as osp
import xlwt
import xlrd
from xlwt import Workbook
from xlrd import open_workbook
from xlutils.copy import copy
import math
import constants as C

class Report:
    """
    Creates a spreadsheet for the run and writes all experiment results to it.
    """
    def __init__(self):
        # self.wb = Workbook()
        # self.sheet = self.wb.add_sheet('Scores')
        # self.header = xlwt.easyxf('font: bold 1', 'align: vert centre, horiz centre', 'alignment: wrap True')
        self.report_id = self.__calc_report_id()
        self.path = self.__init_path()
        self.__init_report()
        
    """
    Returns current report's id. 
    """
    def __calc_report_id(self):
        report_dir_path = osp.join(os.getcwd(), 'reports')
        return len([name for name in os.listdir(report_dir_path) if osp.isfile(os.path.join(report_dir_path, name))])
    
    """
    Initializes report path. 
    """
    def __init_path(self):
        return osp.join(os.getcwd(), 'reports', '{}.xls'.format(self.report_id))
        
    """
    Initializes report with headers and saves it. 
    """
    def __init_report(self):
        wb = Workbook()
        sheet = wb.add_sheet('Scores')
        header = xlwt.easyxf('font: bold 1', 'align: vert centre, horiz centre', 'alignment: wrap True')
        sheet.write(0, 0, 'RunName', header)
        sheet.write(0, 1, 'Setup', header)
        sheet.write(0, 2, 'NumFolds', header)
        sheet.write(0, 3, 'PredType', header)
        sheet.write(0, 4, 'NumEpochs', header)
        sheet.write(0, 5, 'HasValSet', header)
        sheet.write(0, 6, 'UsedUnlabeled', header)
        sheet.write(0, 7, 'NumTrainSamples (labeled)', header)
        sheet.write(0, 8, 'NumTestSamples', header)
        sheet.write(0, 9, 'NumValSamples', header)
        sheet.write(0, 10, 'NumTrainSamples (unlabeled', header)
        sheet.write(0, 11, 'Seed', header)
        sheet.write(0, 12, 'PatchNorm', header)
        sheet.write(0, 13, 'RegNorm', header)
        sheet.write(0, 14, 'DateType', header)                        # Only meaningful in case of class and reg+class.
        sheet.write(0, 15, 'Model', header)
        sheet.write(0, 16, 'ModelSplitLayer', header)
        sheet.write(0, 17, 'LossName', header)
        sheet.write(0, 18, 'ScoreName', header)
        sheet.write(0, 19, 'LastEpochModelMean', header)              # For folded cases mean of score, o.w. score itself. 
        sheet.write(0, 20, 'LastEpochModelStd', header)               # For folded cases std of score, o.w. 0.
        sheet.write(0, 21, 'BestValLossModelMean', header)
        sheet.write(0, 22, 'BestValLossModelStd', header)
        sheet.write(0, 23, 'BestValScoreModelMean', header)
        sheet.write(0, 24, 'BestValScoreModelStd', header)
        sheet.write(0, 25, 'ScoreName', header)
        sheet.write(0, 26, 'LastEpochModelMean', header)              # For folded cases mean of score, o.w. score itself. 
        sheet.write(0, 27, 'LastEpochModelStd', header)               # For folded cases std of score, o.w. 0.
        sheet.write(0, 28, 'BestValLossModelMean', header)
        sheet.write(0, 29, 'BestValLossModelStd', header)
        sheet.write(0, 30, 'BestValScoreModelMean', header)
        sheet.write(0, 31, 'BestValScoreModelStd', header)
        wb.save(self.path)

    """
    """
    def _test_result_to_sheet(self, sheet, test_result, rid, idx):
        init_idx = idx
        for score_name in test_result:
            if len(test_result) == 1 and score_name == 'r2':                   # Make R2 score for reg and reg+class start at the same column. 
                idx = idx + 7
            sheet.write(rid, idx, score_name)
            idx = idx + 1
            for model_name in test_result[score_name]:
                for res in test_result[score_name][model_name]:
                    val = test_result[score_name][model_name][res]
                    if not math.isnan(val):                                     # Leave NaN values are blank in column
                        sheet.write(rid, idx, '{:.4f}'.format(val))
                    idx = idx + 1
            if len(test_result[score_name]) == 1:                               # Has only one model, no val set. So, skip val score columns. 
                idx = idx + 4
        return init_idx + 14                                                    # 7x2=14 columns for results.    
    
    """
    Reads and returns workbook and row index. 
    """
    def __load_workbook(self):
        rb = xlrd.open_workbook(self.path, formatting_info=True)
        r_sheet = rb.sheet_by_index(0)
        num_rows = r_sheet.nrows
        wb = copy(rb)
        return wb, num_rows
        
    """
    Adds given dataset, its score and model name to the report. 
    """
    def add(self, args, test_result):
        # rid = len(self.sheet._Worksheet__rows)
        wb, rid = self.__load_workbook()
        sheet = wb.get_sheet(0)
        
        sheet.write(rid, 0, args['run_name'])
        sheet.write(rid, 1, args['fold_setup'])
        sheet.write(rid, 2, args['num_folds'])                                  # Check for None. 
        sheet.write(rid, 3, args['pred_type'])
        sheet.write(rid, 4, args['max_epoch'])
        sheet.write(rid, 5, 'True' if args['create_val'] == 1 else 'False')
        sheet.write(rid, 6, 'True' if args['use_unlabeled_samples'] == 1 else 'False')
        sheet.write(rid, 7, args['train_size'])
        sheet.write(rid, 8, args['test_size'])
        sheet.write(rid, 9, args['val_size'] if args['val_size'] is not None else 0)
        sheet.write(rid, 10, args['unlabeled_size'] if args['unlabeled_size'] is not None else 0)
        sheet.write(rid, 11, args['seed'])
        sheet.write(rid, 12, 'True' if args['patch_norm'] == 1 else 'False')
        sheet.write(rid, 13, 'True' if args['reg_norm'] == 1 else 'False')
        sheet.write(rid, 14, '' if args['pred_type'] == 'reg' else args['date_type'])
        sheet.write(rid, 15, args['model'])
        sheet.write(rid, 16, args['split_layer'] if args['model'] == 'eadan' else '')
        sheet.write(rid, 17, args['loss_name'])
        idx = self._test_result_to_sheet(test_result=test_result, sheet=sheet, rid=rid, idx=18)
        wb.save(self.path)
        
    # """
    # Saves the report. 
    # """
    # def save(self):
    #     report_id = self.get_report_id()
    #     path = osp.join(osp.join(os.getcwd(), 'reports'), str(report_id) + '.xls')
    #     self.wb.save(path)
    #     print('Report saved to', path)
    #     return report_id
    
if __name__ == '__main__':
    report = Report()