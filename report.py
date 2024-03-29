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
    
    def __init_sheet_for_scores(self, sheet, header, idx):
        for (score_id, _) in enumerate(['kappa', 'r2', 'r', 'mae', 'rmse']):
            cid = idx + score_id * 9                                            # 9 model results
            sheet.write(0, cid + 1, 'ScoreName', header)
            sheet.write(0, cid + 2, 'LastEpochModelMean', header)               # For folded cases mean of score, o.w. score itself. 
            sheet.write(0, cid + 3, 'LastEpochModelStd', header)                # For folded cases std of score, o.w. 0.
            sheet.write(0, cid + 4, 'BestValLossModelMean', header)
            sheet.write(0, cid + 5, 'BestValLossModelStd', header)
            sheet.write(0, cid + 6, 'BestValScoreModelMean', header)
            sheet.write(0, cid + 7, 'BestValScoreModelStd', header)
            sheet.write(0, cid + 8, 'EarlyStoppingModelMean', header)
            sheet.write(0, cid + 9, 'EarlyStoppingModelStd', header)
        return cid + 10                                                          # Return index to write
                        
        
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
        sheet.write(0, 14, 'DateType', header)                                   # Only meaningful in case of class and reg+class.
        sheet.write(0, 15, 'Model', header)
        sheet.write(0, 16, 'ModelSplitLayer', header)
        sheet.write(0, 17, 'LossName', header)
        
        idx = self.__init_sheet_for_scores(sheet=sheet, header=header, idx=17)
        sheet.write(0, idx, 'ReportId', header)
        sheet.write(0, idx + 1, 'PatchSize', header)
        sheet.write(0, idx + 2, 'SampleIdsFromRun', header)
        sheet.write(0, idx + 3, 'UnlabeledBatchSize', header)
        sheet.write(0, idx + 4, 'LearningRate', header)
        sheet.write(0, idx + 5, 'LR_Reg', header)
        sheet.write(0, idx + 6, 'LR_Class', header)
        sheet.write(0, idx + 7, 'UseAtrousConv', header)
        sheet.write(0, idx + 8, 'ReshapeToMosaic', header)
        sheet.write(0, idx + 9, 'StartFold', header)
        sheet.write(0, idx + 10, 'TotalModelParams', header)
        wb.save(self.path)

    """
    """
    def _test_result_to_sheet(self, sheet, test_result, rid, idx):
        init_idx = idx
        for score_name in test_result:
            # if len(test_result) == 1 and score_name == 'r2':                   # Make R2 score for reg and reg+class start at the same column. 
            if len(test_result) == 4 and score_name == 'r2':                     # Only reg results, no kappa. So, skip kappa columns in first encounter. 
                idx = idx + 9
            sheet.write(rid, idx, score_name)
            idx = idx + 1
            for model_name in test_result[score_name]:
                for res in test_result[score_name][model_name]:
                    val = test_result[score_name][model_name][res]
                    if not math.isnan(val):                                     # Leave NaN values are blank in column
                        sheet.write(rid, idx, '{:.4f}'.format(val))
                    idx = idx + 1
            if len(test_result[score_name]) == 1:                               # Has only one model, no val set. So, skip val score columns. 
                idx = idx + 6
        return init_idx + 45                                                    # 9x5=45 columns for results.    
    
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
        sheet.write(rid, 16, args['split_layer'] if args['model'] in C.DAN_MODELS else '')
        sheet.write(rid, 17, args['loss_name'])
        idx = self._test_result_to_sheet(test_result=test_result, sheet=sheet, rid=rid, idx=18)
        sheet.write(rid, idx, self.report_id)
        sheet.write(rid, idx + 1, args['patch_size'])
        sheet.write(rid, idx + 2, args['sample_ids_from_run'] if args['sample_ids_from_run'] is not None else '')
        sheet.write(rid, idx + 3, args['unlabeled']['batch_size'] if args['use_unlabeled_samples'] else '')
        sheet.write(rid, idx + 4, args['lr'])
        sheet.write(rid, idx + 5, args['lr_reg'] if 'lr_reg' in args else '')
        sheet.write(rid, idx + 6, args['lr_class'] if 'lr_class' in args else '')
        sheet.write(rid, idx + 7, '' if 'use_atrous_conv' not in args else 'True' if args['use_atrous_conv'] else 'False')
        sheet.write(rid, idx + 8, 'True' if args['reshape_to_mosaic'] == 1 else 'False')
        sheet.write(rid, idx + 9, args['start_fold'])
        sheet.write(rid, idx + 10, args['total_model_params'])
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