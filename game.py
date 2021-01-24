# this is borrowed from https://github.com/val-iisc/lsc-cnn

import numpy as np
import torch
'''
    Calculates GAME Metric as mentioned by Guerrero-Gómez-Olmedo et al. in 
    Extremely Overlapping Vehicle Counting, in IbPRIA, 2015.

    Parameters:
    -----------
    level - (int) level of GAME metric.
    gt - (np.ndarray) binary map of ground truth (HXW)
    pred - (np.ndarray) binary map of predictions (HXW)

    Returns
    -------
    mae - GAME for the level mentioned in the input.
'''
def game_metric(level, gt, pred):
    assert(gt.shape == pred.shape)
    num_cuts = np.power(2, level)
    H, W = gt.shape
    h = H//num_cuts
    w = W//num_cuts
    gt_new = np.zeros((num_cuts*num_cuts, h, w))
    pred_new = np.zeros((num_cuts*num_cuts, h, w))
    
    for y in range(num_cuts):
        for x in range(num_cuts):
            gt_new[y*num_cuts+x,:,:]=gt[y*h:y*h+h,x*w:x*w+w]
            pred_new[y*num_cuts+x,:,:]=pred[y*h:y*h+h,x*w:x*w+w]
    
    gt_sum = np.sum(np.sum(gt_new, axis=1), axis=1)    
    pred_sum = np.sum(np.sum(pred_new, axis=1), axis=1) 
    mae = np.sum(np.abs(gt_sum - pred_sum))
    return mae
'''
    Wrapper for calculating GAME Metric

    Parameters:
    -----------
    gt - (np.ndarray) binary map of ground truth (HXW)
    pred - (np.ndarray) binary map of predictions (HXW)

    Returns
    -------
    mae_l1, mae_l2, mae_l3 - GAME for 3 different levels
'''
def find_game_metric(gt, pred):

    mae_l1 = game_metric(1, gt, pred)
    mae_l2 = game_metric(2, gt, pred)
    mae_l3 = game_metric(3, gt, pred)

    return mae_l1, mae_l2, mae_l3
