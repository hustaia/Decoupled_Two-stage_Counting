import sys
import os
import shutil

import warnings

from VGG16_Unet import *
from TasselNetv2 import *
from TasselNetv2_VGG16 import *
import utils
import dataset
from game import *
from ap import *

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import time
import random
import h5py
import math
import matplotlib.pyplot as plt
from skimage import measure
from tensorboardX import SummaryWriter


    
parser = argparse.ArgumentParser(description='PyTorch D2CNet')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                    help='path to the pretrained model')
                    
parser.add_argument('gpu',metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task',metavar='TASK', type=str,
                    help='task id to use.')
                    
parser.add_argument('--use_synthesized_data', required=False, default=True, type=bool,
                    help='whether to use synthesized data.')
                    
parser.add_argument('--mode', required=False, default=0, type=int,
                    help='select a mode, 0 for train, 1 for evaluate_counting, 2 for evaluate_localization.')


#create log for ssh check
localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
outputfile = open("./logs/record"+localtime+".txt", 'w')
try:
    from termcolor import cprint
except ImportError:
    cprint = None
    
def log_print(text, color=None, on_color=None, attrs=None, outputfile=outputfile):
    print(text, file=outputfile)
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)



def main():
    
    global args,best_prec1
    
    best_prec1 = 1e6
    
    args = parser.parse_args()
    args.original_lr   = 1e-5
    args.lr            = 1e-5
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5e-3
    args.start_epoch   = 0
    args.epochs        = 400
    args.steps         = [5,20,40,60]
    args.scales        = [1,1,1,1]
    args.workers       = 1
    args.seed          = time.time()
    args.print_freq    = 300
    
    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)

    criterion = (nn.MSELoss(size_average=False).cuda(), 
                nn.L1Loss(size_average=False).cuda())
    PMR = VGG16_Unet()
    CMR = TasselNet_VGG16()
    #CMR = TasselNetv2()
    PMR = PMR.cuda()
    CMR = CMR.cuda()
                      
    optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, PMR.parameters()), lr=args.lr, weight_decay=args.decay)
    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, CMR.parameters()), lr=args.lr, weight_decay=args.decay)

    log_text = "PMR: {}, CMR: {}, criterion: {}, optimizer1: {}, optimizer2: {}".format(PMR, CMR, criterion, optimizer1, optimizer2)
    log_print(log_text, color='white', attrs=['bold'])

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            PMR.load_state_dict(checkpoint['state_dict1'])
            CMR.load_state_dict(checkpoint['state_dict2'])
            #optimizer1.load_state_dict(checkpoint['optimizer1'])
            #optimizer2.load_state_dict(checkpoint['optimizer2'])
            log_text = "=> loaded checkpoint '{}' (epoch {})".format(args.pre, checkpoint['epoch'])
            log_print(log_text, color='white', attrs=['bold'])
        else:
            log_text = "=> no checkpoint found at '{}', use default init instead".format(args.pre)
            log_print(log_text, color='white', attrs=['bold'])
            
            
    if args.mode == 0: #train
        for epoch in range(args.start_epoch, args.epochs):
            
            adjust_learning_rate(optimizer1, epoch)
            adjust_learning_rate(optimizer2, epoch)
            
            train(train_list, PMR, CMR, criterion, optimizer1, optimizer2, epoch)
            is_best = 0
            if epoch % 1 == 0 :
                prec1 = validate(val_list, PMR, CMR, epoch)
                is_best = prec1 < best_prec1
                best_prec1 = min(prec1, best_prec1)
                log_text = ' * best MAE {mae:.3f} '.format(mae=best_prec1)
                log_print(log_text, color='red', attrs=['bold'])
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.pre,
                'state_dict1': PMR.state_dict(),
                'state_dict2': CMR.state_dict(),
                'best_prec1': best_prec1,
                'optimizer1' : optimizer1.state_dict(),
                'optimizer2' : optimizer2.state_dict(),
            }, is_best,args.task)
            
        outputfile.close()
    elif args.mode == 1: #evaluate_counting
        assert os.path.isfile(args.pre), 'must evaluate on pretrained model.'
        epoch = args.start_epoch
        prec1 = validate(val_list, PMR, CMR, epoch)
    elif args.mode == 2: #evaluate_localization
        assert os.path.isfile(args.pre), 'must evaluate on pretrained model.'
        epoch = args.start_epoch
        prec1 = validate_loc(val_list, PMR, CMR, epoch)
   
def train(train_list, PMR, CMR, criterion, optimizer1, optimizer2, epoch):
    
    prob_loss_class = AverageMeter()
    count_loss_class = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       crop=True,
                       scale=True,
                       flip=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]), 
                       train=True, 
                       num_workers=args.workers),
        batch_size=args.batch_size)
    log_text = 'epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr)
    log_print(log_text, color='green', attrs=['bold'])
    
    PMR.eval()
    CMR.eval()
    end = time.time()
    
    for i, (img, count_target, prob_target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        img = img.cuda()
        img = Variable(img)
        count_target = count_target.type(torch.FloatTensor)
        count_target = Variable(count_target).cuda()
        prob_target = prob_target.type(torch.FloatTensor)
        prob_target = Variable(prob_target).cuda()

        probmap = PMR(img)
        cat_probmap = torch.cat([probmap,prob_target],0)
        countmap = CMR(cat_probmap)

        prob_loss = criterion[0](probmap, prob_target)
        rf = CMR.rf
        count_target = torch.nn.functional.conv2d(count_target, torch.ones(1,1,rf,rf).cuda(), bias=None, stride=rf, padding=0, dilation=1, groups=1)
        cat_target = torch.cat([count_target,count_target],0)
        count_loss = criterion[0](countmap, cat_target)
        all_loss = prob_loss + count_loss
        prob_loss_class.update(prob_loss.item(), img.size(0))
        count_loss_class.update(count_loss.item(), img.size(0))
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        all_loss.backward()
        optimizer1.step()
        optimizer2.step()

        if args.use_synthesized_data:
            number = int(999 * random.random()) + 1
            number = '{:0>6d}'.format(number)
            if 'A' in args.train_json:
                img_path = "./data/synthesized_dataset/partA/dotmap/" + number + '.h5'
            elif 'B' in args.train_json:
                img_path = "./data/synthesized_dataset/partB/dotmap/" + number + '.h5'
            elif 'Q' in args.train_json:
                img_path = "./data/synthesized_dataset/QNRF/dotmap/" + number + '.h5'
            prob_file = h5py.File(img_path.replace('dotmap','probmap'),'r')
            probmap = np.asarray(prob_file['density'])
            count_file = h5py.File(img_path.replace('dotmap','densitymap'),'r')
            countmap = np.asarray(count_file['density'])

            h, w = countmap.shape
            dx = int((w - 384) * random.random())
            dy = int((h - 384) * random.random())
            probmap = probmap[dy:dy+384, dx:dx+384]
            countmap = countmap[dy:dy+384, dx:dx+384]
                
            rf = CMR.rf
            countmap = torch.from_numpy(countmap).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
            countmap = torch.nn.functional.conv2d(countmap, torch.ones(1,1,rf,rf), bias=None, stride=rf, padding=0, dilation=1, groups=1)
            probmap = torch.from_numpy(probmap).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
            countmap = countmap.cuda()
            probmap = probmap.cuda()
                
            countmap_pred = CMR(probmap)
            synthesized_count_loss = criterion[0](countmap, countmap_pred)
            optimizer2.zero_grad()
            synthesized_count_loss.backward()
            optimizer2.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log_text = (('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Prob_loss {prob_loss.val:.4f} ({prob_loss.avg:.4f})\t'
                    'Count_loss {count_loss.val:.4f} ({count_loss.avg:.4f})\t')
                    .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, prob_loss=prob_loss_class, count_loss=count_loss_class))
            log_print(log_text, color='green', attrs=['bold'])
            

def validate(val_list, PMR, CMR, epoch):
    print('begin val')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   crop=False,
                   scale=False,
                   flip=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),
                   train=False),
    batch_size=1)    
    
    PMR.eval()
    CMR.eval()
    
    mae = 0
    mse = 0

    for i, (img, _, _, dotmap) in enumerate(test_loader):
        
        img = img.cuda()
        img = Variable(img)

        with torch.no_grad():
            
            probmap = PMR(img)  
            countmap = CMR(probmap)

            gt_count = dotmap.numpy().sum()        
            et_count = countmap.detach().sum().cpu().numpy()
            
            print(i,gt_count,et_count)
            mae += abs(gt_count-et_count)
            mse += ((gt_count-et_count)*(gt_count-et_count))
 
    mae = mae/len(test_loader)
    mse = np.sqrt(mse/(len(test_loader)))

    if epoch%1==0:
        log_text = ' * MAE {mae:.3f}--MSE {mse:.3f} '.format(mae=mae,mse=mse)
        log_print(log_text, color='yellow', attrs=['bold'])

    return mae 

        
def validate_loc(val_list, PMR, CMR, epoch):
    print('begin val localization')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   crop=False,
                   scale=False,
                   flip=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),
                   train=False),
    batch_size=1)    
    
    PMR.eval()
    CMR.eval()
    
    mae = 0
    mse = 0
    game1 = 0
    game2 = 0
    game3 = 0
    tp_sum = 0
    gt_sum = 0
    et_sum = 0
    
    for ii, (img, _, _, dot_target) in enumerate(test_loader):
        
        img = img.cuda()
        img = Variable(img)

        with torch.no_grad():
            
            probmap = PMR(img)  
            countmap = CMR(probmap)
            
            avg_pooled_probmap = nn.functional.avg_pool2d(probmap, 3, stride=1, padding=1)
            max_pooled_probmap = nn.functional.max_pool2d(avg_pooled_probmap, 3, stride=1, padding=1)
            candidate_peak = torch.where(avg_pooled_probmap==max_pooled_probmap, avg_pooled_probmap, torch.full_like(probmap, 0))
            
            countmap = countmap.detach().cpu().numpy().squeeze()
            _, _, H, W = probmap.shape
            rf = CMR.rf
            h, w = H//rf, W//rf
            dotmap = np.zeros([H,W]) 
            cnt = np.zeros([h,w]) 
            
            left = []
            right = []
            up = []
            down = []
            
            for y in range(h-1):
                for x in range(w-1):
                    flag = 0
                    for i in range(2, min(h-y, w-x)+1):
                        if countmap[y:y+i, x:x+i].sum()>1:
                            flag = 1
                            left.append(x)
                            right.append(x+i)
                            up.append(y)
                            down.append(y+i)
                            cnt[y:y+i, x:x+i] += 1.0
                            break
                    if flag == 0:
                        left.append(x)
                        right.append((x+min(h-y, w-x)))
                        up.append(y)
                        down.append((y+min(h-y, w-x)))
                        cnt[y:y+min(h-y, w-x), x:x+min(h-y, w-x)] += 1.0

                                                                      
            for i in range(len(left)):
                left[i] = left[i] * rf
                right[i] = right[i] * rf
                up[i] = up[i] * rf
                down[i] = down[i] * rf
                
                   
            for i in range(len(left)):
                sum = int(round(countmap[up[i]//rf:down[i]//rf, left[i]//rf:right[i]//rf].sum()))
                if sum <= 0:
                    continue
                tmp_peak = candidate_peak.clone()
                for _ in range(sum):
                    argmax = tmp_peak[0, 0, up[i]:down[i], left[i]:right[i]].argmax()    
                    arg_h = argmax // (right[i]-left[i])
                    arg_w = argmax % (right[i]-left[i])
                    dotmap[up[i]+arg_h, left[i]+arg_w] += 1
                    tmp_peak[0, 0, up[i]+arg_h, left[i]+arg_w] = 0
        
            cnt = cv2.resize(cnt, (W,H), interpolation=cv2.INTER_NEAREST)
            dotmap = np.divide(dotmap, cnt)
            dotmap = np.where(dotmap>=0.5, 1, 0)
            

            dot_target = dot_target.numpy().squeeze()
            gt_count = dot_target.sum()  
            et_count = dotmap.sum()
            g1, g2, g3 = find_game_metric(dot_target, dotmap)
            tp = compute_tp(dotmap, dot_target)
            precision = tp / (et_count + 1e-6)
            recall = tp / (gt_count + 1e-6)
                
            print(ii,gt_count,et_count,g1,g2,g3,round(precision,3),round(recall,3))
            mae += abs(gt_count-et_count)
            mse += ((gt_count-et_count)*(gt_count-et_count))
            game1 += g1
            game2 += g2
            game3 += g3
            tp_sum += tp
            gt_sum += gt_count
            et_sum += et_count
 
    mae = mae/len(test_loader)
    mse = np.sqrt(mse/(len(test_loader)))
    game1 = game1/len(test_loader)
    game2 = game2/len(test_loader)
    game3 = game3/len(test_loader)
    ap = tp_sum / float(et_sum)
    ar = tp_sum / float(gt_sum)
    f1 = 2*ap*ar / (ap+ar)
        
    if epoch%1==0:
        log_text = ' * MAE {mae:.3f}--MSE {mse:.3f}--GAME1 {game1:.3f}--GAME2 {game2:.3f}--GAME3 {game3:.3f}--AP {ap:.3f}--AR {ar:.3f}--F1 {f1:.3f}'.format(mae=mae,mse=mse,game1=game1,game2=game2,game3=game3,ap=ap,ar=ar,f1=f1)
        log_print(log_text, color='yellow', attrs=['bold'])

    return mae 
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        
        scale = args.scales[i] if i < len(args.scales) else 1
        
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
if __name__ == '__main__':
    main()  