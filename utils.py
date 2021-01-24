import h5py
import torch
import shutil
import numpy as np
import cv2
import os

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)


def save_checkpoint(state, is_best, task_id, filename='checkpoint.pth.tar'):
    torch.save(state, task_id + filename)
    if is_best:
        shutil.copyfile(task_id + filename, task_id + 'model_best.pth.tar')
        
        
def save_results(gt_data,density_map,output_dir, fname='results.png'):
    gt_data = 255*gt_data/np.max(gt_data)
    density_map = 255*density_map/np.max(density_map)
    #print(gt_data.shape)
    #print(density_map.shape)
    if density_map.shape[1] != gt_data.shape[1] or density_map.shape[0] != gt_data.shape[0]:
        density_map = cv2.resize(density_map, (gt_data.shape[1],gt_data.shape[0])) 
    result_img = np.hstack((gt_data,density_map))
    cv2.imwrite(os.path.join(output_dir,fname),result_img)
    
     
def display_results(gt_data,density_map):
    gt_data = 255*gt_data/np.max(gt_data)
    density_map = 255*density_map/np.max(density_map)
    if density_map.shape[1] != gt_data.shape[1] or density_map.shape[0] != gt_data.shape[0]:
        density_map = cv2.resize(density_map, (gt_data.shape[1],gt_data.shape[0])) 
    result_img = np.hstack((gt_data,density_map))
    result_img  = result_img.astype(np.uint8, copy=False)
    cv2.imshow('Result', result_img)
    #cv2.waitKey(0)