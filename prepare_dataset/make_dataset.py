import glob

import scipy
import scipy.io as io
import scipy.spatial
from matplotlib import cm as CM
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

import os
from PIL import Image
import numpy as np
import h5py
import cv2
#get_ipython().run_line_magic('matplotlib', 'inline')

#this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density
    
    
def gaussian_filter_prob(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        filter = scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        peak = filter[pt[1]][pt[0]]
        density_new = filter / float(peak)
        density = np.maximum(density_new, density)
    print('done.')
    return density
    
def gaussian_filter_prob_fixed(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        sigma =15
        filter = scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        peak = filter[pt[1]][pt[0]]
        density_new = filter / float(peak)
        density = np.maximum(density_new, density)
    print('done.')
    return density

#parta
'''
root = "/data0/dataset/chengjian/shanghaitech/"

part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
path_sets = [part_A_train,part_A_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    print(img_path)
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG','GT_IMG'))
    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    k1 = gaussian_filter_density(k)
    k2 = gaussian_filter_prob(k)
    with h5py.File(img_path.replace('.jpg','_dotmap.h5'), 'w') as hf:
        hf['density'] = k
    with h5py.File(img_path.replace('.jpg','.h5'), 'w') as hf:
        hf['density'] = k1
    with h5py.File(img_path.replace('.jpg','_probmap.h5'), 'w') as hf:
        hf['density'] = k2
'''


#partb    
'''
root = "/data0/dataset/chengjian/shanghaitech/"

part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
path_sets = [part_B_train,part_B_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    print(img_path)
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG','GT_IMG'))
    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    k1 = gaussian_filter(k, 15)
    k2 = gaussian_filter_prob_fixed(k)
    with h5py.File(img_path.replace('.jpg','_dotmap.h5'), 'w') as hf:
        hf['density'] = k
    with h5py.File(img_path.replace('.jpg','.h5'), 'w') as hf:
        hf['density'] = k1
    with h5py.File(img_path.replace('.jpg','_probmap.h5'), 'w') as hf:
        hf['density'] = k2
'''


#qnrf
'''
root = "/data0/dataset/chengjian/UCF_QNRF_normalized/"

qnrf_train = os.path.join(root,'Train')
qnrf_test = os.path.join(root,'Test')
path_sets = [qnrf_train,qnrf_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    print(img_path)
    mat = io.loadmat(img_path.replace('.jpg','_ann.mat'))
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["annPoints"]
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    k1 = gaussian_filter_density(k)
    k2 = gaussian_filter_prob(k)
    with h5py.File(img_path.replace('.jpg','_dotmap.h5'), 'w') as hf:
        hf['density'] = k
    with h5py.File(img_path.replace('.jpg','.h5'), 'w') as hf:
        hf['density'] = k1
    with h5py.File(img_path.replace('.jpg','_probmap.h5'), 'w') as hf:
        hf['density'] = k2
'''


#synthesized_dataset_parta
'''
path = "/data0/dataset/chengjian/virtual_dataset/partA/dotmap"
path_new = path.replace('dotmap','probmap')
if  not os.path.exists(path_new):
    os.makedirs(path_new)
path_new = path.replace('dotmap','densitymap')
if  not os.path.exists(path_new):
    os.makedirs(path_new)

img_paths = []
for img_path in glob.glob(os.path.join(path, '*.mat')):
    img_paths.append(img_path)

for img_path in img_paths:
    print(img_path)
    k = io.loadmat(img_path)
    k = k['dotmap'] / 255.
    k1 = gaussian_filter_prob(k)
    k2 = gaussian_filter_density(k)
    with h5py.File(img_path.replace('.mat','.h5'), 'w') as hf:
        hf['density'] = k
    with h5py.File(img_path.replace('.mat','.h5').replace('dotmap','probmap'), 'w') as hf:
        hf['density'] = k1
    with h5py.File(img_path.replace('.mat','.h5').replace('dotmap','densitymap'), 'w') as hf:
        hf['density'] = k2
'''


#synthesized_dataset_partb
'''
path = "/data0/dataset/chengjian/virtual_dataset/partB/dotmap"
path_new = path.replace('dotmap','probmap')
if  not os.path.exists(path_new):
    os.makedirs(path_new)
path_new = path.replace('dotmap','densitymap')
if  not os.path.exists(path_new):
    os.makedirs(path_new)

img_paths = []
for img_path in glob.glob(os.path.join(path, '*.mat')):
    img_paths.append(img_path)

for img_path in img_paths:
    print(img_path)
    k = io.loadmat(img_path)
    k = k['dotmap'] / 255.
    k1 = gaussian_filter_prob_fixed(k)
    k2 = gaussian_filter(k, 15)
    with h5py.File(img_path.replace('.mat','.h5'), 'w') as hf:
        hf['density'] = k
    with h5py.File(img_path.replace('.mat','.h5').replace('dotmap','probmap'), 'w') as hf:
        hf['density'] = k1
    with h5py.File(img_path.replace('.mat','.h5').replace('dotmap','densitymap'), 'w') as hf:
        hf['density'] = k2
'''