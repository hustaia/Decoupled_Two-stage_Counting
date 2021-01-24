import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
from matplotlib import pyplot as plt
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM

from VGG16_Unet import *
from TasselNetv2_VGG16 import *

import torch
import utils
import cv2

from torchvision import datasets, transforms
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

PMR = VGG16_Unet()
CMR = TasselNet_VGG16()
PMR = PMR.cuda()
CMR = CMR.cuda()

checkpoint = torch.load("/data0/dataset/chengjian/PycharmProjects/test_rec_probmap/56.715_96.698.pth.tar")

PMR.load_state_dict(checkpoint['state_dict1'])
PMR.eval()
CMR.load_state_dict(checkpoint['state_dict2'])
CMR.eval()

img_path = "/data0/dataset/chengjian/shanghaitech/part_A_final/test_data/images/IMG_99.jpg"
img = Image.open(img_path).convert('RGB')

w,h = img.size
img_transformed = transform(img)

rf = CMR.rf
H = int((h + rf - 1) / rf) * rf
W = int((w + rf - 1) / rf) * rf

img1 = torch.zeros([3,H,W])
img1[:,0:h,0:w] = img_transformed[:,0:h,0:w]

img1 = img1.unsqueeze(0).cuda()
with torch.no_grad():
    probmap = PMR(img1)
    countmap = CMR(probmap)

    avg_pooled_probmap = nn.functional.avg_pool2d(probmap,3,stride=1,padding=1)
    max_pooled_probmap = nn.functional.max_pool2d(avg_pooled_probmap,3,stride=1,padding=1)
    candidate_peak = torch.where(avg_pooled_probmap==max_pooled_probmap, avg_pooled_probmap, torch.full_like(probmap, 0))
    
    countmap = countmap.detach().cpu().numpy().squeeze()
    probmap = probmap.detach().cpu().numpy().squeeze()
    dotmap = np.zeros([H,W]) 
    cnt = np.zeros([H//rf,W//rf]) 
    
    left = []
    right = []
    up = []
    down = []   
    
    for y in range(0,H//rf-1,1):
        if np.all(cnt):
            break
        for x in range(0,W//rf-1,1):
            if np.all(cnt[y+1,:]):
                break
            flag = 0
            for ii in range(2,min(H//rf-y, W//rf-x)):
                if countmap[y:y+ii,x:x+ii].sum()>1:
                    flag = 1
                    left.append(x*rf)
                    right.append((x+ii)*rf)
                    up.append(y*rf)
                    down.append((y+ii)*rf)
                    cnt[y:y+ii,x:x+ii] += 1.0
                    break
            if flag == 0:
                left.append(x*rf)
                right.append((x+min(H//rf-y, W//rf-x))*rf)
                up.append(y*rf)
                down.append((y+min(H//rf-y, W//rf-x))*rf)
                cnt[y:y+min(H//rf-y, W//rf-x),x:x+min(H//rf-y, W//rf-x)] += 1.0
          
    for ii in range(len(left)):
        sum = int(round(countmap[up[ii]//rf:down[ii]//rf,left[ii]//rf:right[ii]//rf].sum()))
        if sum <= 0:
            continue
        tmp_peak = candidate_peak.clone()
        for jj in range(sum):
            argmax = tmp_peak[0,0,up[ii]:down[ii],left[ii]:right[ii]].argmax()    
            hh = argmax // (right[ii]-left[ii])
            ww = argmax % (right[ii]-left[ii])
            dotmap[up[ii]+hh,left[ii]+ww] += 1
            tmp_peak[0,0,up[ii]+hh,left[ii]+ww] = 0

    cnt = cv2.resize(cnt, (W,H), interpolation=cv2.INTER_NEAREST)
    dotmap = np.divide(dotmap,cnt)
    dotmap = np.where(dotmap>0.5,1,0)
        
print(img_path.split('/')[-1].split('.')[0], countmap.sum(), dotmap.sum())

plt.clf()
plt.subplot(221)
plt.imshow(img)
plt.subplot(222)
plt.imshow(probmap)
plt.subplot(223)
plt.imshow(countmap)
plt.subplot(224)
plt.axis('scaled')
plt.xlim(0, W)
plt.ylim(0, H)
plt.scatter(np.nonzero(dotmap)[1],H-1-np.nonzero(dotmap)[0],s=4,marker='+')
plt.pause(10) 
