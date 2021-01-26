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

checkpoint = torch.load("./partA.pth.tar")

PMR.load_state_dict(checkpoint['state_dict1'])
PMR.eval()
CMR.load_state_dict(checkpoint['state_dict2'])
CMR.eval()

img_path = "./data/shanghaitech/part_A_final/test_data/images/IMG_100.jpg"
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
        
print(img_path.split('/')[-1].split('.')[0], countmap.sum(), dotmap.sum())

probmap = probmap.detach().squeeze().cpu().numpy()
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
