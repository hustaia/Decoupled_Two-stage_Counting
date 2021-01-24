import os
import random
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import scipy.io as io
import cv2
transform=transforms.Compose([
    transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),])

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, crop=True, scale=True, flip=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            root = root*4
            random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop = crop
        self.scale = scale
        self.flip = flip
        self.train = train
            
    def _crop(self, img, count_target, prob_target):        
        w, h = img.size
        crop_size = (384, 384)
        
        if h < crop_size[1]:
            img = transforms.ToTensor()(img)
            tmp_img = torch.zeros([3,crop_size[1],w])
            tmp_img[:,0:h,:] = img
            img = transforms.ToPILImage()(tmp_img).convert('RGB')
            
            tmp_target = np.zeros([crop_size[1],w])  
            tmp_target[0:h,:] = count_target
            count_target = tmp_target
            
            tmp_target = np.zeros([crop_size[1],w])  
            tmp_target[0:h,:] = prob_target
            prob_target = tmp_target
            
            h = crop_size[1]
            
        if w < crop_size[0]:
            img = transforms.ToTensor()(img)
            tmp_img = torch.zeros([3,h,crop_size[0]])
            tmp_img[:,:,0:w] = img
            img = transforms.ToPILImage()(tmp_img).convert('RGB')
            
            tmp_target = np.zeros([h,crop_size[1]])  
            tmp_target[:,0:w] = count_target
            count_target = tmp_target
            
            tmp_target = np.zeros([h,crop_size[1]])  
            tmp_target[:,0:w] = prob_target
            prob_target = tmp_target
            
            w = crop_size[0] 
            
        dx = int(random.random() * (w - crop_size[0]))
        dy = int(random.random() * (h - crop_size[1]))
        
        img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        count_target = count_target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]
        prob_target = prob_target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

        return img, count_target, prob_target
        
    def _scale(self, img, count_target, prob_target):
        if random.random() > 0.5:
            scale_factor = 0.75 + 0.15 * random.random()
            w, h = img.size
            w_new = int(w * scale_factor)
            h_new = int(h * scale_factor)
            
            img = img.resize((w_new, h_new),Image.ANTIALIAS)
            
            count_sum = count_target.sum()
            count_target = cv2.resize(count_target, (w_new, h_new), interpolation=cv2.INTER_CUBIC)
            count_target = count_target * count_sum / float(count_target.sum() + 1e-6)
            
            prob_max = prob_target.max()
            prob_target = cv2.resize(prob_target, (w_new, h_new), interpolation=cv2.INTER_CUBIC)
            prob_target = prob_target * prob_max / float(prob_target.max() + 1e-6)
        
        return img, count_target, prob_target

    def _align_train(self, img, count_target, prob_target):
        w, h = img.size 

        if w != 384 or h != 384:      
            img = transforms.ToTensor()(img)
            tmp_img = torch.zeros([3,384,384])
            tmp_img[:,0:h,0:w] = img
            img = transforms.ToPILImage()(tmp_img).convert('RGB')
            
            tmp_target = np.zeros([384,384])  
            tmp_target[0:h,0:w] = count_target
            count_target = tmp_target
            
            tmp_target = np.zeros([384,384])  
            tmp_target[0:h,0:w] = prob_target
            prob_target = tmp_target
        
        return img, count_target, prob_target
                
    def _align_test(self, img, count_target, prob_target):
        w, h = img.size 
        
        if w % 32 != 0 or h % 32 != 0:        
            rf = 32
            h1 = int((h + rf - 1) / rf) * rf
            w1 = int((w + rf - 1) / rf) * rf
            
            img = transforms.ToTensor()(img)
            tmp_img = torch.zeros([3,h1,w1])
            tmp_img[:,0:h,0:w] = img
            img = transforms.ToPILImage()(tmp_img).convert('RGB')
            
            tmp_target = np.zeros([h1,w1])  
            tmp_target[0:h,0:w] = count_target
            count_target = tmp_target
            
            tmp_target = np.zeros([h1,w1])  
            tmp_target[0:h,0:w] = prob_target
            prob_target = tmp_target
        
        return img, count_target, prob_target
    
    
    def _flip(self, img, count_target, prob_target):
        if random.random() > 0.8:
            count_target = np.fliplr(count_target)
            prob_target = np.fliplr(prob_target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)    

        return img, count_target, prob_target
            
    def __len__(self):
        return self.nSamples
        
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]

        img = Image.open(img_path).convert('RGB')

        count_path = img_path.replace('.jpg','.h5')
        prob_path = img_path.replace('.jpg','_probmap.h5')
        count_file = h5py.File(count_path, 'r')
        prob_file = h5py.File(prob_path, 'r')
        count_target = np.asarray(count_file['density'])
        prob_target = np.asarray(prob_file['density'])
                    
        if self.crop == True:
            img, count_target, prob_target = self._crop(img, count_target, prob_target)
            
        if self.scale == True:
            img, count_target, prob_target = self._scale(img, count_target, prob_target)
        
        if self.train == True:
            img, count_target, prob_target = self._align_train(img, count_target, prob_target)
        else:
            img, count_target, prob_target = self._align_test(img, count_target, prob_target)
                    
        if self.flip == True:
            img, count_target, prob_target = self._flip(img, count_target, prob_target)
            

        
        count_target = cv2.resize(count_target, (int(count_target.shape[1]), int(count_target.shape[0])), interpolation=cv2.INTER_CUBIC)
        count_target = torch.from_numpy(count_target).type(torch.FloatTensor).unsqueeze(0)
        
        prob_target = cv2.resize(prob_target, (int(prob_target.shape[1]), int(prob_target.shape[0])), interpolation=cv2.INTER_CUBIC)
        prob_target = torch.from_numpy(prob_target).type(torch.FloatTensor).unsqueeze(0)
        
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.train:                
            return img, count_target, prob_target
        else:
            dot_path = img_path.replace('.jpg','_dotmap.h5')
            dot_file = h5py.File(dot_path, 'r')
            dot_target = np.asarray(dot_file['density'])
            h,w = dot_target.shape 
        
            if w % 32 != 0 or h % 32 != 0:        
                rf = 32
                h1 = int((h + rf - 1) / rf) * rf
                w1 = int((w + rf - 1) / rf) * rf
            
                tmp_target = np.zeros([h1,w1])  
                tmp_target[0:h,0:w] = dot_target
                dot_target = tmp_target
                
            return img, count_target, prob_target, dot_target
        
