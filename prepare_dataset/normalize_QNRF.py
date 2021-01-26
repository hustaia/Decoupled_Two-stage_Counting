import os
import glob
import scipy.io as io
from PIL import Image

                                         
for path in ['../data/UCF_QNRF/Train/', '../data/UCF_QNRF/Test/']:
    path_new = path.replace('UCF_QNRF','UCF_QNRF_normalized')
    if  not os.path.exists(path_new):
        os.makedirs(path_new)
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        print(img_path) 
        img = Image.open(img_path).convert('RGB')        
        w, h = img.size
        mat_path = img_path.replace('.jpg','_ann.mat')
        mat = io.loadmat(mat_path)
        gt = mat["annPoints"] 
        if w > 1920 or h > 1920:
            factor = min(1920 / float(w), 1920 / float(h))
            h_new = int(h * factor)
            w_new = int(w * factor)
            img_new = img.resize((w_new, h_new), Image.ANTIALIAS)
            gt_new = gt * factor
        else:
            img_new = img
            gt_new = gt
        img_path_new = img_path.replace('UCF_QNRF','UCF_QNRF_normalized')
        img_new.save(img_path_new, quality=100)    
        mat_path_new = mat_path.replace('UCF_QNRF','UCF_QNRF_normalized')
        io.savemat(mat_path_new, {'annPoints': gt_new}) 