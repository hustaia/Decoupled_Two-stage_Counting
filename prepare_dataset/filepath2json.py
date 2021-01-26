import os
import glob
import sys
from sys import argv
import json
#use like: python filepath2json.py ./data/shanghaitech/part_A_final/train_data/images/ part_A_train.json
if len(sys.argv)!=3:
    print("please enter path and json name")
else:
    paths = []
    for i in range(len(os.listdir(sys.argv[1]))):
        for img_path in glob.glob(os.path.join(sys.argv[1], os.listdir(sys.argv[1])[i])):
            if img_path.split('.')[-1] == 'jpg':
                paths.append(img_path)
    json_res = json.dumps(paths)
    with open(sys.argv[2],'w') as fp:
        fp.write(json_res)

