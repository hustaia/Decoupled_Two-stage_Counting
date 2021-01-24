import torch.nn as nn
import torch
from torchvision import models

import torch.nn.functional as F
import math


def Gauss_initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)   

# --define base Netweork module
class VGG16_frontend(nn.Module):
    def __init__(self,block_num=5,decode_num=4,load_weights=True,bn=False,IF_freeze_bn=False):
        super(VGG16_frontend,self).__init__()
        self.block_num = block_num
        self.load_weights = load_weights
        self.bn = bn
        self.IF_freeze_bn = IF_freeze_bn
        self.decode_num = decode_num

        block_dict = [[64, 64], ['M',128, 128], ['M',256, 256, 256],\
             ['M',512, 512, 512], ['M',512, 512, 512,'M']]

        self.frontend_feat = []
        for i in range(block_num):
            self.frontend_feat += block_dict[i]

        if self.bn:
            self.features = make_layers(self.frontend_feat, in_channels = 1, batch_norm=True)
        else:
            self.features = make_layers(self.frontend_feat, in_channels = 1, batch_norm=False)


        if self.load_weights:
            if self.bn:
                pretrained_model = models.vgg16_bn(pretrained = True)
            else:
                pretrained_model = models.vgg16(pretrained = True)
            pretrained_dict = pretrained_model.state_dict()
            model_dict = self.state_dict()
            # filter out unnecessary keys
            pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in model_dict and k != 'features.0.weight'}                    
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict_1) 
            # load the new state dict
            self.load_state_dict(model_dict)
            weights = (list(pretrained_dict.items())[0][1].data[:,0,:,:] + \
                      list(pretrained_dict.items())[0][1].data[:,1,:,:] + \
                      list(pretrained_dict.items())[0][1].data[:,2,:,:]) / 3.
            weights = weights.unsqueeze(1)
            list(self.features.state_dict().items())[0][1].data[:] = weights

        if IF_freeze_bn:
            self.freeze_bn()

    def forward(self,x):
        if self.bn: 
            x = self.features[ 0: 6](x)
            conv1_feat =x# if self.decode_num>=4 else []
            x = self.features[ 6:13](x)
            conv2_feat =x# if self.decode_num>=3 else []
            x = self.features[ 13:23](x)
            conv3_feat =x# if self.decode_num>=2 else []
            x = self.features[ 23:33](x)
            conv4_feat =x# if self.decode_num>=1 else []
            x = self.features[ 33:43+1](x)
            conv5_feat =x 
        else:
            x = self.features[ 0: 4](x)
            conv1_feat =x# if self.decode_num>=4 else []
            x = self.features[ 4:9](x)
            conv2_feat =x# if self.decode_num>=3 else []
            x = self.features[ 9:16](x)
            conv3_feat =x# if self.decode_num>=2 else []
            x = self.features[ 16:23](x)
            conv4_feat =x# if self.decode_num>=1 else []
            x = self.features[ 23:30+1](x)
            conv5_feat =x 
               
        feature_map = {'conv1':conv1_feat,'conv2': conv2_feat,\
            'conv3':conv3_feat,'conv4': conv4_feat, 'conv5': conv5_feat}   
        # feature_map = feature_map['conv5']
        return feature_map


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class TasselNet_VGG16(nn.Module):
    def __init__(self,frontend_name='VGG16',block_num=5,\
        IF_pre_bn=False,IF_freeze_bn=True,load_weights=True):
        super(TasselNet_VGG16, self).__init__()

        # init parameters
        self.frontend_name = frontend_name
        self.block_num = block_num

        self.IF_pre_bn = IF_pre_bn
        self.IF_freeze_bn = IF_freeze_bn
        self.load_weights = load_weights
        self.rf = 32

        # first, make frontend
        if self.frontend_name == 'VGG16':
            self.frontend = VGG16_frontend(block_num=self.block_num,decode_num=0,\
                load_weights=self.load_weights,bn=self.IF_pre_bn,IF_freeze_bn=self.IF_freeze_bn)
             
        # --predict prob_map with conv5
        self.backend_reg = torch.nn.Sequential(
                torch.nn.Conv2d(512, 512, (1, 1) ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 1, (1, 1) ) ) 

        Gauss_initialize_weights(self.backend_reg) 


    def forward(self,x):
        x = self.frontend(x)
        x = self.backend_reg(x['conv5'])
        return x

if __name__ == '__main__':
    mod = models.vgg16(pretrained = True)
    weights = torch.zeros([64, 1, 3, 3])
    weights[:,0,:,:] = (list(mod.state_dict().items())[0][1].data[:,:,:,0] +\
                      list(mod.state_dict().items())[0][1].data[:,:,:,1] +\
                      list(mod.state_dict().items())[0][1].data[:,:,:,2]) / 3.
    print(list(mod.state_dict().items())[0][1].data[:])
    print(weights.shape)