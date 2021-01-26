import torch.nn as nn
import torch.nn.init as init
import torch
import numpy as np
from torchvision import models
import torch.utils.model_zoo as model_zoo

class VGG16_Unet(nn.Module):
    
    def __init__(self):
        super(VGG16_Unet, self).__init__()
        
        
        vggfeat = list(models.vgg16(pretrained = False).features)
        self.feature1 = nn.Sequential(*vggfeat[0:4])
        self.feature2 = nn.Sequential(*vggfeat[4:9])
        self.feature3 = nn.Sequential(*vggfeat[9:16])
        self.feature4 = nn.Sequential(*vggfeat[16:23])
        self.feature5 = nn.Sequential(*vggfeat[23:30])
        del vggfeat
        
        self.decode1 = nn.Sequential(Conv2d(128, 64, kernel_size = 3, padding = 1))
        self.decode2 = nn.Sequential(Conv2d(256, 128, kernel_size = 3, padding = 1),
                                      Conv2d(128, 64, kernel_size = 3, padding = 1))
        self.decode3 = nn.Sequential(Conv2d(512, 256, kernel_size = 3, padding = 1),
                                      Conv2d(256,128, kernel_size = 3, padding = 1))
        self.decode4 = nn.Sequential(Conv2d(1024, 512, kernel_size = 3, padding = 1),
                                      Conv2d(512, 256, kernel_size = 3, padding = 1))
                                
        self.prob_conv = nn.Sequential(Conv2d(64, 1, kernel_size = 3, padding = 1))  
        
        self.initialize_weights()
        mod = models.vgg16(pretrained = True)

        len1 = len(self.feature1.state_dict().items())
        len2 = len1 + len(self.feature2.state_dict().items())
        len3 = len2 + len(self.feature3.state_dict().items())
        len4 = len3 + len(self.feature4.state_dict().items())
                                
        for i in range(len(self.feature1.state_dict().items())):
            list(self.feature1.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
        for i in range(len(self.feature2.state_dict().items())):
            list(self.feature2.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i+len1][1].data[:]
        for i in range(len(self.feature3.state_dict().items())):
            list(self.feature3.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i+len2][1].data[:]
        for i in range(len(self.feature4.state_dict().items())):
            list(self.feature4.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i+len3][1].data[:]
        for i in range(len(self.feature5.state_dict().items())):
            list(self.feature5.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i+len4][1].data[:]
            
        del mod        

        
    def forward(self, im_data):

        feature1 = self.feature1(im_data)
        feature2 = self.feature2(feature1)
        feature3 = self.feature3(feature2)
        feature4 = self.feature4(feature3)
        feature5 = self.feature5(feature4)        
        
        up_feature5 = nn.functional.interpolate(feature5, scale_factor = 2, mode = "bilinear", align_corners = True)
        cat_feature4 = torch.cat((feature4, up_feature5), 1)
        de_feature4 = self.decode4(cat_feature4)
        del feature5, up_feature5, feature4, cat_feature4
        
        up_feature4 = nn.functional.interpolate(de_feature4, scale_factor = 2, mode = "bilinear", align_corners = True)
        cat_feature3 = torch.cat((feature3, up_feature4), 1)
        de_feature3 = self.decode3(cat_feature3)
        del de_feature4, up_feature4, feature3, cat_feature3
        
        up_feature3 = nn.functional.interpolate(de_feature3, scale_factor = 2, mode = "bilinear", align_corners = True)
        cat_feature2 = torch.cat((feature2, up_feature3), 1)
        de_feature2 = self.decode2(cat_feature2)
        del de_feature3, up_feature3, feature2, cat_feature2
        
        up_feature2 = nn.functional.interpolate(de_feature2, scale_factor = 2, mode = "bilinear", align_corners = True)
        cat_feature1 = torch.cat((feature1, up_feature2), 1)
        de_feature1 = self.decode1(cat_feature1)        
        del de_feature2, up_feature2, feature1, cat_feature1
            
        prob_map = self.prob_conv(de_feature1)
        
        #prob_map = torch.clamp(prob_map,0,1)
        
        return prob_map

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
                
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, relu=True, bn=False):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
        
        