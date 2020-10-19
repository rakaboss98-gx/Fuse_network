#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:58:36 2020

@author: rakshitbhatt98
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from Fuse_Network import *

class Model(nn.Module):
    def __init__(self):
        
        super(Model, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=3,
                               out_channels=16, 
                               kernel_size=[3,3],
                               stride = 2,
                               padding = (1,1))
        self.Conv1_bn = nn.BatchNorm2d(16)
        
        self.fuse1 = Fuse_Net(channels = 16,
                              kernel_size = [3,1],
                              exp_size = 16,
                              out_size = 16,
                              SE = 1,
                              NL = 0,
                              stride = 2)
        
        self.fuse2 = Fuse_Net(channels = 16,
                              kernel_size = [3,1],
                              exp_size = 72,
                              out_size = 24,
                              SE = 0,
                              NL = 0,
                              stride = 2)
        
        self.fuse3 = Fuse_Net(channels = 24,
                              kernel_size = [3,1],
                              exp_size = 88,
                              out_size = 24,
                              SE = 0,
                              NL = 0,
                              stride = 1)
        
        self.fuse4 = Fuse_Net(channels = 24,
                              kernel_size = [5,1],
                              exp_size = 96,
                              out_size = 40,
                              SE = 1,
                              NL = 1,
                              stride = 2)
        
        self.fuse5 = Fuse_Net(channels = 40,
                              kernel_size = [5,1],
                              exp_size = 240,
                              out_size = 40,
                              SE = 1,
                              NL = 1,
                              stride = 1)
        
        self.fuse6 = Fuse_Net(channels = 40,
                              kernel_size = [5,1],
                              exp_size = 240,
                              out_size = 40,
                              SE = 1,
                              NL = 1,
                              stride = 1)
        
        self.fuse7 = Fuse_Net(channels = 40,
                              kernel_size = [5,1],
                              exp_size = 120,
                              out_size = 48,
                              SE = 1,
                              NL = 1,
                              stride = 1)
        
        self.fuse8 = Fuse_Net(channels = 48,
                              kernel_size = [5,1],
                              exp_size = 144,
                              out_size = 48,
                              SE = 1,
                              NL = 1,
                              stride = 1)
        
        self.fuse9 = Fuse_Net(channels = 48,
                              kernel_size = [5,1],
                              exp_size = 288,
                              out_size = 96,
                              SE = 1,
                              NL = 1,
                              stride = 2)
        
        self.fuse10 = Fuse_Net(channels = 96,
                              kernel_size = [5,1],
                              exp_size = 576,
                              out_size = 96,
                              SE = 1,
                              NL = 1,
                              stride = 2)
        
        self.fuse11 = Fuse_Net(channels = 96,
                              kernel_size = [5,1],
                              exp_size = 576,
                              out_size = 96,
                              SE = 1,
                              NL = 1,
                              stride = 2)
        
        self.Conv2 = nn.Conv2d(in_channels=96,
                               out_channels=576, 
                               kernel_size=1,
                               stride = 1)
        self.Conv2_bn = nn.BatchNorm2d(576)
        
        self.adapt_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.Conv3 = nn.Conv2d(in_channels=576,
                               out_channels=1024, 
                               kernel_size=1,
                               stride = 1)
        
        self.drop = torch.nn.Dropout2d(p=0.2)
        
        self.Conv4 = nn.Conv2d(in_channels=1024,
                               out_channels=100, 
                               kernel_size=1,
                               stride = 1)
        
    def _initialize_weights(self):                                                                                        
    # weight initialization                                                                                                  
        for m in self.modules():                                                                                                 
            if isinstance(m, nn.Conv2d):                                                                                         
                nn.init.kaiming_normal_(m.weight, mode='fan_out')                                                                
                if m.bias is not None:                                                                                           
                    nn.init.zeros_(m.bias)                                                                                       
            elif isinstance(m, nn.BatchNorm2d):                                                                                  
                nn.init.ones_(m.weight)                                                                                          
                nn.init.zeros_(m.bias)                                                                                           
            elif isinstance(m, nn.Linear):                                                                                       
                nn.init.normal_(m.weight, 0, 0.01)                                                                               
                if m.bias is not None:                                                                                           
                    nn.init.zeros_(m.bias)
        
    def forward2(self, x):
        x = self.Conv1(x)
        x = self.Conv1_bn(x)
        hswish = Hswish()
        x = hswish.forward(x)
        
        x = self.fuse1.forward(x)
        x = self.fuse2.forward(x)
        x = self.fuse3.forward(x)
        x = self.fuse4.forward(x)
        x = self.fuse5.forward(x)
        x = self.fuse6.forward(x)
        x = self.fuse7.forward(x)
        x = self.fuse8.forward(x)
        x = self.fuse9.forward(x)
        x = self.fuse10.forward(x)
        x = self.fuse11.forward(x)
        x = self.Conv2(x)
        x = self.Conv2_bn(x)
        hswish = Hswish()
        x = hswish.forward(x)
        x = self.adapt_avg_pool(x)
        x = self.Conv3(x)
        x = self.drop(x)
        x = self.Conv4(x)
        
        b = x.size()[0]
        x = torch.reshape(x, (b, 100))
        return(x)
            
            
            
        
        
        
        
        
        
        
        
    



