# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:39:07 2021

@author: Qing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class generator(nn.Module):
    # initializers
    def __init__(self,params, out_channel=2):
        super().__init__()
        factor = params['factor']
        siz_latent = params['siz_l']
        d = params['gen_base_size']
        self.gen_reg = params['gen_reg']
        self.nlevels = int(9-np.log2(factor)) 
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(siz_latent, 100, 1, 1, 0),
            #nn.BatchNorm2d(100),
            nn.Tanh(), # Finish Layer 1 --> 1x1
            nn.Conv2d(100, d*8, 3, 1, 2),
            #nn.BatchNorm2d(d*8),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 2 --> 3x3
            nn.Conv2d(d*8, d*8, 3, 1, 2),
            #nn.BatchNorm2d(d*8),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 3 --> 5x5
            nn.Upsample(scale_factor=2, mode='nearest'), # NN --> 10x10
            nn.Conv2d(d*8, d*4, 3, 1, 1),
            #nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 4 --> 10x10
            nn.Upsample(scale_factor=2, mode='nearest'), # NN --> 20x20
            nn.Conv2d(d*4, d*4, 3, 1, 1),
            #nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 5 --> 20x20
            nn.Upsample(scale_factor=2, mode='nearest'), # NN --> 40x40
            nn.Conv2d(d*4, d*4, 3, 1, 2),
            #nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 6 --> 42x42
            nn.Upsample(scale_factor=2, mode='nearest'), # NN --> 84x84
            nn.Conv2d(d*4, d*2, 4, 1, 2),
            #nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 7 --> 85x85
            nn.Upsample(scale_factor=2, mode='nearest'), # NN --> 170x170
            nn.Conv2d(d*2, d, 3, 1, 1),
            #nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 8 --> 170x170
            nn.Upsample(scale_factor=2, mode='nearest'), # NN --> 340x340
            nn.Conv2d(d, d, 3, 1, 1),
            #nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 9 --> 340x340
            nn.Conv2d(d, out_channel, 3, 1, 1),
            #nn.BatchNorm2d(out_channel),
            nn.Tanh(), # Finish Layer 10 --> 340x340
            )
        

    def weight_init(self):
        for layer in self._modules:
            if layer.find('Conv') != -1:
                layer.weight.data.normal_(0.0, 0.02)
            elif layer.find('BatchNorm2d') != -1:
                layer.weight.data.normal_(1.0, 0.02)
                layer.bias.data.fill_(0)


    # forward method
    def forward(self, input):
        x = self.conv_blocks(input)
        return x
#%%        
    def weightl1norm(self):
        L1norm = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                L1norm = L1norm + torch.norm(param, 1)
        return(self.gen_reg*L1norm)