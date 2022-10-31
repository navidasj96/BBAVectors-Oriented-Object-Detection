import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from .model_parts import CombinationModule
from . import resnet
from .cem import CEM



class CTRBOX(nn.Module):
    def __init__(self, heads, pretrained, head_conv):
        super(CTRBOX, self).__init__()
        
        self.base_network = resnet.res2net50(pretrained=pretrained)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(320, head_conv, kernel_size=3, padding='same', bias=True),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=3, padding=1, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(320, head_conv, kernel_size=3, padding='same', bias=True),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=3, stride=1, padding='same', bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)


    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)
        final=CEM(x)

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(final)
            
        return dec_dict
