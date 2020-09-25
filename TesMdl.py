#
import torch
import torch.nn as nn

#
class ClsMdl(nn.Module):

    def __init__(self):
        super(ClsMdl, self).__init__()
        #
        mdl_net = []
        # block 1   input:3*80*60 -> output:64*40*30
        mdl_net.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))
        mdl_net.append(nn.ReLU())
        mdl_net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        mdl_net.append(nn.ReLU())
        mdl_net.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # block 2   input:64*40*30 -> output:128*20*15
        mdl_net.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        mdl_net.append(nn.ReLU())
        mdl_net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        mdl_net.append(nn.ReLU())
        mdl_net.append(nn.MaxPool2d(kernel_size=2, stride=2))
