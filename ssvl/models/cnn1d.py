import torch
import torch.nn as nn
from .modules import Conv_1d
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, in_channel=6, out_channel=32, n_class=3):
        super(CNN1D, self).__init__()
        self.layer1 = Conv_1d(in_channel, out_channel, shape=7, stride=7, pooling=1)
        self.layer2 = Conv_1d(out_channel, out_channel, shape=3, stride=1, pooling=3)
        self.layer3 = Conv_1d(out_channel, out_channel, shape=3, stride=1, pooling=3)
        self.layer4 = Conv_1d(out_channel, out_channel, shape=3, stride=1, pooling=3)
        self.layer5 = Conv_1d(out_channel, out_channel, shape=3, stride=1, pooling=3)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(out_channel, n_class)

    def forward(self, x, y):
        x = self.layer1(x)        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)        
        x = self.layer5(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.dense(x)
        loss = F.cross_entropy(x, y)
        return loss

    def inference(self, x):
        x = self.layer1(x)        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)        
        x = self.layer5(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.dense(x)
        return x