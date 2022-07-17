from turtle import forward
import torch 
import torch.nn as nn
from vision_token import ViTResNet, BasicBlock
from downSample import Encoder
import torch.nn.functional as F

class LENet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extract = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=7,padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 32, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 32, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 32, kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(kernel_size=7,stride=7)
    
    def forward(self, x):
        x = self.feature_extract(x)
        x = self.pool(x)
        return x.view(x.shape[0],-1)
        
class VTT-IQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.STNet = STNet(BasicBlock, [3, 3, 3])
        self.LENet = LENet()
        self.nn1 = nn.Linear(256,128)
        self.nn2 = nn.Linear(128,1)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std = 1e-6)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x,y):
        x = torch.cat((x,y),dim=1)
        x = F.relu(self.nn1(x))
        x = self.dropout(x)
        x = self.nn2(x)
        return x


