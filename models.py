import torch, torchvision, torch.nn as nn
from torch import Tensor


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

      
class Resnet(nn.Module):
    def __init__(self, model=torchvision.models.resnet18, pretrained: bool = True):
        super().__init__()
        self.net = model(pretrained=pretrained)

        for name, param in self.net.named_parameters():
            if 'layer2' in name:
                break
            param.requires_grad = False
        
        del self.net.avgpool
        del self.net.fc
    
    def forward(self, x: Tensor) -> Tensor:
      """Even deleting avgpool and fc we must override forward function so flatten is not applied"""
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        # x = self.net.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.net.fc(x)

        return x
      
"""Importing Juefei-Xu's LBCNN implementation. Available in https://github.com/juefeix/lbcnn.pytorch"""
from drive.MyDrive.IC import juefei_lbcnn_resnet as juefei_lbcnn_resnet

class LBCResnet(nn.Module):
    def __init__(self, model=juefei_lbcnn_resnet.resnet18, pretrained: bool = False) -> None:
        super().__init__()
        self.net = model(pretrained=pretrained)
        # self.net.avgpool = Identity()
        # self.net.fc = Identity()
        del self.net.fc, self.net.avgpool
    
    def forward(self, x: Tensor) -> Tensor:
        """Even deleting avgpool and fc we must override forward function so flatten is not applied"""
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        # x = self.net.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.net.fc(x)

        return x
      

class MixedNet(nn.Module):
    def __init__(self, l1=1024, l2=512):
        super(MixedNet, self).__init__()
        self.res = Resnet(model=torchvision.models.resnet18, pretrained=True)
        self.lbc = LBCResnet(model=juefei_lbcnn_resnet.resnet18, pretrained=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(1024)
        )

    
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.res(x), self.lbc(x)
        x = torch.cat((x1,x2), 1)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
