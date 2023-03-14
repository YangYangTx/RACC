from logging import raiseExceptions
import torch 
import torch.nn as nn
import torchvision
from Models.se_resnet import se_resnet50

class Att(nn.Module):
    def __init__(self):
        super(Att, self).__init__()

        self.conv = nn.Conv2d(2, 1, 3, padding=3 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class Head(torch.nn.Module):
    def __init__(self, feats_dims, **kwargs):
        super(Head, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        out_num = kwargs.get("out_num", 10)
        self.fc_out = nn.Sequential(nn.Linear(feats_dims[-1], 1024),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(1024, out_num))
        self.sa = Att()

    def forward(self, x):
        x = x[-1]  #  b,c,h1,w1
        x = self.sa(x) * x
        x = self.avg_pool(x)  # b,c,1,1
        x = torch.flatten(x, 1)  # b,c
        x = self.fc_out(x)  # b,1
        return x

class bqvsModel_res50(nn.Module):
    def __init__(self, model_name, num_classes, is_pretrained=True):
        super(bqvsModel, self).__init__()
        self.model_name = model_name
        self.num_class = num_classes
        self.is_pretrained = is_pretrained
        if self.is_pretrained:
            self.base_model = getattr(torchvision.models, self.model_name)(True)
      
        if hasattr(self.base_model, 'fc'):
            self.base_model.last_layer_name = 'fc'
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
            self.base_model.fc = nn.Linear(feature_dim, 10)
            #self.fc2 = nn.Linear(768, 10)

        elif hasattr(self.base_model, 'heads'):
            self.base_model.last_layer_name = 'heads'
            
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).head.in_features
            self.base_model.heads = nn.Linear(feature_dim, self.num_class)
    
        else:
            print ("check model")

class bqvsModel(nn.Module):
    def __init__(self, model_name, num_classes, is_pretrained=True):
        super(bqvsModel, self).__init__()
        self.num_classes = num_classes
        
        self.base_model = se_resnet50(num_classes=1000)
        self.base_model.fc = nn.Linear(2048, self.num_classes)

    def forward(self, x):
        x = self.base_model(x)
        #x = self.fc2(x)
        #x = self.head(x)
        #x = out.view(n, -1)
        return x
if __name__ == "__name__":
    net = bqvsModel('vit_b_16', 10, is_pretrained=True)
