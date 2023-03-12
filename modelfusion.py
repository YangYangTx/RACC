from logging import raiseExceptions
import torch 
import torch.nn as nn
import torchvision


class bqvsModelfusion(nn.Module):
    def __init__(self, model_name, model_name2, num_classes, is_pretrained=True):
        super(bqvsModelfusion, self).__init__()
        self.model_name = model_name
        self.num_class = num_classes
        base_model1 = getattr(torchvision.models, self.model_name)(True)
        print(base_model1)
        self.model1 = nn.Sequential(*list(base_model1.children())[:-1])
        
        base_model2 = getattr(torchvision.models, self.model_name)(True)
        self.model2 = nn.Sequential(*list(base_model2.children())[:-1])
        
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
     
        self.fc1 = torch.nn.Linear(4096, 4096)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in')
        self.fc2 = nn.Linear(4096, 10)
        
    def forward(self, image1, image2):
        x1 = self.model1(image1)
        x2 = self.model2(image2)
        fusionfea = torch.cat((x1, x2), dim=1).squeeze(-1).squeeze(-1)
        x = self.fc1(fusionfea)
        x = self.dropout(x)
        x = self.relu(x) 
        x = self.fc2(x)  
       
        return x

