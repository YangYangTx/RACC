from logging import raiseExceptions
import torch 
import torch.nn as nn
import torchvision

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class SELayer(nn.Module):
    def __init__(self, channel):
        super(SELayer, self).__init__()
        self.fc1 = nn.Linear(channel, channel)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in')

    def forward(self, x):
        y = self.fc1(x)
        y = torch.sigmoid(y)
        return torch.mul(x, y)




class bqvsModelfusion_trm(nn.Module):
    def __init__(self, model_name, model_name2, num_classes, is_pretrained=True):
        super(bqvsModelfusion, self).__init__()
        self.model_name = model_name
        self.num_class = num_classes
        
        # ===================
        """
        base_model1 = se_resnet50(num_classes=1000)
        self.model1 = nn.Sequential(*list(base_model1.children())[:-1])
        
        base_model2 = se_resnet50(num_classes=1000)
        self.model2 = nn.Sequential(*list(base_model2.children())[:-1])
        """
        # =====================
        base_model1 = getattr(torchvision.models, self.model_name)(True)
        self.model1 = torch.nn.Sequential(*list(base_model1.children())[:-2])
        base_model2 = getattr(torchvision.models, self.model_name)(True)
        self.model2 = torch.nn.Sequential(*list(base_model2.children())[:-2])

        self.feature_dim = 2048
        self.max_length = 49
        self.total_length = self.max_length * 2
        self.fusion_layers = 2
        self.image_cls = nn.Parameter(torch.randn(1, 1, self.feature_dim))
        self.text_cls = nn.Parameter(torch.randn(1, 1, self.feature_dim))

        self.position_embeddings = nn.Embedding(self.total_length, self.feature_dim)
        self.register_buffer("position_ids", torch.arange(self.total_length).expand((1, -1)))
        self.modality_embeddings = nn.Embedding(2, self.feature_dim)
        self.register_buffer("modality_ids", torch.cat(
            [torch.zeros(self.max_length), torch.ones(self.max_length)]).long().expand(1, -1))
        
        self.ln = nn.LayerNorm(self.feature_dim)
        self.dropout = nn.Dropout(0.1)
        layer = nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=8)
        self.attn = nn.TransformerEncoder(layer, self.fusion_layers)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(self.feature_dim, self.feature_dim)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in')
     
        self.fc2 = nn.Linear(self.feature_dim, 10)

    def forward(self, image1, image2):
        """
        x1 = self.model1(image1) 
        x2 = self.model2(image2) 
        fusionfea = torch.cat((x1, x2), dim=1).squeeze(-1).squeeze(-1)
        x = self.fc1(fusionfea)
        x = self.bn1(x.unsqueeze(1)).squeeze(1)
        x = self.act(x)
        x = self.droppath_se(self.se(x))  + x
        x = self.fc2(x)
        return x
        """
        x1_ = self.model1(image1) 
        x2_ = self.model2(image2)  # b, 2048, 7, 7
        b, dim, _, _ = x1_.shape # 16, 2048, 49
        
        x1 = x1_.reshape(b, dim, self.max_length).permute(0, 2, 1)
        x2 = x2_.reshape(b, dim, self.max_length).permute(0, 2, 1)
        print (x1.shape) #b ., 49, 2048
        
       
        fusion_f = torch.cat((x1, x2), dim=1).squeeze(-1).squeeze(-1) #b, 96, 2048
        

        #fusion_f = torch.cat([image_f, text_f], dim=1)
        fusion_f += self.position_embeddings(self.position_ids)
        fusion_f += self.modality_embeddings(self.modality_ids)
        fusion_f = self.dropout(self.ln(fusion_f))

        fusion_f = self.attn(fusion_f.permute(1, 0, 2))
        fusion_f = self.fc1(fusion_f.permute(1, 0, 2))  # LND -> NLD
        fusion_f = self.relu(fusion_f[:, 1:, :].mean(dim=1))
        x = self.fc2(fusion_f)
        return x
    
class bqvsModelfusion_se(nn.Module):
    def __init__(self, model_name, model_name2, num_classes, is_pretrained=True):
        super(bqvsModelfusion, self).__init__()
        self.model_name = model_name
        self.num_class = num_classes

        # ===================
        base_model1 = se_resnet50(num_classes=1000)
        self.model1 = base_model1
       

        base_model2 = se_resnet50(num_classes=1000)
        self.model2 = base_model2
      
        self.hidden = 4096

        self.act = nn.RReLU()
        self.bn1 = nn.BatchNorm1d(1)
        self.se = SELayer(self.hidden)
        self.droppath_se = DropPath(0.2)
        self.dropout = nn.Dropout(p=0.5)


        self.fc1 = torch.nn.Linear(self.hidden, self.hidden)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in')
        self.fc2 = nn.Linear(self.hidden, 10)

    def forward(self, image1, image2):
        _, x1 = self.model1(image1)
        _, x2 = self.model2(image2)
     
        fusionfea = torch.cat((x1, x2), dim=1).squeeze(-1).squeeze(-1)
        x = self.fc1(fusionfea)
        x = self.bn1(x.unsqueeze(1)).squeeze(1)
        x = self.act(x)

        x = self.droppath_se(self.se(x))  + x
        x = self.fc2(x)

        return x

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

