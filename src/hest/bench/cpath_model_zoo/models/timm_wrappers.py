import torch
import timm

class TimmCNNEncoder(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet50.tv_in1k', 
                 kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': True, 'num_classes': 0}, 
                 pool: bool = True):
        super().__init__()
        if kwargs.get('pretrained', False) == False:
            # give a warning
            print(f"Warning: {model_name} is used to instantiate a CNN Encoder, but no pretrained weights are loaded. This is expected if you will be loading weights from a checkpoint.")
        self.model = timm.create_model(model_name, **kwargs)
        self.model_name = model_name
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
    
    def forward(self, x):
        out = self.forward_features(x)
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out
    
    def forward_features(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        return out
                           
class TimmViTEncoder(torch.nn.Module):
    def __init__(self, model_name: str = "vit_large_patch16_224", 
                 kwargs: dict = {'dynamic_img_size': True, 'pretrained': True, 'num_classes': 0}):
        super().__init__()
        if kwargs.get('pretrained', False):
            # give a warning
            print(f"Warning: {model_name} is used to instantiate a Timm ViT Encoder, but no pretrained weights are loaded. This is expected if you will be loading weights from a checkpoint.")
        self.model = timm.create_model(model_name, **kwargs)
        self.model_name = model_name
    
    def forward(self, x):
        out = self.model(x)
        return out
    
    def forward_features(self, x):
        out = self.model.forward_features(x)
        return out