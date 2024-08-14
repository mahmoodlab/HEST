import torch
import timm

from transformers import ViTModel

class HFViTEncoder(torch.nn.Module):
    def __init__(self, model_name: str = "owkin/phikon", 
                 kwargs: dict = {'add_pooling_layer': False}):
        super().__init__()
        self.model = ViTModel.from_pretrained(model_name, **kwargs)
        self.model_name = model_name
    
    def forward(self, x):
        out = self.forward_features(x)
        out = out.last_hidden_state[:, 0, :]
        return out
    
    def forward_features(self, x):
        out = self.model(pixel_values=x)
        return out