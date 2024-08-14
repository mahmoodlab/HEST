import traceback
from abc import abstractmethod

import torch
from loguru import logger

from hest.bench.cpath_model_zoo.utils.constants import get_constants
from hest.bench.cpath_model_zoo.utils.transform_utils import \
    get_eval_transforms
        
        
class InferenceEncoder(torch.nn.Module):
    
    def __init__(self, weights_path=None, **build_kwargs):
        super(InferenceEncoder, self).__init__()
        
        self.weights_path = weights_path
        self.model, self.eval_transforms, self.precision = self._build(weights_path=self.weights_path, **build_kwargs)
        
    def forward(self, x):
        z = self.model(x)
        return z
        
    @abstractmethod
    def _build(self, **build_kwargs):
        pass
        

class ConchInferenceEncoder(InferenceEncoder):
    name = 'conch_v1'
    
    def _build(self, _):
        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except:
            traceback.print_exc()
            raise Exception("Please install CONCH pip install `git+https://github.com/Mahmoodlab/CONCH.git`")
        
        try:
            model, _ = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch")
        except:
            traceback.print_exc()
            raise Exception("Failed to download CONCH model, make sure that you were granted access and that you correctly registered your token")
        
        eval_transform = None
        precision = torch.float32
        
        return model, eval_transform, precision
    
    def forward(self, x):
        return self.model.encode_image(x, proj_contrast=False, normalize=True)
    
    
class CTransPathInferenceEncoder(InferenceEncoder):
    def _build(self, weights_path):
        from torch import nn

        from .models.ctranspath.ctran import ctranspath
        
        model = ctranspath(img_size=224)
        model.head = nn.Identity()
        state_dict = torch.load(weights_path)['model']
        state_dict = {key: val for key, val in state_dict.items() if 'attn_mask' not in key}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Missing keys: {missing}")
        logger.info(f"Unexpected keys: {unexpected}")

        mean, std = get_constants('imagenet')
        eval_transform = get_eval_transforms(mean, std)
        precision = torch.float32
        
        return model, eval_transform, precision

    
class CustomInferenceEncoder(InferenceEncoder):
    def __init__(self, weights_path, name, model, transforms, precision):
        super().__init__(weights_path)
        self.model = model
        self.transforms = transforms
        self.precision = precision
        
    def _build(self, weights_path):
        return self.model, self.transforms, self.precision
    

class PhikonInferenceEncoder(InferenceEncoder):
    def _build(self):
        from transformers import ViTModel
        
        model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        mean, std = get_constants('imagenet')
        eval_transform = get_eval_transforms(mean, std)
        precision = torch.float32
        return model, eval_transform, precision
    
    def forward(self, x):
        out = self.forward_features(x)
        out = out.last_hidden_state[:, 0, :]
        return out
    
    def forward_features(self, x):
        out = self.model(pixel_values=x)
        return out
    

class PlipInferenceEncoder(InferenceEncoder):
    def _build(self, weights_path):
        from transformers import CLIPImageProcessor, CLIPVisionModel

        from .models.plip.post_processor import CLIPVisionModelPostProcessor
        model_name = "vinid/plip"
        img_transforms_clip = CLIPImageProcessor.from_pretrained(model_name)
        model = CLIPVisionModel.from_pretrained(
            model_name)  # Use for feature extraction
        model = CLIPVisionModelPostProcessor(model)
        def _eval_transform(img): return img_transforms_clip(
            img, return_tensors='pt', padding=True)['pixel_values'].squeeze(0)
        eval_transform = _eval_transform
        precision = torch.float32
        
        missing, unexpected = model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
        logger.info(f"Missing keys: {missing}")
        logger.info(f"Unexpected keys: {unexpected}")
        return model, eval_transform, precision
    
    def forward(self, x):
        return self.model(x).pooler_output
    

class RemedisInferenceEncoder(InferenceEncoder):
    def _build(self, weights_path):
        from .models.remedis.remedis_models import resnet152_remedis
        ckpt_path = weights_path
        model = resnet152_remedis(ckpt_path=ckpt_path, pretrained=True)
        missing, unexpected = model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
        logger.info(f"Missing keys: {missing}")
        logger.info(f"Unexpected keys: {unexpected}")
        
        precision = torch.float32
        eval_transform = None
        return model, eval_transform, precision
    
    
class ResNet50InferenceEncoder(InferenceEncoder):
    def _build(
        self, 
        pretrained=True, 
        timm_kwargs={"features_only": True, "out_indices": [3], "num_classes": 0},
        pool=True
    ):
        import timm

        model = timm.create_model("resnet50.tv_in1k", pretrained=pretrained, **timm_kwargs)
        mean, std = get_constants('imagenet')
        eval_transform = get_eval_transforms(mean, std)
        precision = torch.float32
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
        
        return model, eval_transform, precision
    
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
                     
    
class UNIInferenceEncoder(InferenceEncoder):
    def _build(
        self, 
        timm_kwargs={"dynamic_img_size": True, "num_classes": 0, "init_values": 1.0}
    ):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        
        try:
            model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, **timm_kwargs)
        except:
            traceback.print_exc()
            raise Exception("Failed to download UNI model, make sure that you were granted access and that you correctly registered your token")
        
        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        precision = torch.float16
        return model, eval_transform, precision
    
def inf_encoder_factory(enc_name):
    if enc_name == 'conch_v1':
        return ConchInferenceEncoder
    elif enc_name == 'uni_v1':
        return UNIInferenceEncoder
    elif enc_name == 'ctranspath':
        return CTransPathInferenceEncoder
    elif enc_name == 'phikon':
        return PhikonInferenceEncoder
    elif enc_name == 'plip':
        return PlipInferenceEncoder
    elif enc_name == 'remedis':
        return RemedisInferenceEncoder
    elif enc_name == 'resnet50':
        return ResNet50InferenceEncoder
    else:
        raise ValueError(f"Unknown encoder name {enc_name}")