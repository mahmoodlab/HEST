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
        self.model, self.eval_transforms, self.precision = self._build(weights_path, **build_kwargs)
        
    def forward(self, x):
        z = self.model(x)
        return z
        
    @abstractmethod
    def _build(self, **build_kwargs):
        pass
        

class ConchInferenceEncoder(InferenceEncoder):
    
    def _build(self, _):
        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except:
            traceback.print_exc()
            raise Exception("Please install CONCH `pip install git+https://github.com/Mahmoodlab/CONCH.git`")
        
        try:
            model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch")
        except:
            traceback.print_exc()
            raise Exception("Failed to download CONCH model, make sure that you were granted access and that you correctly registered your token")
        
        eval_transform = preprocess
        precision = torch.float32
        
        return model, eval_transform, precision
    
    def forward(self, x):
        return self.model.encode_image(x, proj_contrast=False, normalize=False)
    
    
class CTransPathInferenceEncoder(InferenceEncoder):
    def _build(self, weights_path):
        from torch import nn

        from .ctranspath.ctran import ctranspath
        
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
        self.eval_transforms = transforms
        self.precision = precision
        
    def _build(self, weights_path):
        return None, None, None
    

class PhikonInferenceEncoder(InferenceEncoder):

    def _load(self):
        from transformers import ViTModel
        model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        return model

    def _build(self, _, return_cls=True):
        self.return_cls = return_cls
        model = self._load()
        mean, std = get_constants('imagenet')
        eval_transform = get_eval_transforms(mean, std)
        precision = torch.float32
        return model, eval_transform, precision
    
    def forward(self, x):
        out = self.forward_features(x)
        if self.return_cls:
            out = out.last_hidden_state[:, 0, :]
        else:
            class_token = out.last_hidden_state[:, 0, :]
            patch_tokens = out.last_hidden_state[:, 1:, :]
            out = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        return out
    
    def forward_features(self, x):
        out = self.model(pixel_values=x)
        return out

class PhikonV2InferenceEncoder(PhikonInferenceEncoder):
    def _load(self):
        from transformers import AutoModel
        model = AutoModel.from_pretrained("owkin/phikon-v2")
        return model

class H0MiniInferenceEncoder(InferenceEncoder):
    import timm
    
    def _build(
        self,
        _,
        return_cls=True,
        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
    ):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        model = timm.create_model(
            "hf-hub:bioptimus/H0-mini",
            pretrained=True,
            **timm_kwargs
        )
        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        precision = torch.float16
        self.return_cls = return_cls
        
        return model, eval_transform, precision

    def forward(self, x):
        output = self.model(x)
    
        class_token = output[:, 0]
        if self.return_cls:
            return class_token
        
        patch_tokens = output[:, 5:]
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        return embedding


class RemedisInferenceEncoder(InferenceEncoder):
    def _build(self, weights_path):
        from .remedis.remedis_models import resnet152_remedis
        ckpt_path = weights_path
        model = resnet152_remedis(ckpt_path=ckpt_path, pretrained=True)
        precision = torch.float32
        eval_transform = None
        return model, eval_transform, precision

    def forward(self, x):
        x = x.permute((0, 3, 1, 2))
        return self.model.forward(x)
    
    
class ResNet50InferenceEncoder(InferenceEncoder):
    def _build(
        self, 
        _,
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
        _,
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
    

class GigaPathInferenceEncoder(InferenceEncoder):
    def _build(
        self, 
        _,
        timm_kwargs={}
        ):
        import timm
        from torchvision import transforms
        
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, **timm_kwargs)

        eval_transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        precision = torch.float32
        return model, eval_transform, precision
    
    
class VirchowInferenceEncoder(InferenceEncoder):
    import timm
    
    def _build(
        self,
        _,
        return_cls=False,
        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
    ):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=True,
            **timm_kwargs
        )
        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

        precision = torch.float32
        self.return_cls = return_cls
        
        return model, eval_transform, precision
        
    def forward(self, x):
        output = self.model(x)
        class_token = output[:, 0]

        if self.return_cls:
            return class_token
        else:
            patch_tokens = output[:, 1:]
            embeddings = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
            return embeddings
        

class Virchow2InferenceEncoder(InferenceEncoder):
    import timm
    
    def _build(
        self,
        _,
        return_cls=False,
        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
    ):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            **timm_kwargs
        )
        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        precision = torch.float16
        self.return_cls = return_cls
        
        return model, eval_transform, precision

    def forward(self, x):
        output = self.model(x)
    
        class_token = output[:, 0]
        if self.return_cls:
            return class_token
        
        patch_tokens = output[:, 5:]
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        return embedding

class HOptimus0InferenceEncoder(InferenceEncoder):
    
    def _build(
        self,
        _,
        timm_kwargs={'init_values': 1e-5, 'dynamic_img_size': False}
    ):
        import timm
        from torchvision import transforms

        model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, **timm_kwargs)

        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])
        
        precision = torch.float16
        return model, eval_transform, precision
         
class HibouLargeInferenceEncoder(InferenceEncoder):
    
    def _build(
        self,
        _,
        return_cls=True,
    ):
        from transformers import AutoImageProcessor, AutoModel
        _eval_transform = AutoImageProcessor.from_pretrained("histai/hibou-L", trust_remote_code=True)
        model = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)

        def eval_transform(img):
            return _eval_transform(
                img,
                return_tensors='pt'
            )['pixel_values'].squeeze()

        precision = torch.float32
        self.return_cls = return_cls
        
        return model, eval_transform, precision

    def forward(self, x):
        output = self.model(x).last_hidden_state
    
        class_token = output[:, 0]
        if self.return_cls:
            return class_token
        
        patch_tokens = output[:, 5:]
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        return embedding

class KaikoBase8InferenceEncoder(InferenceEncoder):
    
    def _build(
        self,
        _,
        return_cls=True,
    ):
        from torchvision.transforms import v2
        model = torch.hub.load("kaiko-ai/towards_large_pathology_fms", "vitb8", trust_repo=True)
        eval_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(size=224),
                v2.CenterCrop(size=224),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )

        precision = torch.float32
        self.return_cls = return_cls
        
        return model, eval_transform, precision

    def forward(self, x):
        output = self.model.forward_features(x)
    
        class_token = output[:, 0]
        if self.return_cls:
            return class_token
        
        patch_tokens = output[:, 1:]
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        return embedding



def inf_encoder_factory(enc_name):
    if enc_name == 'conch_v1':
        return ConchInferenceEncoder
    elif enc_name == 'uni_v1':
        return UNIInferenceEncoder
    elif enc_name == 'ctranspath':
        return CTransPathInferenceEncoder
    elif enc_name == 'phikon':
        return PhikonInferenceEncoder
    elif enc_name == 'phikon_v2':
        return PhikonV2InferenceEncoder 
    elif enc_name == "h0_mini":
        return H0MiniInferenceEncoder
    elif enc_name == "hibou_large":
        return HibouLargeInferenceEncoder
    elif enc_name == "kaiko_base_8":
        return KaikoBase8InferenceEncoder
    elif enc_name == 'remedis':
        return RemedisInferenceEncoder
    elif enc_name == 'resnet50':
        return ResNet50InferenceEncoder
    elif enc_name == 'gigapath':
        return GigaPathInferenceEncoder
    elif enc_name == 'virchow':
        return VirchowInferenceEncoder
    elif enc_name == 'virchow2':
        return Virchow2InferenceEncoder
    elif enc_name == 'hoptimus0':
        return HOptimus0InferenceEncoder
    else:
        raise ValueError(f"Unknown encoder name {enc_name}")