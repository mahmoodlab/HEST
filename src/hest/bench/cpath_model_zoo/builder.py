from .models.custom_weight_loaders import load_pretrained_weights_into_model_cocavit
from .models.vision_transformer_custom import vit_large_w_pooler
from .models.densenetbackbone import DenseNetBackbone
from .models.remedis_models import resnet152_remedis
from .models.phikon import ibot_vit
from .models import TimmCNNEncoder, TimmViTEncoder, HFViTEncoder
import timm
import os
from functools import partial
import torch
from loguru import logger
from .model_registry import _MODEL_CONFIGS
from .utils import get_eval_transforms, get_constants
import torch.nn as nn
import torchvision.models as models
from .models.post_processor import CLIPVisionModelPostProcessor

def get_encoder(model_name, overwrite_kwargs={}, img_size = 224):
    config = _MODEL_CONFIGS[model_name]
    for k in overwrite_kwargs:
        if k not in config:
            raise ValueError(f"Invalid overwrite key: {k}")
        config[k] = overwrite_kwargs[k]
    model, eval_transform = build_model(config)
    mean, std = get_constants(config['img_norm'])
    
    if eval_transform is None:
        eval_transform = get_eval_transforms(mean, std, target_img_size=img_size)
    return model, eval_transform, config

def load_resnet18_ciga(ckpt_path):
    def clean_state_dict_ciga(state_dict):
        state_dict = {k.replace("model.resnet.", ''):v for k,v in state_dict.items() if 'fc.' not in k}
        return state_dict
    base_encoder = models.resnet18(weights=None)
    base_encoder.fc = nn.Identity()
    state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
    state_dict = clean_state_dict_ciga(state_dict)
    base_encoder.load_state_dict(state_dict, strict=True)
    return base_encoder


def build_model(config):
    logger.info(f"Building model with config: {config['name']}")
    load_state_dict = False
    eval_transform = None
    if config.get("checkpoint_path", None) is not None:
        if not os.path.exists(config["checkpoint_path"]):
            if os.environ.get("CHECKPOINT_PATH", None) is not None:
                config["checkpoint_path"] = os.environ["CHECKPOINT_PATH"]
            else:
                raise ValueError(f"checkpoint_path does not exist: {config['checkpoint_path']} and no CHECKPOINT_PATH environment variable set")
        load_state_dict = True
    if config['loader'] == 'timm_wrapper_cnn':
        # uses timm to load a CNN model, then wraps it in a custom module that adds pooling
        model = TimmCNNEncoder(**config['loader_kwargs'])
    elif config['loader'] == 'hf_wrapper_vit':
        model = HFViTEncoder(**config['loader_kwargs'])
    elif config['loader'] == 'conch_openclip_custom':
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained(**config['loader_kwargs'], checkpoint_path=config["checkpoint_path"])
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    elif config['loader'] == 'timm':
        # uses timm to load a model
        model = timm.create_model(**config['loader_kwargs'])
    elif config['loader'] == 'ctranspath_loader':
        from .models.ctran import ctranspath
        ckpt_path = config["checkpoint_path"]
        assert os.path.isfile(ckpt_path)
        model = ctranspath(img_size=224)
        model.head = nn.Identity()
        state_dict = torch.load(ckpt_path)['model']
        state_dict = {key: val for key, val in state_dict.items() if 'attn_mask' not in key}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        load_state_dict = False
    ### Kimia Net
    elif config['loader'] == 'kimianet_loader':
        ckpt_path = config["checkpoint_path"]
        assert os.path.isfile(ckpt_path)
        model = models.densenet121()
        state_dict = torch.load(ckpt_path, map_location='cpu')
        state_dict = {"features."+k[len("module.model.0."):]:v for k,v in state_dict.items() if "fc_4" not in k}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        assert missing_keys == ['classifier.weight', 'classifier.bias']
        model = DenseNetBackbone(model)
        load_state_dict = False
    elif config['loader'] == 'ciga_loader':
        model = load_resnet18_ciga(config["checkpoint_path"])
        load_state_dict = False
    elif config['loader'] == 'remedis_loader':
        ckpt_path = config["checkpoint_path"]
        model = resnet152_remedis(ckpt_path=ckpt_path, pretrained=True)
        load_state_dict = False
    elif config['loader'] == 'plip_loader':
        from transformers import CLIPImageProcessor, CLIPVisionModel
        model_name = "vinid/plip"
        img_transforms_clip = CLIPImageProcessor.from_pretrained(model_name)
        model = CLIPVisionModel.from_pretrained(
            model_name)  # Use for feature extraction
        model = CLIPVisionModelPostProcessor(model)
        def _eval_transform(img): return img_transforms_clip(
            img, return_tensors='pt', padding=True)['pixel_values'].squeeze(0)
        eval_transform = _eval_transform
    elif config['loader'] == 'ibot_uni':
        ckpt_path = config["checkpoint_path"]
        model = ibot_vit.iBOTViT(architecture="vit_base_pancan", encoder="teacher", weights_path=ckpt_path)
        
        load_state_dict = False
    elif config['loader'] == 'pathchat':
        kwargs = {}
        add_kwargs = {'pooler_n_queries_contrast': 1}
        add_kwargs['legacy'] = False
        kwargs.update(add_kwargs)
        model = vit_large_w_pooler(**kwargs, init_values=1e-6)
        ckpt_path = config["checkpoint_path"]
        checkpoint = ckpt_path.split('/')[-1]
        enc_name = os.path.dirname(ckpt_path).split('/')[-1]
        assets_dir = os.path.dirname(os.path.dirname(ckpt_path))
        load_pretrained_weights_into_model_cocavit(
            model, enc_name, checkpoint, assets_dir)

        load_state_dict = False
    
    elif config['loader'] == 'gigapath':
        from torchvision import transforms
        model = timm.create_model(model_name='vit_giant_patch14_dinov2', 
                **{'img_size': 224, 'in_chans': 3, 
                'patch_size': 16, 'embed_dim': 1536, 
                'depth': 40, 'num_heads': 24, 'init_values': 1e-05, 
                'mlp_ratio': 5.33334, 'num_classes': 0})
        ckpt_path = config["checkpoint_path"]
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        eval_transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )
        load_state_dict = False
        
    else:
        raise ValueError(f"Unsupported loader type: {config['loader']}")
    if load_state_dict:
        ckpt_path = config["checkpoint_path"]
        strict = config.get("load_state_dict_strict", False)
        logger.info(f"Loading model from checkpoint: {ckpt_path}")
        logger.info(f"load_state_dict_strict: {strict}")
        missing, unexpected = model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), 
                              strict=strict)
        logger.info(f"Missing keys: {missing}")
        logger.info(f"Unexpected keys: {unexpected}")
    return model, eval_transform
