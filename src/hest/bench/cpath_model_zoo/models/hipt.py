### Dependencies
# Base Dependencies
import os
import pickle
import sys

# LinAlg / Stats / Plotting Dependencies
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm

# Torch Dependencies
import torch
import torch.multiprocessing
import torchvision
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat
torch.multiprocessing.set_sharing_strategy('file_system')

# Fishing Rod Dependencies
from fishing_rod.wsi_core.wsi_utils import filterPatch
from fishing_rod.extract_features.encoder_utils import get_encoder, get_eval_transforms

def flatseq2grid(seq, w, h):
    n, d = seq.shape
    assert w*h == n
    return seq.reshape(w, h, d).transpose(0,1).transpose(0,2).unsqueeze(dim=0)

def grid2flatseq(grid):
    return grid.flatten(2, 3).transpose(1,2)

class HIPTDS(torch.nn.Module):
    """
    HIPT Model (ViT-4K) for encoding non-square images (with [256 x 256] patch tokens), with 
    [256 x 256] patch tokens encoded via ViT-256 using [16 x 16] patch tokens.
    """
    def __init__(self,
            model_lv1: torch.nn.Module,
            device_lv1=torch.device('cuda:0'),
            model_lv2: torch.nn.Module = None,
            device_lv2=torch.device('cuda:1'),
            token_size_lv2_p1: int = 512,
            token_size_lv2_p2: int = 256
    ):
        super().__init__()
        self.model_lv1 = model_lv1
        self.model_lv2 = model_lv2
        self.token_size_lv2_p1 = token_size_lv2_p1
        self.token_size_lv2_p2 = token_size_lv2_p2
        self.device_lv1 = device_lv1
        self.device_lv2 = device_lv2
    
    def forward_ds(self, x1, x2, return_feats_lv1=False):
        batch_lv1_p1 = x1.unfold(2, self.token_size_lv2_p1, self.token_size_lv2_p1).unfold(3, self.token_size_lv2_p1, self.token_size_lv2_p1)
        batch_lv1_p2 = x2.unfold(2, self.token_size_lv2_p2, self.token_size_lv2_p2).unfold(3, self.token_size_lv2_p2, self.token_size_lv2_p2)
        batch_lv1_p1 = rearrange(batch_lv1_p1, 'b c p1 p2 w h -> (b p1 p2) c w h')
        batch_lv1_p2 = rearrange(batch_lv1_p2, 'b c p1 p2 w h -> (b p1 p2) c w h')
        features_cls_lv1_p1 = self.model_lv1(batch_lv1_p1.to(self.device_lv1, non_blocking=True))
        features_cls_lv1_p2 = self.model_lv1(batch_lv1_p2.to(self.device_lv1, non_blocking=True))
        features_cls_lv1 = torch.cat((features_cls_lv1_p1, features_cls_lv1_p2), dim=1)

        _, _, w_p1, h_p1 = x1.shape
        _, _, w_p2, h_p2 = x2.shape
        w_lv1_p1, h_lv1_p1 = w_p1 // self.token_size_lv2_p1, h_p1 // self.token_size_lv2_p1
        w_lv1_p2, h_lv1_p2 = w_p2 // self.token_size_lv2_p2, h_p2 // self.token_size_lv2_p2
        assert w_lv1_p1 == w_lv1_p2 and h_lv1_p1 == h_lv1_p2

        if self.model_lv2 is None:
            return features_cls_lv1
        else:
            features_cls_lv1 = flatseq2grid(features_cls_lv1, w_lv1_p1, h_lv1_p1)
            features_cls_lv1 = features_cls_lv1.to(self.device_lv2, non_blocking=True)
            features_cls_lv2 = self.model_lv2.forward(features_cls_lv1)
            if return_feats_lv1:
                return features_cls_lv2, features_cls_lv1
            return features_cls_lv2
        
    def forward(self, x, return_feats_lv1=False):
        x1, x2 = prepare_img_tensor_ds(x, patch_size=self.token_size_lv2_p1)
        return self.forward_ds(x1, x2, return_feats_lv1=return_feats_lv1)


class HIPT(torch.nn.Module):
    """
    HIPT Model (ViT-4K) for encoding non-square images (with [256 x 256] patch tokens), with 
    [256 x 256] patch tokens encoded via ViT-256 using [16 x 16] patch tokens.
    """
    def __init__(self,
        model_lv1: torch.nn.Module,
        device_lv1=torch.device('cuda:0'), 
        model_lv2: torch.nn.Module = None,
        device_lv2=torch.device('cuda:1'),
        token_size_lv2: int = 256,
        minibatch_size: int = 256,
        patch_filter_params: dict = {'isWhitePatch': {'satThresh': 5}, 'isBlackPatch': {'rgbThresh': 40}}):

        super().__init__()
        self.model_lv1 = model_lv1
        self.model_lv2 = model_lv2
        self.token_size_lv2 = token_size_lv2
        self.minibatch_size = minibatch_size

        self.device_lv1 = device_lv1
        self.device_lv2 = device_lv2
        self.patch_filter_params = patch_filter_params
    
    def forward(self, x, verbose=False):
        """
        Forward pass of HIPT (given an image tensor x), outputting the [CLS] token from ViT-4K.
        1. x is center-cropped such that the W / H is divisible by the patch token size in ViT-4K (e.g. - 256 x 256).
        2. x then gets unfolded into a "batch" of [256 x 256] images.
        3. A pretrained ViT-256 model extracts the CLS token from each [256 x 256] image in the batch.
        4. These batch-of-features are then reshaped into a 2D feature grid (of width "w_lv1" and height "h_lv1".)
        5. This feature grid is then used as the input to ViT-4K, outputting [CLS]_4K.
        
        Args:
            - x (torch.Tensor): [1 x C x W' x H'] image tensor.
        
        Return:
            - features_cls_lv2 (torch.Tensor): [1 x 192] cls token (d_4k = 192 by default).
        """
        batch_lv1, w_lv1, h_lv1 = prepare_img_tensor(x)                    # 1. [1 x 3 x W x H]
        if verbose: print('Post prepare_img shape: ', batch_lv1.shape)

        batch_lv1 = batch_lv1.unfold(2, self.token_size_lv2, self.token_size_lv2).unfold(3, self.token_size_lv2, self.token_size_lv2)           # 2. [1 x 3 x w_lv1 x h_lv1 x 256 x 256] 
        if verbose: print('Post unfold shape: ', batch_lv1.shape)
        batch_lv1 = rearrange(batch_lv1, 'b c p1 p2 w h -> (b p1 p2) c w h')    # 2. [B x 3 x 256 x 256], where B = (1*w_lv1*h_lv1)
        if verbose: print('Post rearrange shape: ', batch_lv1.shape)
                                                                              
        features_cls_lv1 = []
        for mini_bs in range(0, batch_lv1.shape[0], self.minibatch_size):                       # 3. B may be too large for ViT-256. We further take minibatches of 256.
            minibatch_lv1 = batch_lv1[mini_bs:mini_bs+self.minibatch_size].to(self.device_lv1, non_blocking=True)
            features_cls_lv1.append(self.model_lv1(minibatch_lv1).detach().cpu()) # 3. Extracting ViT-256 features from [256 x 3 x 256 x 256] image batches.

        features_cls_lv1 = torch.vstack(features_cls_lv1)                         # 3. [B x 384], where 384 == dim of ViT-256 [ClS] token.
        if verbose: print('Post model_lv1.forward + vstack shape: ', features_cls_lv1.shape)
        features_cls_lv1 = flatseq2grid(features_cls_lv1, w_lv1, h_lv1)
        if verbose: print('Post model_lv1.forward + reshape shape: ', features_cls_lv1.shape)

        if self.model_lv2 is None:
            return features_cls_lv1
        else:
            features_cls_lv1 = features_cls_lv1.to(self.device_lv2, non_blocking=True)  # 4. [1 x 384 x w_lv1 x h_lv1]
            features_cls_lv2 = self.model_lv2.forward(features_cls_lv1)                  # 5. [1 x 192], where 192 == dim of ViT-4K [ClS] token.
            if verbose: print('Post model_lv2.forward', features_cls_lv1.shape)
            return features_cls_lv2
    
    
    def forward_asset_dict(self, x: torch.Tensor):
        """
        Forward pass of HIPT (given an image tensor x), with certain intermediate representations saved in 
        a dictionary (that is to be stored in a H5 file). See walkthrough of how the model works above.
        
        Args:
            - x (torch.Tensor): [1 x C x W' x H'] image tensor.
        
        Return:
            - asset_dict (dict): Dictionary of intermediate feature representations of HIPT and other metadata.
                - features_cls_lv1 (np.array): [B x 384] extracted ViT-256 cls tokens
                - features_mean_lv1 (np.array): [1 x 384] mean ViT-256 cls token (exluding non-tissue patches)
                - contains_tissue_lv1 (np.array): [B,]-dim array with either 1 (tissue-containing) of 0 (white space) for each corresponding image in batch_lv1.
                - features_4k (np.array): [1 x 192] extracted ViT-4K cls token.
                - features_4k (np.array): [1 x 576] feature vector (concatenating mean ViT-256 + ViT-4K cls tokens)
    
        """
        batch_lv1, w_lv1, h_lv1 = prepare_img_tensor(x, patch_size=self.token_size_lv2)
        batch_lv1 = batch_lv1.unfold(2, 256, 256).unfold(3, 256, 256)
        batch_lv1 = rearrange(batch_lv1, 'b c p1 p2 w h -> (b p1 p2) c w h')
        
        features_cls_lv1 = []
        for mini_bs in range(0, batch_lv1.shape[0], 256):
            minibatch_lv1 = batch_lv1[mini_bs:mini_bs+256].to(self.device_lv1, non_blocking=True)
            features_cls_lv1.append(self.model_lv1(minibatch_lv1).detach().cpu())

        features_cls_lv1 = torch.vstack(features_cls_lv1)
        if self.patch_filter_params != None:
            ### Creates a [B,]-dim np.array with either 1 (tissue-containing) of 0 (white space) for each corresponding image in batch_lv1.
            contains_tissue_lv1 = self.filter_tissue(batch_lv1)
            ### Takes mean of ViT-256 features with only tissue-containing patches.
            features_mean_lv1 = features_cls_lv1[np.where(contains_tissue_lv1)].mean(dim=0).unsqueeze(dim=0)
        else:
            features_mean_lv1 = features_cls_lv1.mean(dim=0).unsqueeze(dim=0)

        features_grid256 = features_cls_lv1.reshape(w_lv1, h_lv1, 384).transpose(0,1).transpose(0,2).unsqueeze(dim=0)
        features_grid256 = features_grid256.to(self.device_lv2, non_blocking=True)
        features_cls_lv2 = self.model_lv2.forward(features_grid256).detach().cpu()
        features_mean_lv1_cls_lv2 = torch.cat([features_mean_lv1, features_cls_lv2], dim=1)
        
        asset_dict = {
            'features_cls_lv1': np.expand_dims(features_cls_lv1.numpy(), axis=0),
            'features_mean_lv1': features_mean_lv1.numpy(),
            'contains_tissue_lv1': np.expand_dims(contains_tissue_lv1, axis=0),
            'features_cls_lv2': features_cls_lv2.numpy(),
            'features_mean_lv1_cls_lv2': features_mean_lv1_cls_lv2.numpy()
        }
        return asset_dict				
    
    def filter_tissue(self, batch_lv1: torch.Tensor):
        """
        Helper function that filters each tissue patch in the batch as tissue-containing image (1) or white space (0).
        
        Args:
            - batch_lv1 (torch.Tensor): [B x C x 256 x 256] image tensor batch following unrolling the [1 x 3 x W x H] image tensor)
            into B [256 x 256 x 3] image patches).
            
        Return:
            - contains_tissue_lv1 (np.array): [B,]-dim array with either 1 (tissue-containing) of 0 (white space) for each corresponding image in batch_lv1.
        
        """
        to_filter = np.array([filterPatch(img, patch_filter_params=self.patch_filter_params) 
                              for img in tensorbatch2im(batch_lv1)])
        contains_tissue_lv1 = 1-to_filter
        return contains_tissue_lv1
    

def load_hipt(
        enc_name_lv1: str, 
        checkpoint_lv1='checkpoint.pth',
        device_lv1=torch.device('cuda:0'),
        enc_name_lv2: str=None,
        checkpoint_lv2: str=None,
        device_lv2=None,
        which_hipt='hipt',
        **kwargs
    ):
    
    print("initializing hipt with lv1")
    model_lv1, _, _ = get_encoder(
        enc_name=enc_name_lv1, 
        checkpoint=checkpoint_lv1, 
        extract_mode='patch',
        device=torch.device('cpu')
        )
    
    if enc_name_lv2 is not None:
        print("initializing hipt with lv2")
        model_lv2, _, _ = get_encoder(
            enc_name=enc_name_lv2, 
            checkpoint=checkpoint_lv2, 
            assets_dir=os.path.join('/'.join(os.path.abspath(__file__).split('/')[:-1]), '../../../assets/ckpts', enc_name_lv1, 'hipt_models'),
            extract_mode='patch',
            device=torch.device('cpu')
        )
    else:
        model_lv2 = None
    
    if which_hipt == 'hipt':
        model = HIPT(
            model_lv1=model_lv1, 
            device_lv1=device_lv1, 
            model_lv2=model_lv2, 
            device_lv2=device_lv2, 
            **kwargs
        )
    elif which_hipt == 'hiptds':
        model = HIPTDS(
            model_lv1=model_lv1, 
            device_lv1=device_lv1, 
            model_lv2=model_lv2, 
            device_lv2=device_lv2, 
            **kwargs
        )
        print('initializing hiptds')
    else:
        raise NotImplementedError

    return model

def roll_batch2img(batch: torch.Tensor, w: int, h: int, patch_size=256):
    """
    Rolls an image tensor batch (batch of [256 x 256] images) into a [W x H] Pil.Image object.
    
    Args:
        batch (torch.Tensor): [B x 3 x 256 x 256] image tensor batch.
        
    Return:
        Image.PIL: [W x H X 3] Image.
    """
    batch = batch.reshape(w, h, 3, patch_size, patch_size)
    img = rearrange(batch, 'p1 p2 c w h-> c (p1 w) (p2 h)').unsqueeze(dim=0)
    return Image.fromarray(tensorbatch2im(img)[0])


def tensorbatch2im(input_image, imtype=np.uint8):
    r""""
    Converts a Tensor array into a numpy image array.
    
    Args:
        - input_image (torch.Tensor): (B, C, W, H) Torch Tensor.
        - imtype (type): the desired type of the converted numpy array
        
    Returns:
        - image_numpy (np.array): (B, W, H, C) Numpy Array.
    """
    if not isinstance(input_image, np.ndarray):
        image_numpy = input_image.cpu().float().numpy()  # convert it into a numpy array
        #if image_numpy.shape[0] == 1:  # grayscale to RGB
        #    image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def prepare_img_tensor(img: torch.Tensor, patch_size=256):
    """
    Helper function that takes a non-square image tensor, and takes a center crop s.t. the width / height
    are divisible by 256.
    
    (Note: "_lv1" for w / h is should technically be renamed as "_ps", but may not be easier to read.
    Until I need to make HIPT with patch_sizes != 256, keeping the naming convention as-is.)
    
    Args:
        - img (torch.Tensor): [1 x C x W' x H'] image tensor.
        - patch_size (int): Desired patch size to evenly subdivide the image.
    
    Return:
        - img_new (torch.Tensor): [1 x C x W x H] image tensor, where W and H are divisble by patch_size.
        - w_lv1 (int): # of [256 x 256] patches of img_new's width (e.g. - W/256)
        - h_lv1 (int): # of [256 x 256] patches of img_new's height (e.g. - H/256)
    """
    make_divisble = lambda l, patch_size: (l - (l % patch_size))
    b, c, w, h = img.shape
    load_size = make_divisble(w, patch_size), make_divisble(h, patch_size)
    w_lv1, h_lv1 = w // patch_size, h // patch_size
    img_new = transforms.CenterCrop(load_size)(img)
    return img_new, w_lv1, h_lv1

def prepare_img_tensor_ds(img: torch.Tensor, patch_size=512):
    """
    Helper function that takes a non-square image tensor, and takes a center crop s.t. the width / height
    are divisible by 256.
    
    (Note: "_lv1" for w / h is should technically be renamed as "_ps", but may not be easier to read.
    Until I need to make HIPT with patch_sizes != 256, keeping the naming convention as-is.)
    
    Args:
        - img (torch.Tensor): [1 x C x W' x H'] image tensor.
        - patch_size (int): Desired patch size to evenly subdivide the image.
    
    Return:
        - img_new (torch.Tensor): [1 x C x W x H] image tensor, where W and H are divisble by patch_size.
        - w_lv1 (int): # of [256 x 256] patches of img_new's width (e.g. - W/256)
        - h_lv1 (int): # of [256 x 256] patches of img_new's height (e.g. - H/256)
    """
    make_divisble = lambda l, patch_size: (l - (l % patch_size))
    # pil to tensor
    if isinstance(img, torch.Tensor):
        if len(img.shape) == 4:
            b, c, w, h = img.shape
            assert b == 1
            img = img.squeeze(dim=0)
        else:
            c, w, h = img.shape

        img = img.detach().cpu().transpose(0,1).transpose(1,2)
        img = Image.fromarray(np.array(img, np.uint8))
        
    elif isinstance(img, Image.Image):
        w, h = img.size
    
    load_w, load_h = make_divisble(w, patch_size), make_divisble(h, patch_size)
    img = transforms.CenterCrop((load_w, load_h))(img)
    eval_transform_p1 = get_eval_transforms(img_resize=-1)
    eval_transform_p2 = get_eval_transforms(img_resize=(load_w//2, load_h//2))
    return eval_transform_p1(img).unsqueeze(dim=0), eval_transform_p2(img).unsqueeze(dim=0)