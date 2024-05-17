import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image, ImageOps
from PIL import ImageFont
from PIL import ImageDraw 
from scipy.stats import rankdata
from typing import Callable, List, Optional, Sequence, Tuple, Union

# Torch Dependencies
import torch
import torch.nn as nn
import torch.multiprocessing
from torchvision import transforms
torch.multiprocessing.set_sharing_strategy('file_system')
import sklearn.decomposition
import pdb

def cmap_map(function, cmap):
    r""" 
    Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    
    Args:
    - function (function)
    - cmap (matplotlib.colormap)
    
    Returns:
    - matplotlib.colormap
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap', cdict, 1024)

def scale_img(img, scale=0.5, resample=Image.Resampling.LANCZOS):
    w,h = img.size
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    new_w, new_h = int(w*scale[0]), int(h*scale[1])
    return img.resize((new_w, new_h), resample=resample)

def concat_images(imgs, how='horizontal', gap=0, scale=1, border_params={'border':0, 'fill':'black'}):
    r"""
    Function to concatenate list of images (vertical or horizontal).

    Args:
        - imgs (list of PIL.Image): List of PIL Images to concatenate.
        - how (str): How the images are concatenated (either 'horizontal' or 'vertical')
        - gap (int): Gap (in px) between images

    Return:
        - dst (PIL.Image): Concatenated image result.
    """
    gap_dist = (len(imgs)-1)*gap
    if border_params['border'] > 0:
        imgs = [ImageOps.expand(img, **border_params) for img in imgs]
    
    if how == 'vertical':
        w, h = np.max([img.width for img in imgs]), np.sum([img.height for img in imgs])
        h += gap_dist
        curr_h = 0
        dst = Image.new('RGBA', (w, h), color=(255, 255, 255, 0))
        for img in imgs:
            dst.paste(img, (0, curr_h))
            curr_h += img.height + gap
    elif how == 'horizontal':
        w, h = np.sum([img.width for img in imgs]), np.max([img.height for img in imgs])
        w += gap_dist
        curr_w = 0
        dst = Image.new('RGBA', (w, h), color=(255, 255, 255, 0))

        for idx, img in enumerate(imgs):
            dst.paste(img, (curr_w, 0))
            curr_w += img.width + gap
    else:
        raise

    if scale != 1: 
        return scale_img(dst, scale)
    return dst

concat_images_1D = concat_images

def concat_images_2D(imgs: list, w=6, h=2, scale=1, border_params={'border':0, 'fill':'black'}):
    assert w*h == len(imgs)
    imgs = [concat_images(imgs[i:i+w], how='horizontal', border_params=border_params) for i in range(0, len(imgs), w)]
    if border_params['border'] > 0: imgs = [ImageOps.expand(img, **border_params) for img in imgs]
    imgs = concat_images(imgs, how='vertical')
    if scale != 1: return scale_img(imgs, scale)
    return imgs

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


def add_margin(pil_img, top, right, bottom, left, color):
    r"""
    Adds custom margin to PIL.Image.
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


class VisionTransformerVisualizer(nn.Module):
    def __init__(self, model, eval_transform, token_size=16, cmap=cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet), device=None):
        super().__init__()
        self.model = model
        self.eval_transform = eval_transform
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.model = self.model.detach().to(device)
            self.device = device

        self.token_size = token_size
        self.cmap=cmap


    def get_attention_maps(self, img: PIL.Image, block=-1, scale=1, threshold=0.5, offset=0, alpha=0.5):
        def _normalize_scores(attns, size=(256,256)):
            rank = lambda v: rankdata(v)*100/len(v)
            color_block = [rank(attn.flatten()).reshape(size) for attn in attns][0]
            return color_block
    
        w, h = img.size
        img1 = img.copy()
        a_1 = self.get_attention_scores(img1, block=block)
        save_img = np.array(img.copy())
        nhs = a_1.shape[1]

        if offset:
            img2 = add_margin(img.crop((offset, offset, w, h)), top=0, left=0, bottom=offset, right=offset, color=(255,255,255))
            a_2 = self.get_attention_scores(img2, block=block)
    
        ths, hms = [], []
        for i in range(nhs):
            score_1 = _normalize_scores(a_1[:,i,:,:], size=(h,w))
            overlay = np.ones_like(score_1)*100
            if offset:
                score_2 = _normalize_scores(a_2[:,i,:,:], size=(h,w))
                new_score_2 = np.zeros_like(score_2)
                new_score_2[offset:h, offset:w] = score_2[:(h-offset), :(w-offset)]
                overlay[offset:h, offset:w] += 100
                score = (score_1+new_score_2)/overlay
            else:
                score = score_1 / overlay
            
            color_block = (self.cmap(score)*255)[:,:,:3].astype(np.uint8)
            img_hm = cv2.addWeighted(color_block, alpha, save_img.copy(), 1-alpha, 0, save_img.copy())
            hms.append(img_hm)

            if threshold is not None:
                mask = score.copy()
                mask[mask < threshold] = 0
                mask[mask > threshold] = 0.95

                mask_block = (self.cmap(mask)*255)[:,:,:3].astype(np.uint8)
                img_mask = cv2.addWeighted(mask_block, alpha, save_img.copy(), 1-alpha, 0, save_img.copy())
                img_mask[mask==0] = 0
                img_inverse = save_img.copy()
                img_inverse[mask == 0.95] = 0
                ths.append(img_mask+img_inverse)
                
        hms = [Image.fromarray(img) for img in hms]
        ths = [Image.fromarray(img) for img in ths]
        return hms, ths
    
    @torch.no_grad()
    def get_attention_scores(self, img: PIL.Image, block: int=-1):
        r"""
        Forward pass in ViT model with attention scores saved.
        
        Args:
        - img (PIL.Image):          H x W Image 
        
        Returns:
        - attention (torch.Tensor): [1, H, W, 3] torch.Tensor of attention map for the CLS token.
        """

        w, h = img.size
        trsformed_img = self.eval_transform(img)
        _, trsformed_h, trsformed_w = trsformed_img.size()
        batch = trsformed_img.unsqueeze(0)
        batch = batch.to(self.device, non_blocking=True)

        attn = self.model.get_attention(batch, block_num=block)
        # attn: 1, nh, n_tokens+1, n_tokens+1 (1 for cls token)
        nh = attn.shape[1] # number of heads
        w_seq, h_seq = trsformed_w//self.token_size, trsformed_h//self.token_size # number of tokens along w,h of the image
        # attn of cls token to all other tokens
        attn = attn[:, :, 0, 1:].reshape((w_seq * h_seq), nh, -1)
        attn = attn.reshape(1, nh, h_seq, w_seq)
        attn = nn.functional.interpolate(attn, scale_factor=(h/h_seq, w/w_seq), mode="nearest").cpu().numpy()

        return attn
    
    ### DinoV2 Viz

    def get_patch_tokens(self, x, which_tokens='x_prenorm'):
        if isinstance(x, List) and isinstance(x[0], PIL.Image.Image):
            x = torch.stack([self.eval_transform(_x) for _x in x]).to(self.device)
        elif isinstance(x, PIL.Image.Image):
            x = self.eval_transform(x).to(self.device)
        elif isinstance(x, torch.Tensor):
            x = x.to(self.device)

        with torch.no_grad():
            out = self.model.forward_features(x)
            if isinstance(out, torch.Tensor):
                patch_tokens = out.detach().cpu()[:,1:,:].numpy()
            elif isinstance(out, dict):
                for k,v in out.items():
                    if isinstance(v, torch.Tensor): out[k] = v.detach().cpu()
                if which_tokens == 'x_prenorm':
                    patch_tokens = out['x_prenorm'][:,1:,:].numpy()
                elif which_tokens == 'x_norm_patchtokens':
                    patch_tokens = out['x_norm_patchtokens'].numpy()
                else:
                    raise NotImplementedError
        return patch_tokens
                
    def get_pca_heatmaps(self, imgs, binarize_params={'c': 0, 't': 120, 'is_greater': 1}, which_tokens='x_prenorm'):
        def _pca_feats_to_img(feats, side_len_tokens, resize_factor):
            if isinstance(side_len_tokens, int):
                side_len_tokens = (side_len_tokens, side_len_tokens)
            h_seq, w_seq = side_len_tokens
            img = Image.fromarray(feats.reshape(h_seq, w_seq, 3))
            return scale_img(img, resize_factor, resample=Image.Resampling.NEAREST)

        def _binarize_mask(pca_mask, c=0, t=120, is_greater=1):
            return (pca_mask[:, c]<t, pca_mask[:, c]>=t) if is_greater else (pca_mask[:, c]>=t, pca_mask[:, c]<t)

        w, h = imgs[0].size
        x = torch.stack([self.eval_transform(img) for img in imgs]).to(self.device)
        # x: B, C, H, W
        w_seq = x.shape[-1] // self.token_size
        h_seq = x.shape[-2] // self.token_size
        patch_tokens = self.get_patch_tokens(x, which_tokens=which_tokens)
        token_dim = patch_tokens.shape[-1]

        scale_minmax = lambda x: np.uint8(((x-x.min())/(x.max()-x.min()))*255)
        pca = sklearn.decomposition.PCA(n_components=3)
        pca = pca.fit(patch_tokens.reshape(-1, token_dim))
        pca_features_all = [pca.transform(patch_tokens[i]) for i in range(len(imgs))]
        #scaled_masks_all = [pca_feats_to_img(scale_minmax(feats), side_len_tokens, token_size) for feats in pca_features_all]

        masks_fg_all, masks_bg_all = zip(*[_binarize_mask(scale_minmax(feats), **binarize_params) for feats in pca_features_all])

        scale_fg_per_channel = lambda x: np.uint8(np.dstack([(x[:,c]-x[:,c].mean())/(x[:,c].std()**2)+0.5 for c in range(3)])[0]*255)
        scaled_masks_fg_all = []
        for feats, mask_fg in zip(pca_features_all, masks_fg_all):
            img = np.zeros_like(feats, dtype=np.uint8)
            img[mask_fg] = scale_fg_per_channel(feats[mask_fg, :])
            scaled_masks_fg_all.append(_pca_feats_to_img(img, (h_seq, w_seq), (w / w_seq, h / h_seq)))

        return scaled_masks_fg_all
