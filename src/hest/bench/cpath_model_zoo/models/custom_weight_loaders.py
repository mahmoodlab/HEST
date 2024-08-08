import os
import logging
import shutil
import torch

from .vision_transformer_latest import resize_pos_embed

def get_available_checkpoints(
        enc_name,
        lab_drive_dirs=[
            '/media/partners',
            '/media/lab_drive',
            '/mnt/lab_drive',
            '/media/fedshyvana/lab_drive'
        ],
        pretrained_folder='Project_SSL/PretrainedModels',
        to_exclude=['checkpoint.pth', 'checkpoint0000.pth']
):

    for lab_drive_dir in lab_drive_dirs:
        pretrained_model_dir = os.path.join(
            lab_drive_dir, pretrained_folder, enc_name)
        if os.path.isdir(pretrained_model_dir):
            checkpoints = sorted([e.name for e in os.scandir(pretrained_model_dir) if (
                e.name.endswith('.pth') and e.name not in to_exclude)])
            return checkpoints


def download_pretrained_weights(
        enc_name, checkpoint,
        assets_dir=os.path.join(
            '/'.join(os.path.abspath(__file__).split('/')[:-1]), '../../assets/ckpts'),
        lab_drive_dirs=[
            '/media/partners',
            '/media/lab_drive',
            '/mnt/lab_drive',
            '/media/fedshyvana/lab_drive',
            '/media/cloud'
        ],
        subdir='',
        pretrained_folder='Project_SSL/PretrainedModels'
):
    r"""
    Downloads pretrained weights from the lab drive (assumed to be in /<lab_drive_dir>/Project_SSL/PretrainedModels/)
    Args:
        - enc_name (str): Name of the encoder (finds folder that is named <enc_name>, which the model checkpoint assumed to be in this folder)
        - checkpoint (str): Name of the checkpoint file (including extension)
        - assets_dir (str): Path to where checkpoints are saved.
        - lab_drive_dirs (list of str): Paths to where lab drive is potentially mounted.
        - subdir (str): Subdirectory to locate checkpoint.
        - pretrained_folder (str): Local path within lab drive where the checkpoints are stored (kept static at the moment).

    Return:
        - None
    """
    assert os.path.isdir(assets_dir)
    LAB_DRIVE_FOUND = False
    CKPT_FOUND = False
    for lab_drive_dir in lab_drive_dirs:
        pretrained_dir = os.path.join(lab_drive_dir, pretrained_folder)
        if os.path.isdir(pretrained_dir):
            LAB_DRIVE_FOUND = True
            if 'dinov2' in enc_name:
                src = os.path.join(
                    pretrained_dir, 
                    enc_name, 
                    'eval', 
                    f'training_{os.path.splitext(checkpoint)[0]}', 
                    'teacher_checkpoint.pth'
                )
            else:
                src = os.path.join(
                    pretrained_dir, 
                    enc_name,
                    subdir, 
                    checkpoint
                )

            dst = os.path.join(assets_dir, enc_name, checkpoint)

            if os.path.isfile(src):
                CKPT_FOUND = True
                os.makedirs(os.path.join(assets_dir, enc_name), exist_ok=True)
                logging.info(f'Downloading from {src}...')
                shutil.copyfile(src, dst)
                logging.info('Success!!')
                return dst
            else:
                logging.info(f'{src} does not exist.')

    if not LAB_DRIVE_FOUND:
        logging.info('Lab Drive is not attached in any of the following locations:', lab_drive_dirs)
        assert False
    
    if not CKPT_FOUND:
        logging.info(f'Checkpoint {checkpoint} not found.')
        assert False


def load_pretrained_weights_into_model_cocavit(
        model, 
        enc_name, 
        checkpoint, 
        assets_dir=os.path.join('/'.join(os.path.abspath(__file__).split('/')[:-1]), '../../assets/ckpts')
):
    r"""
    Loads model weights from ckpt_path into model, meant to be compatible with timm models.

    Args:
        - model (torch.nn): PyTorch model used as image encoder.
        - enc_name (str): Name of the encoder (finds folder that is named <enc_name>, which the model checkpoint assumed to be in this folder)
        - checkpoint (str): Name of the checkpoint file (including extension)
        - model (torch.nn): PyTorch model used as image encoder.
        - assets_dir (str): Path to where checkpoints are saved.
    Return:
        - model (torch.nn): PyTorch model used as image encoder (now with pretrained weights)
    """
    ckpt_path = os.path.join(assets_dir, enc_name, checkpoint)
    if os.path.isfile(ckpt_path):
        logging.info(f'Checkpoint {ckpt_path} found!')
    else:
        logging.info(
            f'Checkpoint {ckpt_path} not found! Downloading from lab drive...')
        download_pretrained_weights(
            enc_name,
            checkpoint=checkpoint,
            pretrained_folder='PretrainedModels/vision_language',
            subdir='checkpoints',
            assets_dir=assets_dir
        )

    state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
    state_dict = {k.replace('module.visual.', ''): v for k,
                  v in state_dict.items() if 'visual' in k}
    
    #### DINO v2 specific? ###
    import re
    # trunk.blocks.0.0.attn.proj.weight
    pattern = r"\.0\.(?=\d.)" # pattern to match "trunk.blocks.0.x.attn.proj.weight"
    replacement = "." 
    state_dict = {re.sub(pattern, replacement, k): v for k, v in state_dict.items()}
    #### DINO v2 specific? ###

    state_dict['trunk.pos_embed'] = resize_pos_embed(model.trunk, state_dict['trunk.pos_embed'])
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logging.info(f'Missing Keys: {missing_keys}')
    logging.info(f'Unexpected Keys: {unexpected_keys}')
    return model


def load_pretrained_weights_into_model_timmvit(model, enc_name, checkpoint, assets_dir=os.path.join('/'.join(os.path.abspath(__file__).split('/')[:-1]), '../../assets/ckpts')):
    r"""
    Loads model weights from ckpt_path into model, meant to be compatible with timm models.

    Args:
        - model (torch.nn): PyTorch model used as image encoder.
        - enc_name (str): Name of the encoder (finds folder that is named <enc_name>, which the model checkpoint assumed to be in this folder)
        - checkpoint (str): Name of the checkpoint file (including extension)
        - model (torch.nn): PyTorch model used as image encoder.
        - assets_dir (str): Path to where checkpoints are saved.
    Return:
        - model (torch.nn): PyTorch model used as image encoder (now with pretrained weights)
    """
    ckpt_path = os.path.join(assets_dir, enc_name, checkpoint)

    if os.path.isfile(ckpt_path):
        logging.info(f'Checkpoint {ckpt_path} found!')
    else:
        logging.info(f'Checkpoint {ckpt_path} not found! Downloading from lab drive...')
        ckpt_path = download_pretrained_weights(
            enc_name, checkpoint, assets_dir, pretrained_folder='PretrainedModels/vision')

    if 'ijepa' in enc_name:
        state_dict_key = 'teacher'
    elif 'mocov3' in enc_name:
        state_dict_key = 'state_dict'
    elif 'supervised' in enc_name:
        state_dict_key = 'state_dict'
    else:
        state_dict_key = 'teacher'

    state_dict = torch.load(ckpt_path, map_location="cpu")[state_dict_key]
    if enc_name.split('.')[1] in ['dino', 'dino_hipt']:
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    elif 'supervised' in enc_name:
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
    elif 'dinov2' in enc_name:
        state_dict = clean_state_dict_dinov2(state_dict)
    elif 'ijepa' in enc_name:
        state_dict = clean_state_dict_ijepa(state_dict)
    elif 'mocov3' in enc_name:
        state_dict = {k.replace('base_encoder.', ''): v for k,
                  v in state_dict.items() if 'base_encoder.' in k}
        state_dict = {k.replace("module.", ""): v for k,
                      v in state_dict.items()}
    else:
        state_dict = {k.replace("module.", ""): v for k,
                      v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k,
                      v in state_dict.items()}

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict, strict=False)
    logging.info(f'Missing Keys: {missing_keys}')
    logging.info(f'Unexpected Keys: {unexpected_keys}')
    print(f'Missing Keys: {missing_keys}')
    print(f'Unexpected Keys: {unexpected_keys}')

    if enc_name.split('.')[1] in ['dino', 'dino_hipt']:
        assert len(missing_keys) == 0
        for k in unexpected_keys:
            assert k.split('.')[0] == 'head'
    elif enc_name.split('.')[1] in ['dinov2']:
        assert len(missing_keys) == 0
        print(unexpected_keys)
        for k in unexpected_keys:
            assert k.split('.')[0] == 'dino_head'
    return model
