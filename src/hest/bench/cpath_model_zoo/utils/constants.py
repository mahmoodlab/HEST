IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]
NONE_MEAN = None
NONE_STD = None

def get_constants(norm='imagenet'):
    if norm == 'imagenet':
        return IMAGENET_MEAN, IMAGENET_STD
    elif norm == 'openai_clip':
        return OPENAI_MEAN, OPENAI_STD
    elif norm == 'none':
        return NONE_MEAN, NONE_STD
    else:
        raise ValueError(f"Invalid norm: {norm}")