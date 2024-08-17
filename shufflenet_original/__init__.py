import torch
from .shufflenetv1 import ShuffleNetV1
from .shufflenetv2 import ShuffleNetV2

def create_shufflenet_original(version, width, group=None):
    torch.manual_seed(0)
    if version == 'v1':
        model = ShuffleNetV1(model_size=width, group=group)
    elif version == 'v2':
        model = ShuffleNetV2(model_size=width)
    return model