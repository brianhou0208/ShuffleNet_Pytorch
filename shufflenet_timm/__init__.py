import torch
from .shufflenet import create_shufflenet_timm as _create_shufflenet_timm

name_list = [
    'shufflenetv1_050_g3',
    'shufflenetv1_100_g3',
    'shufflenetv1_150_g3',
    'shufflenetv1_200_g3',
    'shufflenetv1_050_g8',
    'shufflenetv1_100_g8',
    'shufflenetv1_150_g8',
    'shufflenetv1_200_g8',
    'shufflenetv2_050',
    'shufflenetv2_100',
    'shufflenetv2_150',
    'shufflenetv2_200',
]

def create_shufflenet_timm(name):
    torch.manual_seed(0)
    if name in name_list:
        return _create_shufflenet_timm(name)
