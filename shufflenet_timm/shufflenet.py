"""
An implementation of ShuffleNet & ShuffleNet V2 Model as defined in:
ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices - https://arxiv.org/abs/1707.01083
Shufflenet v2: Practical guidelines for efficient cnn architecture design - https://arxiv.org/pdf/1807.11164

Original implementation: https://github.com/megvii-model/ShuffleNet-Series
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectAdaptivePool2d, Linear
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import checkpoint_seq
from timm.models.registry import register_model, generate_default_cfgs

__all__ = ['ShuffleNet']

class ShuffleV1Block(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size=3,
            stride=1,
            groups=3,
            first_layer=False,
            rd_ratio=4,
    ):
        super(ShuffleV1Block, self).__init__()
        assert stride in [1, 2]
        assert groups in [3, 8]
        self.stride = stride
        self.in_chs = in_chs
        self.mid_chs = out_chs // rd_ratio
        self.out_chs = out_chs - in_chs if stride == 2 else out_chs
        self.kernel = kernel_size
        self.padding = kernel_size // 2
        self.groups = groups
        pointwise1 = [
            nn.Conv2d(
                self.in_chs, self.mid_chs, 1, groups=1 if first_layer else self.groups, bias=False),
            nn.BatchNorm2d(self.mid_chs),
            nn.ReLU(inplace=True),
        ]
        depthwise = [
            nn.Conv2d(
                self.mid_chs, self.mid_chs, self.kernel, self.stride, 
                padding=self.padding, groups=self.mid_chs, bias=False
            ),
            nn.BatchNorm2d(self.mid_chs),
        ]

        self.branch_main_1 = nn.Sequential(*pointwise1, *depthwise)
        self.branch_main_2 = nn.Sequential(
            nn.Conv2d(self.mid_chs, self.out_chs, 1, groups=self.groups, bias=False),
            nn.BatchNorm2d(self.out_chs),
        )

        if self.stride == 2:
            self.branch_proj = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.branch_proj = nn.Identity()

    def forward(self, x):
        x_proj = self.branch_proj(x)
        x_main = self.branch_main_1(x)
        x_main = self.channel_shuffle(x_main)
        x_main = self.branch_main_2(x_main)
        if self.stride == 1:
            return F.relu(x_proj + x_main)
        else:
            x_main = F.relu(x_main)
            return torch.cat((x_proj, x_main), 1)
    def channel_shuffle(self, x):
        B, C, H, W = x.shape
        assert (C % self.groups == 0)
        G = C // self.groups

        x = x.reshape(B, G, self.groups, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B, C, H, W)
        return x

class ShuffleV2Block(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size=3,
            stride=1,
            rd_ratio=2
    ):
        super(ShuffleV2Block, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.in_chs = in_chs
        self.mid_chs = out_chs // rd_ratio
        self.out_chs = out_chs - in_chs
        self.kernel = kernel_size
        self.padding = kernel_size // 2

        pointwise1 = [
            nn.Conv2d(self.in_chs, self.mid_chs, 1, bias=False),
            nn.BatchNorm2d(self.mid_chs),
            nn.ReLU(inplace=True),
        ]
        depthwise = [
            nn.Conv2d(
                self.mid_chs, self.mid_chs, self.kernel, self.stride, 
                padding=self.padding, groups=self.mid_chs, bias=False
            ),
            nn.BatchNorm2d(self.mid_chs),
        ]
        pointwise2 = [
            nn.Conv2d(self.mid_chs, self.out_chs, 1, bias=False),
            nn.BatchNorm2d(self.out_chs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*pointwise1, *depthwise, *pointwise2)

        if self.stride == 2:
            self.branch_proj = nn.Sequential(
                nn.Conv2d(
                    self.in_chs, self.in_chs, self.kernel, self.stride, 
                    padding=self.padding, groups=self.in_chs, bias=False
                ),
                nn.BatchNorm2d(self.in_chs),
                nn.Conv2d(self.in_chs, self.in_chs, 1, bias=False),
                nn.BatchNorm2d(self.in_chs),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch_proj = nn.Identity()
    
    def forward(self, x):
        x_proj, x_main = self.channel_shuffle(x)
        x_proj = self.branch_proj(x_proj)
        x_main = self.branch_main(x_main)
        return torch.cat((x_proj, x_main), 1)
    
    def channel_shuffle(self, x):
        if self.stride == 2:
            return x, x
        B, C, H, W = x.shape
        assert (C % 4 == 0)
        x = x.reshape(B * C // 2, 2, H * W)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, C // 2, H, W)
        return x[0], x[1]

class ShuffleNetV1(nn.Module):
    def __init__(
            self, 
            cfgs, 
            num_classes=1000, 
            in_chans=3,
            output_stride=32,
            global_pool='avg',
            drop_rate=0.2,
        ):
        super(ShuffleNetV1, self).__init__()
        # setting of inverted residual blocks
        assert output_stride == 32, 'only output_stride==32 is valid, dilation not supported'
        self.cfgs = cfgs
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.feature_info = []

        # building first layer
        stem_k, stem_s, stem_chs, _, _ = self.cfgs[0][0]
        stem_p = stem_k // 2
        self.conv_stem = nn.Conv2d(in_chans, stem_chs, stem_k, stem_s, stem_p, bias=False)
        self.feature_info.append(dict(num_chs=stem_chs, reduction=2, module=f'conv_stem'))
        self.bn1 = nn.BatchNorm2d(stem_chs)
        self.act1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=stem_k, stride=stem_s, padding=stem_p)
        prev_chs = stem_chs

        # building blocks
        stages = nn.ModuleList([])
        stage_idx = 0
        net_stride = 2
        first_layer = True
        for cfg in self.cfgs[1:]:
            layers = []
            stride = 1
            for kernel, stride, out_chs, repeat, groups in cfg:
                for _ in range(repeat):
                    if first_layer:
                        layers.append(
                            ShuffleV1Block(prev_chs, out_chs, kernel, stride, groups, first_layer)
                        )
                        first_layer = False
                    else:
                        layers.append(
                            ShuffleV1Block(prev_chs, out_chs, kernel, stride, groups)
                        )
                    prev_chs = out_chs
            
                if stride > 1:
                    net_stride *= 2
                self.feature_info.append(dict(
                    num_chs=prev_chs, reduction=net_stride, module=f'blocks.{stage_idx}'))
            stage_idx += 1
            stages.append(nn.Sequential(*layers))
        self.blocks = nn.Sequential(*stages) 
        
        # building last several layers
        self.num_features = out_chs
        # self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.global_pool = nn.AvgPool2d(7)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(out_chs, num_classes, bias=False) if num_classes > 0 else nn.Identity()
        self._initialize_weights()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^conv_stem|bn1',
            blocks=[
                (r'^blocks\.(\d+)' if coarse else r'^blocks\.(\d+)\.(\d+)', None),
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier
    
    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        # cannot meaningfully change pooling of efficient head after creation
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.num_features, num_classes, bias=False) if num_classes > 0 else nn.Identity()
    
    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)
        return x
    
    def forward_head(self, x):
        x = self.global_pool(x)
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
    
    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'conv_stem' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class ShuffleNetV2(nn.Module):
    def __init__(
            self, 
            cfgs, 
            num_classes=1000, 
            in_chans=3,
            output_stride=32,
            global_pool='avg',
            drop_rate=0.2,
        ):
        super(ShuffleNetV2, self).__init__()
        # setting of inverted residual blocks
        assert output_stride == 32, 'only output_stride==32 is valid, dilation not supported'
        self.cfgs = cfgs
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.feature_info = []

        # building first layer
        stem_k, stem_s, stem_chs, _ = self.cfgs[0][0]
        stem_p = stem_k // 2
        self.conv_stem = nn.Conv2d(in_chans, stem_chs, stem_k, stem_s, stem_p, bias=False)
        self.feature_info.append(dict(num_chs=stem_chs, reduction=2, module=f'conv_stem'))
        self.bn1 = nn.BatchNorm2d(stem_chs)
        self.act1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=stem_k, stride=stem_s, padding=stem_p)
        prev_chs = stem_chs

        # building blocks
        stages = nn.ModuleList([])
        stage_idx = 0
        net_stride = 2
        for cfg in self.cfgs[1:-1]:
            layers = []
            stride = 1
            for kernel, stride, out_chs, repeat in cfg:
                for _ in range(repeat):
                    if repeat == 1:
                        layers.append(
                            ShuffleV2Block(prev_chs, out_chs, kernel, stride)
                        )
                    else:
                        layers.append(
                            ShuffleV2Block(prev_chs // 2, out_chs, kernel, stride)
                        )
                    prev_chs = out_chs
            
                if stride > 1:
                    net_stride *= 2
                self.feature_info.append(dict(
                    num_chs=prev_chs, reduction=net_stride, module=f'blocks.{stage_idx}'))
            stage_idx += 1
            stages.append(nn.Sequential(*layers))
        self.blocks = nn.Sequential(*stages) 

        kernel, stride, out_chs, _ = self.cfgs[-1][0]
        self.conv_head = nn.Conv2d(prev_chs, out_chs, kernel, stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chs)
        self.act2 = nn.ReLU(inplace=True)
        
        # building last several layers
        self.num_features = self.head_hidden_size = out_chs
        # self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        # self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_pool = nn.AvgPool2d(7)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.head_hidden_size, num_classes, bias=False) if num_classes > 0 else nn.Identity()
        self._initialize_weights()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^conv_stem|bn1',
            blocks=[
                (r'^blocks\.(\d+)' if coarse else r'^blocks\.(\d+)\.(\d+)', None),
                (r'^conv_head|bn2')
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier
    
    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        # cannot meaningfully change pooling of efficient head after creation
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.head_hidden_size, num_classes, bias=False) if num_classes > 0 else nn.Identity()
    
    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x
    
    def forward_head(self, x):
        x = self.global_pool(x)
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
    
    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'conv_stem' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def checkpoint_filter_fn(state_dict, model: nn.Module):
    out_dict = {}
    for k, v in state_dict.items():
        if 'total' in k:
            continue
        out_dict[k] = v
    return out_dict

def _create_shufflenetv1(variant, width=1.0, group=3, pretrained=False, **kwargs):
    """
    Constructs a ShuffleNetV1 model
    """
    if group == 3:
        if width == 0.5:
            dim = [12, 120, 240, 480]
        elif width == 1.0:
            dim = [24, 240, 480, 960]
        elif width == 1.5:
            dim = [24, 360, 720, 1440]
        elif width == 2.0:
            dim = [48, 480, 960, 1920]
    elif group == 8:
        if width == 0.5:
            dim = [16, 192, 384, 768]
        elif width == 1.0:
            dim = [24, 384, 768, 1536]
        elif width == 1.5:
            dim = [24, 576, 1152, 2304]
        elif width == 2.0:
            dim = [48, 768, 1536, 3072]

    cfgs = [
        # kernel, stride, channel, repeat, group
        # stage1
        [[3, 2, dim[0], 1, group]],
        # stage2
        [[3, 2, dim[1], 1, group],
         [3, 1, dim[1], 3, group],
        ],
        # stage3
        [[3, 2, dim[2], 1, group],
         [3, 1, dim[2], 7, group],
        ],
        # stage4
        [[3, 2, dim[3], 1, group],
         [3, 1, dim[3], 3, group],
        ],
    ]
    model_kwargs = dict(
        cfgs=cfgs,
        **kwargs,
    )
    return build_model_with_cfg(
        ShuffleNetV1,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True),
        **model_kwargs,
    )

def _create_shufflenetv2(variant, width=1.0, pretrained=False, **kwargs):
    """
    Constructs a ShuffleNetV2 model
    """
    if width == 0.5:
        dim = [24, 48, 96, 192, 1024]
    elif width == 1.0:
        dim = [24, 116, 232, 464, 1024]
    elif width == 1.5:
        dim = [24, 176, 352, 704, 1024]
    elif width == 2.0:
        dim = [24, 244, 488, 976, 2048]

    cfgs = [
        # kernel, stride, channel, repeat
        # stage1
        [[3, 2, dim[0], 1]],
        # stage2
        [[3, 2, dim[1], 1],
         [3, 1, dim[1], 3],
        ],
        # stage3
        [[3, 2, dim[2], 1],
         [3, 1, dim[2], 7],
        ],
        # stage4
        [[3, 2, dim[3], 1],
         [3, 1, dim[3], 3],
        ],
        # Stage5(Conv5)
        [[1, 1, dim[4], 1]]
    ]
    model_kwargs = dict(
        cfgs=cfgs,
        **kwargs,
    )
    return build_model_with_cfg(
        ShuffleNetV2,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True),
        **model_kwargs,
    )

def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }

default_cfgs = generate_default_cfgs({
    'shufflenetv1_050_g3.untrained': _cfg(),
    'shufflenetv1_100_g3.untrained': _cfg(),
    'shufflenetv1_150_g3.untrained': _cfg(),
    'shufflenetv1_200_g3.untrained': _cfg(),

    'shufflenetv1_050_g8.untrained': _cfg(),
    'shufflenetv1_100_g8.untrained': _cfg(),
    'shufflenetv1_150_g8.untrained': _cfg(),
    'shufflenetv1_200_g8.untrained': _cfg(),

    'shufflenetv2_050.untrained': _cfg(),
    'shufflenetv2_100.untrained': _cfg(),
    'shufflenetv2_150.untrained': _cfg(),
    'shufflenetv2_200.untrained': _cfg(),
})

@register_model
def shufflenetv1_050_g3(pretrained=False, **kwargs) -> ShuffleNetV1:
    """ ShuffleNetV1-0.5x """
    model = _create_shufflenetv1('shufflenetv1_050_g3', width=0.5, group=3, pretrained=pretrained, **kwargs)
    return model

@register_model
def shufflenetv1_100_g3(pretrained=False, **kwargs) -> ShuffleNetV1:
    """ ShuffleNetV1-1.0x """
    model = _create_shufflenetv1('shufflenetv1_100_g3', width=1.0, group=3, pretrained=pretrained, **kwargs)
    return model

@register_model
def shufflenetv1_150_g3(pretrained=False, **kwargs) -> ShuffleNetV1:
    """ ShuffleNetV1-1.5x """
    model = _create_shufflenetv1('shufflenetv1_150_g3', width=1.5, group=3, pretrained=pretrained, **kwargs)
    return model

@register_model
def shufflenetv1_200_g3(pretrained=False, **kwargs) -> ShuffleNetV1:
    """ ShuffleNetV1-2.0x """
    model = _create_shufflenetv1('shufflenetv1_200_g3', width=2.0, group=3, pretrained=pretrained, **kwargs)
    return model

@register_model
def shufflenetv1_050_g8(pretrained=False, **kwargs) -> ShuffleNetV1:
    """ ShuffleNetV1-0.5x """
    model = _create_shufflenetv1('shufflenetv1_050_g8', width=0.5, group=8, pretrained=pretrained, **kwargs)
    return model

@register_model
def shufflenetv1_100_g8(pretrained=False, **kwargs) -> ShuffleNetV1:
    """ ShuffleNetV1-1.0x """
    model = _create_shufflenetv1('shufflenetv1_100_g8', width=1.0, group=8, pretrained=pretrained, **kwargs)
    return model

@register_model
def shufflenetv1_150_g8(pretrained=False, **kwargs) -> ShuffleNetV1:
    """ ShuffleNetV1-1.5x """
    model = _create_shufflenetv1('shufflenetv1_150_g8', width=1.5, group=8, pretrained=pretrained, **kwargs)
    return model

@register_model
def shufflenetv1_200_g8(pretrained=False, **kwargs) -> ShuffleNetV1:
    """ ShuffleNetV1-2.0x """
    model = _create_shufflenetv1('shufflenetv1_200_g8', width=2.0, group=8, pretrained=pretrained, **kwargs)
    return model


@register_model
def shufflenetv2_050(pretrained=False, **kwargs) -> ShuffleNetV2:
    """ ShuffleNetV2-0.5x """
    model = _create_shufflenetv2('shufflenetv2_050', width=0.5, pretrained=pretrained, **kwargs)
    return model

@register_model
def shufflenetv2_100(pretrained=False, **kwargs) -> ShuffleNetV2:
    """ ShuffleNetV2-1.0x """
    model = _create_shufflenetv2('shufflenetv2_100', width=1.0, pretrained=pretrained, **kwargs)
    return model

@register_model
def shufflenetv2_150(pretrained=False, **kwargs) -> ShuffleNetV2:
    """ ShuffleNetV2-1.5x """
    model = _create_shufflenetv2('shufflenetv2_150', width=1.5, pretrained=pretrained, **kwargs)
    return model

@register_model
def shufflenetv2_200(pretrained=False, **kwargs) -> ShuffleNetV2:
    """ ShuffleNetV2-2.0x """
    model = _create_shufflenetv2('shufflenetv2_200', width=2.0, pretrained=pretrained, **kwargs)
    return model

def create_shufflenet_timm(name):
    import timm
    return timm.create_model(name, pretrained=False)


if __name__ == "__main__":
    import timm
    torch.manual_seed(0)
    test_data = torch.rand(5, 3, 224, 224)
    model = timm.create_model('shufflenetv1_150_g8', pretrained=False)
    model.eval()
    test_outputs = model(test_data)