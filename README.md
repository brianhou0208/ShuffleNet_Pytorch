# ShuffleNet and ShuffleNetV2
> [!NOTE]
This is unofficial implement repo
- Reimplement ShuffleNet and ShuffleNetV2 for timm style
- Mapping original model weight for timm style model

## Original Pretrain Weight
- OneDrive download: [Link](https://1drv.ms/f/s!AgaP37NGYuEXhRfQxHRseR7eSxXo)
- BaiduYun download: [Link](https://pan.baidu.com/s/1EUQVoFPb74yZm0JWHKjFOw) (extract code: mc24)

## Quick Start

```python
import torch
from shufflenet_original import create_shufflenet_original
from shufflenet_timm import create_shufflenet_timm

x = torch.rand(1, 3, 224, 224)

model_original = create_shufflenet_original('v2', '1.0x', None).eval()
model_timm = create_shufflenet_timm('shufflenetv2_100').eval()

y_original = model_original(x)
y_timm = model_timm(x)

print('All close: ', torch.allclose(y_timm, y_original))
>> All close:  True

print('MAE: ', torch.mean(torch.abs(y_timm - y_original)).item())
>> MAE:  0.0

print('Original Model')
flops_params(model_original)
>> | FLOPs:297.368 MFLOPS   MACs:145.814 MMACs   Params:2.2788 M 

print('Timm Model')
flops_params(model_timm)
>> | FLOPs:297.368 MFLOPS   MACs:145.814 MMACs   Params:2.2788 M 
```

## Usage

```bash
python valid.py
```

```bash
python mapping_weight.py
```

## Results

- requirement package [calflops](https://github.com/MrYxJ/calculate-flops.pytorch)

|            Model            |  FLOPs  |  MACs   | Params | Top-1 | Top-5 |
|:---------------------------:|:-------:|:-------:|:------:|:-----:|:-----:|
| ShuffleNetV1 0.5x (group=3) | 79.82M  | 37.59M  | 0.72M  | 57.3  | 80.0  |
| ShuffleNetV1 0.5x (group=8) | 88.20M  | 40.47M  | 1.01M  | 58.8  | 81.0  |
|      ShuffleNetV2 0.5x      | 84.17M  | 40.48M  | 1.37M  | 61.1  | 82.6  |
| ShuffleNetV1 1.0x (group=3) | 284.20M | 137.46M | 1.86M  | 67.8  | 87.7  |
| ShuffleNetV1 1.0x (group=8) | 290.84M | 138.36M | 2.43M  | 68.0  | 86.4  |
|      ShuffleNetV2 1.0x      | 297.37M | 145.81M | 2.28M  | 69.4  | 88.9  |
| ShuffleNetV1 1.5x (group=3) | 598.19M | 292.44M | 3.44M  | 71.6  | 90.2  |
| ShuffleNetV1 1.5x (group=8) | 600.93M | 290.19M | 4.27M  | 71.0  | 89.6  |
|      ShuffleNetV2 1.5x      | 605.90M | 298.97M | 3.50M  | 72.6  | 90.6  |
| ShuffleNetV1 2.0x (group=3) |  1.07G  | 524.05M | 5.45M  | 74.1  | 91.4  |
| ShuffleNetV1 2.0x (group=8) |  1.07G  | 521.94M | 6.52M  | 72.9  | 90.8  |
|      ShuffleNetV2 2.0x      |  1.19G  | 590.78M | 7.40M  | 75.0  | 92.4  |

## Acknowledgement
This repository is built using the [timm](https://github.com/huggingface/pytorch-image-models) and [Megvii Research](https://github.com/megvii-model/ShuffleNet-Series)

## Citation
```
@inproceedings{zhang2018shufflenet,
    title={Shufflenet: An extremely efficient convolutional neural network for mobile devices},
    author={Zhang, Xiangyu and Zhou, Xinyu and Lin, Mengxiao and Sun, Jian},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={6848--6856},
    year={2018}
}
```
```
@inproceedings{ma2018shufflenet, 
    title={Shufflenet v2: Practical guidelines for efficient cnn architecture design},  
    author={Ma, Ningning and Zhang, Xiangyu and Zheng, Hai-Tao and Sun, Jian},  
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},  
    pages={116--131}, 
    year={2018} 
}
```