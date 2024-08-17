import torch
from shufflenet_original import create_shufflenet_original
from shufflenet_timm import create_shufflenet_timm

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    model_original = create_shufflenet_original('v2', '1.0x', None).eval()
    model_timm = create_shufflenet_timm('shufflenetv2_100').eval()

    y_original = model_original(x)
    y_timm = model_timm(x)

    print('All close: ', torch.allclose(y_timm, y_original))
    print('MAE: ', torch.mean(torch.abs(y_timm - y_original)).item())
