import torch
from shufflenet_original import create_shufflenet_original
from shufflenet_timm import create_shufflenet_timm

def rename_layer(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def map_weight(m1, m2):
    for k1, k2 in zip(m1.keys(), m2.keys()):
        m1[k1] = m2[k2]
    return m1


if __name__ == "__main__":
    model_original = create_shufflenet_original('v2', '1.0x', None).eval()
    checkpoint = torch.load('/path/original/weight/ShuffleNetV2.1.0x.pth.tar', map_location='cpu')
    state_dict = checkpoint["state_dict"]

    model_timm = create_shufflenet_timm('shufflenetv2_100').eval()

    new_state_dict = rename_layer(state_dict)

    model_original.load_state_dict(new_state_dict)

    s = map_weight(model_timm.state_dict(), model_original.state_dict())
    model_timm.load_state_dict(s)

    x = torch.rand(1, 3, 224, 224)
    y_original = model_original(x)
    y_timm = model_timm(x)
    print('All close: ', torch.allclose(y_timm, y_original))
    print('MAE: ', torch.mean(torch.abs(y_timm - y_original)).item())

    torch.save(model_timm.state_dict(), '/path/save/weight/shufflenetv2_100_in1k.pth.tar')

    model_timm = create_shufflenet_timm('shufflenetv2_100').eval()
    model_timm.load_state_dict(torch.load('/path/save/weight/shufflenetv2_100_in1k.pth.tar'))
    y_timm2 = model_timm(x)
    print('All close: ', torch.allclose(y_timm, y_timm2))
    print('All close: ', torch.allclose(y_original, y_timm2))