import torch

model = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=True)
components = list(model.children())
extractor = torch.nn.Sequential(*list(model.children())[0][:-1])

video = torch.randn(1, 3, 100, 224, 224)
print(extractor)

with torch.no_grad():
    features = extractor(video)
    N, C, T, H, W = features.shape
    features = features.permute(0, 2, 1, 3, 4).reshape(1, T, -1)

    print(features.shape)
