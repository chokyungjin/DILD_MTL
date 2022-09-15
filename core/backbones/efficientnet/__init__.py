from .model import EfficientNet

def build_efficientnet(name, pretrained=False):
    if pretrained:
        model = EfficientNet.from_pretrained(name)
    else:
        model = EfficientNet.from_name(name,in_channels=3)

    return model