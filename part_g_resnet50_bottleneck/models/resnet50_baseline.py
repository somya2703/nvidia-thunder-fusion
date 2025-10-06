import torch
import torchvision
import os

def get_resnet50_baseline(pretrained=False, device='cuda'):
    model = torchvision.models.resnet50(pretrained=pretrained)
    model.eval()
    model.to(device)
    return model