import torch
import torch.nn as nn
import torch.hub as hub

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class DeformationDetector(nn.Module):
    def __init__(self):
        super(DeformationDetector, self).__init__()
        self.yolov5 = hub.load('ultralytics/yolov5',
                               'yolov5l', autoshape=False, pretrained=False)

    def forward(self, x):
        return self.yolov5(x)


model = DeformationDetector().to(DEVICE)
result = model(torch.randn((1, 3, 1280, 1280)).to(DEVICE))
