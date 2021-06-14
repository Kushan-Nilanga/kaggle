import torch
import torch.nn as nn
import torch.hub as hub


class DeformationDetector(nn.Module):
    def __init__(self):
        super(DeformationDetector, self).__init__()
        self.yolov5 = hub.load('ultralytics/yolov5',
                               'yolov5l', autoshape=False, pretrained=False)

    def forward(self, x):
        return self.yolov5(x)


model = DeformationDetector()

result = model(torch.randn((16, 3, 1280, 1280)))
print(len(result[0]))