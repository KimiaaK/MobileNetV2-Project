import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV2(nn.Module):
    #initializes the model by loading the pre-trained MobileNetV2.
    #replaces the final layer to match the number of classes in your classification task
    def __init__(self, num_classes):
        super(MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        #This is called during training and inference to get the model’s predictions.
        return self.model(x)
