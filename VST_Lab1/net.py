import torch
import torch.nn as nn


def depthwise_separable_conv(in_channels, out_channels, stride):
    return nn.Sequential(
        
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        
        
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class my_network(nn.Module):
    def __init__(self, num_classes=12):
        super(my_network, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            depthwise_separable_conv(32, 64, 1),
            depthwise_separable_conv(64, 128, 2),
            depthwise_separable_conv(128, 128, 1),
            depthwise_separable_conv(128, 256, 2),
            depthwise_separable_conv(256, 256, 1),
            depthwise_separable_conv(256, 512, 2),
            
            depthwise_separable_conv(512, 512, 1),
            depthwise_separable_conv(512, 512, 1),
            depthwise_separable_conv(512, 512, 1),
            depthwise_separable_conv(512, 512, 1),
            depthwise_separable_conv(512, 512, 1),
            
            depthwise_separable_conv(512, 1024, 2),
            depthwise_separable_conv(1024, 1024, 1),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x