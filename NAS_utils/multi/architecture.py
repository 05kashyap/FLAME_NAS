import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class FlexibleFireCNN(nn.Module):
    def __init__(self, architecture, num_classes=2):
        super(FlexibleFireCNN, self).__init__()
        self.architecture = architecture
        self.num_classes = num_classes

        # RGB stream
        self.rgb_layers = self._make_stream(in_channels=3)
        # Thermal stream
        self.thermal_layers = self._make_stream(in_channels=1)

        # Calculate feature map size after convolutions (assume input 224x224, 2x2 pooling per layer)
        feature_size = 224 // (2 ** architecture['num_layers'])
        last_filters = architecture['num_filters'][architecture['num_layers']-1]
        self.feature_dim = feature_size * feature_size * last_filters

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * 2, architecture['fc_size']),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(architecture['fc_size'], num_classes)
        )

    def _make_stream(self, in_channels):
        layers = []
        for i in range(self.architecture['num_layers']):
            out_channels = self.architecture['num_filters'][i]
            kernel_size = self.architecture['filter_sizes'][i]
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, rgb, thermal):
        rgb_feat = self.rgb_layers(rgb)
        thermal_feat = self.thermal_layers(thermal)
        rgb_feat = rgb_feat.view(rgb_feat.size(0), -1)
        thermal_feat = thermal_feat.view(thermal_feat.size(0), -1)
        combined = torch.cat([rgb_feat, thermal_feat], dim=1)
        logits = self.classifier(combined)
        return logits
