import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SingleLabelCNN(nn.Module):
    def __init__(self, architecture, num_classes=9, pretrained=True):
        super(SingleLabelCNN, self).__init__()
        self.architecture = architecture
        self.num_classes = num_classes
        self.pretrained = pretrained
        # Define backbone feature dimensions
        self.feature_dims = {
            'vgg16': 512,
            'seresnet50': 2048,
            'xception': 2048
        }
        
        self.backbone = self._create_backbone()
        self.backbone_features = self.feature_dims[architecture['base_architecture']]
        self.additional_layers = self._create_additional_layers()
        final_dim = self._calculate_final_dim()
        
        # Modified classifier for single-label classification (no sigmoid at the end)
        self.feature_extractor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(final_dim, architecture['fc_size']),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(architecture['fc_size'], 128),
            nn.ReLU(inplace=True)
        )
        
        self.final_layer = nn.Linear(128, num_classes)
    
    def _create_backbone(self):
        # Reuse backbone creation from multi-label implementation
        base_arch = self.architecture['base_architecture']
        model = timm.create_model(base_arch, pretrained=self.pretrained)
        
        if base_arch == 'vgg16':
            model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            model = model.features
        elif base_arch == 'seresnet50':
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model = nn.Sequential(*list(model.children())[:-2])
        elif base_arch == 'xception':
            model.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            model = nn.Sequential(*list(model.children())[:-1])
        
        return model

    def _create_additional_layers(self):
        # Reuse additional layers creation from multi-label implementation
        layers = nn.ModuleList()
        in_channels = self.backbone_features
        
        for i in range(self.architecture['num_layers']):
            out_channels = self.architecture['num_filters'][i]
            kernel_size = self.architecture['filter_sizes'][i]
            
            if self.architecture['base_architecture'] == 'seresnet50':
                layers.append(SEBlock(in_channels, out_channels, 
                                   self.architecture['reduction_ratio']))
            else:
                layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ))
            in_channels = out_channels
        
        return layers

    def _calculate_final_dim(self):
        # Reuse dimension calculation from multi-label implementation
        if self.architecture['base_architecture'] == 'vgg16':
            h, w = 7, 7
        else:
            h, w = 1, 1
        
        channels = self.backbone_features
        
        for i in range(self.architecture['num_layers']):
            channels = self.architecture['num_filters'][i]
        
        return channels

    def forward(self, x):
        x = self.backbone(x)
        
        if self.architecture['base_architecture'] != 'vgg16':
            if len(x.shape) == 2:
                batch_size = x.shape[0]
                x = x.view(batch_size, self.backbone_features, 1, 1)
        
        for layer in self.additional_layers:
            x = layer(x)
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        features = self.feature_extractor(x)  # This will be 128-dimensional
        logits = self.final_layer(features)
        return features, logits
    
class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, out_channels, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.backbone(x)
        
        if self.architecture['base_architecture'] != 'vgg16':
            if len(x.shape) == 2:
                batch_size = x.shape[0]
                x = x.view(batch_size, self.backbone_features, 1, 1)
        
        for layer in self.additional_layers:
            x = layer(x)
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        features = self.feature_extractor(x) 
        logits = self.final_layer(features)
        return features, logits