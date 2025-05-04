import random

class SingleArchitectureSearchSpace:
    ''' Search space for single-label classification architectures, same as multi rn'''
    def __init__(self):
        self.base_architectures = ['seresnet50', 'xception']
        self.num_layers = [0, 2, 5, 6]
        self.num_filters = [256, 512, 1024, 2048]
        self.filter_sizes = [3, 5]
        self.fc_sizes = [512, 1024, 2048]
        self.reduction_ratios = [4, 8, 16]
        self.vgg_configs = ['A', 'B', 'D', 'E']
        
    def sample(self):
        return {
            'base_architecture': random.choice(self.base_architectures),
            'num_layers': random.choice(self.num_layers),
            'num_filters': [random.choice(self.num_filters) for _ in range(max(self.num_layers))],
            'filter_sizes': [random.choice(self.filter_sizes) for _ in range(max(self.num_layers))],
            'fc_size': random.choice(self.fc_sizes),
            'reduction_ratio': random.choice(self.reduction_ratios),
            'vgg_config': random.choice(self.vgg_configs)
        }

