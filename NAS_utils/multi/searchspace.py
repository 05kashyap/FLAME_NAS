import random

class FireArchitectureSearchSpace:
    def __init__(self):
        self.base_architectures = ['simple_cnn', 'resnet18']  # Add more if needed
        self.num_layers = [2, 3, 4]
        self.num_filters = [64, 128, 256, 512]
        self.filter_sizes = [3, 5]
        self.fc_sizes = [128, 256, 512]
        # No reduction ratios or vgg configs for fire detection

    def sample(self):
        return {
            'base_architecture': random.choice(self.base_architectures),
            'num_layers': random.choice(self.num_layers),
            'num_filters': [random.choice(self.num_filters) for _ in range(max(self.num_layers))],
            'filter_sizes': [random.choice(self.filter_sizes) for _ in range(max(self.num_layers))],
            'fc_size': random.choice(self.fc_sizes)
        }
