import torch
import torch.nn.functional as F
import torch.nn as nn

class FireArchitectureController(nn.Module):
    def __init__(self, search_space):
        super(FireArchitectureController, self).__init__()
        self.search_space = search_space
        self.lstm = nn.LSTMCell(100, 100)
        self.base_arch_embed = nn.Embedding(len(search_space.base_architectures), 100)
        self.num_layers_embed = nn.Embedding(len(search_space.num_layers), 100)
        self.num_filters_embed = nn.Embedding(len(search_space.num_filters), 100)
        self.filter_sizes_embed = nn.Embedding(len(search_space.filter_sizes), 100)
        self.fc_sizes_embed = nn.Embedding(len(search_space.fc_sizes), 100)
        self.base_arch_out = nn.Linear(100, len(search_space.base_architectures))
        self.num_layers_out = nn.Linear(100, len(search_space.num_layers))
        self.num_filters_out = nn.Linear(100, len(search_space.num_filters))
        self.filter_sizes_out = nn.Linear(100, len(search_space.filter_sizes))
        self.fc_sizes_out = nn.Linear(100, len(search_space.fc_sizes))

    def forward(self, x, hidden):
        outputs = {}
        log_probs = []
        h, c = hidden

        # Base architecture
        embed = self.base_arch_embed(x)
        h, c = self.lstm(embed, (h, c))
        logits = self.base_arch_out(h)
        probs = F.softmax(logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).squeeze()
        log_probs.append(log_prob[0, action])
        outputs['base_architecture'] = self.search_space.base_architectures[action.item()]

        # Number of layers
        embed = self.num_layers_embed(x)
        h, c = self.lstm(embed, (h, c))
        logits = self.num_layers_out(h)
        probs = F.softmax(logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).squeeze()
        log_probs.append(log_prob[0, action])
        num_layers = self.search_space.num_layers[action.item()]
        outputs['num_layers'] = num_layers

        # Filters and filter sizes
        filters = []
        filter_sizes = []
        for _ in range(num_layers):
            embed = self.num_filters_embed(x)
            h, c = self.lstm(embed, (h, c))
            logits = self.num_filters_out(h)
            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze()
            log_probs.append(log_prob[0, action])
            filters.append(self.search_space.num_filters[action.item()])

            embed = self.filter_sizes_embed(x)
            h, c = self.lstm(embed, (h, c))
            logits = self.filter_sizes_out(h)
            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze()
            log_probs.append(log_prob[0, action])
            filter_sizes.append(self.search_space.filter_sizes[action.item()])

        outputs['num_filters'] = filters
        outputs['filter_sizes'] = filter_sizes

        # FC size
        embed = self.fc_sizes_embed(x)
        h, c = self.lstm(embed, (h, c))
        logits = self.fc_sizes_out(h)
        probs = F.softmax(logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).squeeze()
        log_probs.append(log_prob[0, action])
        outputs['fc_size'] = self.search_space.fc_sizes[action.item()]

        return outputs, (h, c), torch.stack(log_probs).sum()

