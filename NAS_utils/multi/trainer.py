# from dataloading import next_batch_multi
# from utils import save_architecture
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
from tqdm import tqdm
from .architecture import FlexibleMultiLabelCNN, FlexibleFireCNN
import numpy as np
import random

class FireNASTrainer:
    def __init__(self, search_space, controller, model_path, batch_size=32, performance_estimate_epochs=3, subset_fraction=0.2):
        self.search_space = search_space
        self.controller = controller
        self.model_path = model_path
        self.batch_size = batch_size
        self.pecs = performance_estimate_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.subset_fraction = subset_fraction

    def train_model(self, architecture, train_loader):
        model = FlexibleFireCNN(architecture, num_classes=2).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        model.train()
        for epoch in range(self.pecs):
            total_loss = 0
            for batch in train_loader:
                rgb = batch['rgb'].to(self.device)
                thermal = batch['thermal'].to(self.device)
                labels = batch['labels'].to(self.device)
                optimizer.zero_grad()
                logits = model(rgb, thermal)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        return -avg_loss  # Lower loss is better

    def train(self, full_train_loader, num_iterations=20):
        self.controller = self.controller.to(self.device)
        optimizer = torch.optim.Adam(self.controller.parameters())
        best_score = float('-inf')
        best_architecture = None
        best_model = None
        for iteration in tqdm(range(num_iterations), desc='NAS iterations'):
            x = torch.zeros(1, dtype=torch.long).to(self.device)
            hidden = (torch.zeros(1, 100).to(self.device), torch.zeros(1, 100).to(self.device))
            architecture, _, log_prob = self.controller(x, hidden)

            # Sample a subset of the data for this architecture
            subset_size = int(self.subset_fraction * len(full_train_loader.dataset))
            indices = random.sample(range(len(full_train_loader.dataset)), subset_size)
            subset = torch.utils.data.Subset(full_train_loader.dataset, indices)
            subset_loader = torch.utils.data.DataLoader(subset, batch_size=self.batch_size, shuffle=True, num_workers=2)

            score = self.train_model(architecture, subset_loader)
            reward = torch.tensor([score]).to(self.device)
            loss = -log_prob * reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if score > best_score:
                best_score = score
                best_architecture = architecture
                best_model = FlexibleFireCNN(best_architecture, num_classes=2)
                best_model.load_state_dict(best_model.state_dict())
        torch.save(best_model.state_dict(), self.model_path)
        return best_model, best_architecture

# class NASTrainer:
#     def __init__(self, search_space, controller, model_path, architecture_path, batch_size=32, num_classes=9, performance_estimate_epochs=6):
#         self.search_space = search_space
#         self.controller = controller
#         self.num_classes = num_classes
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model_path = model_path
#         self.architecture_path = architecture_path
#         self.batch_size = batch_size
#         self.pecs = performance_estimate_epochs
        
#     def train_model(self, architecture, X_train, Y_train, X_test, Y_test):

#         model = FlexibleMultiLabelCNN(architecture, num_classes=self.num_classes)
#         model = model.to(self.device)
        
#         backbone_params = []
#         new_params = []
        
#         for name, param in model.named_parameters():
#             if 'backbone' in name:
#                 backbone_params.append(param)
#             else:
#                 new_params.append(param)
                
#         optimizer = optim.Adam([
#             {'params': backbone_params, 'lr': 1e-5},
#             {'params': new_params, 'lr': 1e-3}
#         ])
        
#         criterion = nn.BCELoss()
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=6)
        
#         best_val_loss = float('inf')
#         best_accuracy = 0.0
#         patience = 6
#         patience_counter = 0

#         for epoch in range(self.pecs):
#             # Training phase
#             model.train()
#             total_loss = 0
#             train_correct = 0
#             train_total = 0
            
#             train_size = len(X_train)
            
#             for i in range(train_size // self.batch_size):
#                 batch_xs, batch_ys = next_batch_multi(X_train, Y_train, 
#                                                     self.batch_size, i, train_size)
#                 data, target = batch_xs.to(self.device), batch_ys.to(self.device)
                
#                 optimizer.zero_grad()
#                 _, output = model(data)
#                 loss = criterion(output, target)
#                 loss.backward()
#                 optimizer.step()
                
#                 total_loss += loss.item()
#                 predictions = (output > 0.5).float()
#                 train_correct += (predictions == target).float().sum().item() / self.num_classes
#                 train_total += target.size(0)
                
#                 del batch_xs, batch_ys
#                 gc.collect()
            
#             avg_train_loss = total_loss / (train_size // self.batch_size)
#             train_accuracy = train_correct / train_total
            
#             # Validation phase
#             model.eval()
#             val_loss = 0
#             val_correct = 0
#             val_total = 0
            
#             test_size = len(X_test)
            
#             with torch.no_grad():
#                 for i in range(test_size // self.batch_size):
#                     batch_xs, batch_ys = next_batch_multi(X_test, Y_test, 
#                                                         self.batch_size, i, test_size)
#                     data, target = batch_xs.to(self.device), batch_ys.to(self.device)
#                     _, output = model(data)
#                     val_loss += criterion(output, target).item()
                    
#                     predictions = (output > 0.5).float()
#                     val_correct += (predictions == target).float().sum().item() / self.num_classes
#                     val_total += target.size(0)
                    
#                     del batch_xs, batch_ys
#                     gc.collect()
            
#             avg_val_loss = val_loss / (test_size // self.batch_size)
#             val_accuracy = val_correct / val_total
            
#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 best_accuracy = val_accuracy
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
#                 if patience_counter >= patience:
#                     break
        
#         return best_accuracy

#     def train(self, X_train, Y_train, X_val, Y_val, num_iterations=50):
#         self.controller = self.controller.to(self.device)
#         optimizer = torch.optim.Adam(self.controller.parameters())
        
#         best_accuracy = 0
#         best_model = None
#         best_architecture = None
#         print("\nStarting Neural Architecture Search")
#         print("=" * 35)
#         nas_bar = tqdm(range(num_iterations), desc='NAS iterations')
#         for iteration in nas_bar:
            
#             x = torch.zeros(1, dtype=torch.long).to(self.device)
#             hidden = (torch.zeros(1, 100).to(self.device), torch.zeros(1, 100).to(self.device))
#             architecture, _, log_prob = self.controller(x, hidden)
            
#             accuracy = self.train_model(architecture, X_train, Y_train, X_val, Y_val)
            
#             reward = torch.tensor([accuracy]).to(self.device)
#             loss = -log_prob * reward
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             if accuracy > best_accuracy:
#                 best_accuracy = accuracy
#                 best_architecture = architecture
#                 best_model = FlexibleMultiLabelCNN(best_architecture, num_classes=self.num_classes)
#                 best_model.load_state_dict(best_model.state_dict())
#             nas_bar.set_postfix(accuracy=accuracy, best_accuracy=best_accuracy)
        
#         print("\nNeural Architecture Search completed!")
#         print("=" * 35)
#         self.print_architecture(best_architecture, "Best Architecture Found")
#         print(f"Best Accuracy Achieved: {best_accuracy:.4f}")

#         # Save the best model
#         torch.save(best_model.state_dict(), self.model_path)
#         print(f"Best model saved to {self.model_path}")

#         save_architecture(self.architecture_path, best_architecture, best_accuracy)
        
#         print(f"Architecture details saved to {self.architecture_path}")
        
#         return best_model
    
#     def print_architecture(self, architecture, title="Architecture Details"):
#         """
#         Helper method to print architecture details in a formatted way
#         """
#         print(f"\n{title}")
#         print("=" * len(title))
#         print(f"Base Architecture: {architecture['base_architecture']}")
#         print(f"Number of Layers: {architecture['num_layers']}")
#         print("\nLayer Details:")
#         for i in range(architecture['num_layers']):
#             print(f"  Layer {i+1}:")
#             print(f"    - Filters: {architecture['num_filters'][i]}")
#             print(f"    - Filter Size: {architecture['filter_sizes'][i]}")
#         print(f"\nFC Layer Size: {architecture['fc_size']}")
        
#         # Print architecture-specific parameters
#         if architecture['base_architecture'] == 'seresnet50':
#             print(f"SE Reduction Ratio: {architecture['reduction_ratio']}")
#         if architecture['base_architecture'] == 'vgg16':
#             print(f"VGG Configuration: {architecture['vgg_config']}")

#     def evaluate(self, model, X_test, Y_test, model_path=None):
#         model = model.to(self.device)
#         model.eval()

#         if model_path:
#             print(f"Loading model from {model_path}")
#             model.load_state_dict(torch.load(model_path))

#         test_batch_size = 48
#         test_size = len(X_test)
#         all_features = []
#         correct_predictions = 0
#         total_predictions = 0
        
#         print("\nStarting evaluation...")
        
#         with torch.no_grad():
#             for i in range(test_size // test_batch_size):
#                 batch_xs, batch_ys = next_batch_multi(X_test, Y_test, 
#                                                     test_batch_size, i, test_size)
#                 batch_xs = batch_xs.to(self.device)
#                 batch_ys = batch_ys.to(self.device)
                
#                 features, logits = model(batch_xs)
                
#                 features_np = features.cpu().numpy()
#                 all_features.append(features_np)
                
#                 predictions = (logits > 0.5).float()
#                 correct_predictions += (predictions == batch_ys).sum().item()
#                 total_predictions += batch_ys.numel()
#         accuracy = correct_predictions / total_predictions

#         print(f'Test Accuracy: {accuracy:.4f}')
        
        
#         all_features = np.vstack(all_features)
        
#         return all_features, accuracy