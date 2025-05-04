from dataloading import next_batch_single
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from tqdm import tqdm
from .architecture import SingleLabelCNN
import gc
import numpy as np
import xlwt
from utils import save_architecture, load_architecture_from_file, compute_metrics, get_confusion_matrix
import random

def compute_accuracy(logits, labels):
    """Compute the number of correct predictions."""
    predicted = torch.argmax(logits, dim=1)
    actual = torch.argmax(labels, dim=1)
    correct = (predicted == actual).sum().item()
    return correct

class SingleLabelNASTrainer:
    #TODO: Add eval function
    def __init__(self, search_space, controller, model_path, architecture_path, batch_size=32, num_classes=9, performance_estimate_epochs=6):
        self.search_space = search_space
        self.controller = controller
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.architecture_path = architecture_path
        self.batch_size = batch_size
        self.pecs = performance_estimate_epochs
        
    def train_model(self, architecture, X_train, Y_train, X_val, Y_val):
        model = SingleLabelCNN(architecture, num_classes=self.num_classes)
        model = model.to(self.device)
        
        backbone_params = []
        new_params = []
        
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                new_params.append(param)
                
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': 0.1},
            {'params': new_params, 'lr': 0.1}
        ])
        
        criterion = nn.CrossEntropyLoss()  # Changed from BCELoss for single-label        
        best_loss = float('inf')
        best_accuracy = 0.0
        patience = 6
        patience_counter = 0

        for epoch in range(self.pecs):
            model.train()
            total_loss = 0
            correct = 0
            train_total = 0
            
            train_size = len(X_train)
            
            for i in range(train_size // self.batch_size):
                batch_xs, batch_ys = next_batch_single(X_train, Y_train, 
                                                    self.batch_size, i, train_size)
                batch_xs = batch_xs.to(self.device)
                batch_ys = batch_ys.to(self.device)
                
                optimizer.zero_grad()
                _, logits = model(batch_xs)
                loss = criterion(logits, torch.argmax(batch_ys, dim=1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == torch.argmax(batch_ys, dim=1)).sum().item()

                
                del batch_xs, batch_ys
                gc.collect()
            
            avg_loss = total_loss / (train_size // self.batch_size)
            accuracy = correct / train_size
            
            # Validation phase
            model.eval()
            # val_loss = 0
            # val_correct = 0
            # val_total = 0
            
            # val_size = len(X_val)
            
            # with torch.no_grad():
            #     for i in range(val_size // self.batch_size):
            #         batch_xs, batch_ys = next_batch_single(X_val, Y_val, 
            #                                             self.batch_size, i, val_size)
            #         data = batch_xs.to(self.device)
            #         target = torch.argmax(batch_ys, dim=1).to(self.device)
                    
            #         _, output = model(data)
            #         val_loss += criterion(output, target).item()
                    
            #         _, predicted = torch.max(output.data, 1)
            #         val_correct += (predicted == target).sum().item()
            #         val_total += target.size(0)
                    
            #         del batch_xs, batch_ys
            #         gc.collect()
            
            # avg_val_loss = val_loss / (val_size // self.batch_size)
            # val_accuracy = val_correct / val_total
                        
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_accuracy = accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return best_accuracy

    def train(self, X_train, Y_train, X_val, Y_val, num_iterations=50, epochs=32):
        self.controller = self.controller.to(self.device)
        optimizer = torch.optim.Adam(self.controller.parameters())
        best_accuracy = 0
        best_model = None
        best_architecture = None
        
        print("\nStarting Single-Label Neural Architecture Search")
        print("=" * 45)
        
        nas_bar = tqdm(range(num_iterations), desc='NAS iterations')
        for iteration in nas_bar:
            x = torch.zeros(1, dtype=torch.long).to(self.device)
            hidden = (torch.zeros(1, 100).to(self.device), torch.zeros(1, 100).to(self.device))
            
            architecture, _, log_prob = self.controller(x, hidden)
            accuracy = self.train_model(architecture, X_train, Y_train, X_val, Y_val)
            
            reward = torch.tensor([accuracy]).to(self.device)
            loss = -log_prob * reward
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_architecture = architecture
                best_model = SingleLabelCNN(best_architecture, num_classes=self.num_classes)
                best_model.load_state_dict(best_model.state_dict())
            
            nas_bar.set_postfix(accuracy=accuracy, best_accuracy=best_accuracy)
        
        print("\nSingle-Label Neural Architecture Search completed!")
        print("=" * 45)
        self.print_architecture(best_architecture, "Best Architecture Found")
        print(f"Best Accuracy Achieved: {best_accuracy:.4f}")
        
        # Save the best model
        now_time = str(datetime.datetime.now())
        # model_path = f'./model_saving/nas_single_{now_time}.pth'
        torch.save(best_model.state_dict(), self.model_path)
        print(f"Best model saved to {self.model_path}")
        
        save_architecture(self.architecture_path, best_architecture, best_accuracy)
        
        print(f"Architecture details saved to {self.architecture_path}")

        return best_model

    def print_architecture(self, architecture, title="Architecture Details"):
        print(f"\n{title}")
        print("=" * len(title))
        print(f"Base Architecture: {architecture['base_architecture']}")
        print(f"Number of Layers: {architecture['num_layers']}")
        print("\nLayer Details:")
        for i in range(architecture['num_layers']):
            print(f"  Layer {i+1}:")
            print(f"    - Filters: {architecture['num_filters'][i]}")
            print(f"    - Filter Size: {architecture['filter_sizes'][i]}")
        print(f"\nFC Layer Size: {architecture['fc_size']}")
        
        if architecture['base_architecture'] == 'seresnet50':
            print(f"SE Reduction Ratio: {architecture['reduction_ratio']}")
        if architecture['base_architecture'] == 'vgg16':
            print(f"VGG Configuration: {architecture['vgg_config']}")

    def evaluate(self, model, X_test, Y_test, model_path=None):
        model = model.to(self.device)
        model.eval()
        
        if model_path:
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path))

        test_batch_size = 48
        test_size = len(X_test)
        all_features = []
        total_correct = 0
        confusion_matrix = torch.zeros(9, 9)
        
        print('Start to evaluate test data')
        with torch.no_grad():
            # Ensure we process all samples by adjusting the range
            num_batches = (test_size + test_batch_size - 1) // test_batch_size
            
            for i in range(num_batches):
                # Calculate the actual batch size for the last batch
                start_idx = i * test_batch_size
                end_idx = min((i + 1) * test_batch_size, test_size)
                current_batch_size = end_idx - start_idx
                
                # Get batch
                batch_xs, batch_ys = next_batch_single(X_test[start_idx:end_idx], 
                                                    Y_test[start_idx:end_idx], 
                                                    current_batch_size, 0, current_batch_size)
                
                # Move to device
                batch_xs = batch_xs.to(self.device)
                batch_ys = batch_ys.to(self.device)
                
                # Get features and predictions
                features, logits = model(batch_xs)
                
                # Store features
                all_features.append(features.cpu().numpy())
                
                # Compute accuracy
                patch_correct = compute_accuracy(logits, batch_ys)
                total_correct += patch_correct
                
                _, predictions = torch.max(logits, 1)
                true_labels = torch.argmax(batch_ys, dim=1)

                confusion_matrix += get_confusion_matrix(predictions, true_labels, 9)
                print(f"{i}/{num_batches}    {patch_correct}")        

        # Reshape features and save to Excel
        all_features = np.concatenate(all_features, axis=0)
        print(f"Feature shape: {all_features.shape[1]}, {all_features.shape[0]}")
        
        # # Save features to Excel
        # book = xlwt.Workbook()
        # sheet1 = book.add_sheet('sheet1', cell_overwrite_ok=True)
        
        # for i in range(all_features.shape[0]):
        #     sheet1.write(i, 0, Y_test[i])
        #     for j in range(all_features.shape[1]):
        #         sheet1.write(i, j+1, float(all_features[i][j]))
        
        # book.save('./NAS_single_feature.xls')
        
        # Print final accuracy
        metrics = compute_metrics(confusion_matrix)

        final_accuracy = total_correct / test_size
        print(f"Final test accuracy: {final_accuracy:.4f}")
        print(f"Sensitivity: {metrics['macro_avg']['sensitivity']:.4f}")
        print(f"Specificity: {metrics['macro_avg']['specificity']:.4f}")
        print(f"F1-Score: {metrics['macro_avg']['f1']:.4f}")
        return all_features, final_accuracy
    
def train_best_architecture_single(architecture, X_train_list, Y_train_list, model_path, lr_value=0.1, epochs=50, batch_size=32, patience=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = SingleLabelCNN(architecture, num_classes=9)
    model = model.to(device)
    
    # backbone_params = []
    # new_params = []
    
    # for name, param in model.named_parameters():
    #     if 'backbone' in name:
    #         backbone_params.append(param)
    #     else:
    #         new_params.append(param)
            
    # optimizer = optim.Adam([
    #     {'params': backbone_params, 'lr': 0.1},
    #     {'params': new_params, 'lr': 0.1}
    # ])
    optimizer = optim.Adadelta(model.parameters(), lr=lr_value)
    criterion = nn.CrossEntropyLoss()  # Changed from BCELoss for single-label
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience)
    
    best_loss = float('inf')
    best_accuracy = 0.0


    train_size = len(X_train_list)
    losses = []
    accuracies = []

    for epoch in range(epochs):
        # Shuffle data
        c = list(zip(X_train_list, Y_train_list))
        random.Random(random.randint(0, 10000)).shuffle(c)
        X_train_list, Y_train_list = zip(*c)
        
        model.train()
        total_loss = 0.0
        correct = 0

        for i in range(train_size // batch_size):
            batch_xs, batch_ys = next_batch_single(X_train_list, Y_train_list, batch_size, i, train_size)

            # Move to device
            batch_xs = batch_xs.to(device)
            batch_ys = batch_ys.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            _, logits = model(batch_xs)

            # Compute loss
            loss = criterion(logits, torch.argmax(batch_ys, dim=1))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            correct += (predicted == torch.argmax(batch_ys, dim=1)).sum().item()
        
        # Calculate average loss and accuracy for the epoch
        avg_loss = total_loss / (train_size // batch_size)
        accuracy = correct / train_size
        now_time = str(datetime.datetime.now())
        print(f"{now_time} epoch_{epoch+1}, training loss {avg_loss}, accuracy {accuracy}")

        scheduler.step(avg_loss)
        # After scheduler.step(avg_loss), add:
        current_lr = optimizer.param_groups[0]['lr']
        if epoch > 0:  # Compare with previous lr
            if current_lr != prev_lr:
                print(f'Learning rate decreased from {prev_lr} to {current_lr}')
        prev_lr = current_lr
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            print("Saving model...")
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

    print('Finished training!')
    print(f"Best accuracy: {best_accuracy}")
    return best_accuracy
