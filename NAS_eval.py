import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataloading import get_fire_dataloaders
from NAS_utils.multi.architecture import FlexibleFireCNN
import argparse
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

parser = argparse.ArgumentParser(description='Train fire detection model')
parser.add_argument('--labels-file', type=str, required=True, help='Path to labels file')
parser.add_argument('--base-dir', type=str, required=True, help='Base directory for dataset')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
parser.add_argument('--save_path', type=str,default="/kaggle/working/", help='Path to save the model')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
args = parser.parse_args()

BATCH_SIZE = args.batch_size

# Paths (edit as needed)
save_path = args.save_path
arch_path = save_path + '/model_saving/fire_nas_architecture.json'
model_path = save_path + '/model_saving/fire_nas_model.pth'
labels_file = args.labels_file  # e.g., 'Data/archive/Frame_Pair_Labels.txt'
base_dir = args.base_dir       # e.g., 'Data/archive/'
BATCH_SIZE = args.batch_size  # e.g., 32
EPOCHS = args.epochs  # e.g., 1

os.makedirs(save_path + '/model_saving', exist_ok=True)

# Load architecture
with open(arch_path, 'r') as f:
    best_architecture = json.load(f)

# Get dataloaders
train_loader, val_loader, test_loader = get_fire_dataloaders(
    labels_file=labels_file,
    base_dir=base_dir,
    batch_size=BATCH_SIZE,
    train_ratio=0.8,
    val_ratio=0.1
)

# Merge train and val for full training
from torch.utils.data import ConcatDataset, DataLoader
full_train_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
full_train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Train the model from scratch on full training set
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FlexibleFireCNN(best_architecture, num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in full_train_loader:
        rgb = batch['rgb'].to(device)
        thermal = batch['thermal'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        logits = model(rgb, thermal)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(full_train_loader):.4f}")

# Save the retrained model
torch.save(model.state_dict(), model_path)

# Evaluate on test set
model.eval()
all_fire_preds = []
all_smoke_preds = []
all_fire_labels = []
all_smoke_labels = []

with torch.no_grad():
    for batch in test_loader:
        rgb_imgs = batch['rgb'].to(device)
        thermal_imgs = batch['thermal'].to(device)
        labels = batch['labels']
        outputs = model(rgb_imgs, thermal_imgs)
        preds = torch.sigmoid(outputs)
        all_fire_preds.extend((preds[:, 0] > 0.5).cpu().numpy())
        all_smoke_preds.extend((preds[:, 1] > 0.5).cpu().numpy())
        all_fire_labels.extend(labels[:, 0].numpy())
        all_smoke_labels.extend(labels[:, 1].numpy())

fire_accuracy = sum(1 for p, l in zip(all_fire_preds, all_fire_labels) if p == l) / len(all_fire_preds)
smoke_accuracy = sum(1 for p, l in zip(all_smoke_preds, all_smoke_labels) if p == l) / len(all_smoke_preds)

print(f"Test Fire Detection Accuracy: {fire_accuracy:.4f}")
print(f"Test Smoke Detection Accuracy: {smoke_accuracy:.4f}")

fire_cm = confusion_matrix(all_fire_labels, all_fire_preds)
smoke_cm = confusion_matrix(all_smoke_labels, all_smoke_preds)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(fire_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Fire Detection Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 2, 2)
sns.heatmap(smoke_cm, annot=True, fmt='d', cmap='Oranges')
plt.title('Smoke Detection Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()

# ROC Curves
all_fire_probs = np.array(all_fire_preds)
all_smoke_probs = np.array(all_smoke_preds)
fpr_fire, tpr_fire, _ = roc_curve(all_fire_labels, all_fire_probs)
fpr_smoke, tpr_smoke, _ = roc_curve(all_smoke_labels, all_smoke_probs)
auc_fire = auc(fpr_fire, tpr_fire)
auc_smoke = auc(fpr_smoke, tpr_smoke)

plt.figure(figsize=(8, 6))
plt.plot(fpr_fire, tpr_fire, label=f'Fire ROC (AUC = {auc_fire:.2f})')
plt.plot(fpr_smoke, tpr_smoke, label=f'Smoke ROC (AUC = {auc_smoke:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.savefig('roc_curves.png')
plt.show()

# Class-wise Accuracy Bar Plot
fire_tp = fire_cm[1, 1]
fire_tn = fire_cm[0, 0]
fire_fp = fire_cm[0, 1]
fire_fn = fire_cm[1, 0]
smoke_tp = smoke_cm[1, 1]
smoke_tn = smoke_cm[0, 0]
smoke_fp = smoke_cm[0, 1]
smoke_fn = smoke_cm[1, 0]

fire_sensitivity = fire_tp / (fire_tp + fire_fn) if (fire_tp + fire_fn) > 0 else 0
fire_specificity = fire_tn / (fire_tn + fire_fp) if (fire_tn + fire_fp) > 0 else 0
smoke_sensitivity = smoke_tp / (smoke_tp + smoke_fn) if (smoke_tp + smoke_fn) > 0 else 0
smoke_specificity = smoke_tn / (smoke_tn + smoke_fp) if (smoke_tn + smoke_fp) > 0 else 0

labels = ['Fire Sensitivity', 'Fire Specificity', 'Smoke Sensitivity', 'Smoke Specificity']
values = [fire_sensitivity, fire_specificity, smoke_sensitivity, smoke_specificity]

plt.figure(figsize=(8, 5))
sns.barplot(x=labels, y=values, palette='viridis')
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Class-wise Sensitivity and Specificity')
plt.savefig('classwise_accuracy.png')
plt.show()


# Visualize some predictions
for batch in test_loader:
    rgb_imgs = batch['rgb'].to(device)
    thermal_imgs = batch['thermal'].to(device)
    labels = batch['labels']
    frame_numbers = batch['frame_number']
    outputs = model(rgb_imgs, thermal_imgs)
    preds = torch.sigmoid(outputs)
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    for i in range(5):
        if i >= len(rgb_imgs):
            break
        rgb_img = rgb_imgs[i].cpu().permute(1, 2, 0).numpy()
        rgb_img = rgb_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        rgb_img = np.clip(rgb_img, 0, 1)
        thermal_img = thermal_imgs[i].cpu().squeeze().numpy()
        true_fire = labels[i, 0].item()
        true_smoke = labels[i, 1].item()
        pred_fire = preds[i, 0].item()
        pred_smoke = preds[i, 1].item()
        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title(f"RGB - Frame {frame_numbers[i]}")
        axes[i, 0].axis('off')
        axes[i, 1].imshow(thermal_img, cmap='inferno')
        axes[i, 1].set_title(f"Thermal - Frame {frame_numbers[i]}")
        axes[i, 1].axis('off')
        axes[i, 2].axis('off')
        axes[i, 2].text(0.1, 0.7, f"True Fire: {true_fire:.0f}, Pred: {pred_fire:.2f}", fontsize=12)
        axes[i, 2].text(0.1, 0.5, f"True Smoke: {true_smoke:.0f}, Pred: {pred_smoke:.2f}", fontsize=12)
        fire_color = 'green' if (pred_fire > 0.5) == true_fire else 'red'
        smoke_color = 'green' if (pred_smoke > 0.5) == true_smoke else 'red'
        axes[i, 2].text(0.1, 0.3, f"Fire: {'Correct' if fire_color == 'green' else 'Incorrect'}", color=fire_color, fontsize=12)
        axes[i, 2].text(0.1, 0.1, f"Smoke: {'Correct' if smoke_color == 'green' else 'Incorrect'}", color=smoke_color, fontsize=12)
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.show()
    break  # Just show one batch
