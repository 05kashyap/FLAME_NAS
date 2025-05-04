import os
import argparse
import torch
from dataloading import get_fire_dataloaders
from NAS_utils.multi.searchspace import FireArchitectureSearchSpace
from NAS_utils.multi.controller import FireArchitectureController
from NAS_utils.multi.trainer import FireNASTrainer

parser = argparse.ArgumentParser(description='Train NAS fire detection model')
parser.add_argument('--labels-file', type=str, required=True, help='Path to labels file')
parser.add_argument('--base-dir', type=str, required=True, help='Base directory for dataset')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
parser.add_argument('--pecs', type=int, default=3, help='Epochs for performance estimate')
parser.add_argument('--iters', type=int, default=20, help='NAS iterations')
parser.add_argument('--subset-fraction', type=float, default=0.2, help='Fraction of data for each NAS sample')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
PERF_ESTIMATE_EPOCHS = args.pecs
NUM_ITERATIONS = args.iters
SUBSET_FRACTION = args.subset_fraction

# Placeholder paths
labels_file = args.labels_file  # e.g., 'Data/archive/Frame_Pair_Labels.txt'
base_dir = args.base_dir       # e.g., 'Data/archive/'

train_loader, val_loader, test_loader = get_fire_dataloaders(
    labels_file=labels_file,
    base_dir=base_dir,
    batch_size=BATCH_SIZE,
    train_ratio=0.8,
    val_ratio=0.1
)
os.mkdir('model_saving', exist_ok=True)
model_path = 'model_saving/fire_nas_model.pth'
search_space = FireArchitectureSearchSpace()
controller = FireArchitectureController(search_space)
trainer = FireNASTrainer(
    search_space, controller, model_path=model_path,
    batch_size=BATCH_SIZE, performance_estimate_epochs=PERF_ESTIMATE_EPOCHS,
    subset_fraction=SUBSET_FRACTION
)

print("Starting NAS for fire detection...")
best_model, best_architecture = trainer.train(train_loader, num_iterations=NUM_ITERATIONS)
print("NAS completed. Best architecture:", best_architecture)