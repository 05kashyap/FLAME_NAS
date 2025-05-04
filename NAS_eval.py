import os
import argparse

from dataloading import load_data_list_multi, load_data_list_single
from utils import load_architecture_from_file
from eval import  evaluate_and_visualize_single, svm_classifier_single, svm_classifier_multi

from NAS_utils.multi.searchspace import MultiArchitectureSearchSpace
from NAS_utils.multi.trainer import NASTrainer, train_best_architecture_multi
from NAS_utils.multi.architecture import FlexibleMultiLabelCNN

from NAS_utils.single.architecture import SingleLabelCNN
from NAS_utils.single.searchspace import SingleArchitectureSearchSpace
from NAS_utils.single.trainer import SingleLabelNASTrainer, train_best_architecture_single

parser = argparse.ArgumentParser(description='Extensively train and evaluate NAS protein localization model')
parser.add_argument('--mode', type=str, choices=['single', 'multi'],
                   required=True, help='Training mode: single or multi')
parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs for training (default: 100)')
parser.add_argument('--no-train', action='store_true', 
                    help='Skip training phase')
parser.add_argument('--no-eval', action='store_true',
                    help='Skip evaluation phase')
parser.add_argument('--pretrained', action='store_true',
                    help='Use pretrained model')

args = parser.parse_args()

MODE = args.mode
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
TRAIN = not args.no_train
EVAL =  not args.no_eval
PRETRAINED = args.pretrained

# # Define the best architecture found by NAS
# best_architecture = {
#     'base_architecture': 'seresnet50',  # TODO: replace with best found architecture from NAS
#     'num_layers': 6,
#     'num_filters': [512, 1024, 2048, 1024, 512, 256],
#     'filter_sizes': [3, 3, 3, 3, 3, 3],
#     'fc_size': 1024,
#     'reduction_ratio': 8,
#     'vgg_config': 'D' 
# }
model_dir = './model_saving'

# model_path = ""
# architecture_path = ""
# Load data
print("Loading data...")
if MODE == 'single':
    model_save_dir = model_dir + '/single/nas_models/'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_path = os.path.join(model_save_dir, 'proteinModelNAS_single.pth')
    #'model_saving/single/nas_architectures/proteinArchitectureNAS_single.json'
    architecture_path = model_dir + '/single/nas_architectures/'
    architecture_path = os.path.join(architecture_path, 'proteinArchitectureNAS_single.json')
    X_train_list, X_test_list, Y_train_list, Y_test_list = load_data_list_single()
elif MODE == 'multi':
    model_path_last = f'model_saving/multi/nas_models/'
    if not os.path.exists(model_path_last):
        os.makedirs(model_path_last)
    model_path = os.path.join(model_path_last, 'proteinModelNAS_multi.pth')

    architecture_path = 'model_saving/multi/nas_architectures/proteinArchitectureNAS_multi.json'
    X_train_list, X_test_list, Y_train_list, Y_test_list = load_data_list_multi()
else:
    raise ValueError("Invalid mode. Please select either 'single' or 'multi'.")

# Create validation split from training data
# val_size = int(0.1 * len(X_train_list))
# X_val = X_train_list[-val_size:]
# Y_val = Y_train_list[-val_size:]
# X_train = X_train_list[:-val_size]
# Y_train = Y_train_list[:-val_size]

print(f"Training samples: {len(X_train_list)}")
print(f"Test samples: {len(X_test_list)}")

# Train the best architecture for longer
model = None
best_architecture = load_architecture_from_file(architecture_path)
print(f"Best architecture: {best_architecture}")
if MODE == 'single':

    if TRAIN:
        print("\nStarting extended training of best architecture...")

        best_accuracy = train_best_architecture_single(
            architecture=best_architecture,
            X_train_list=X_train_list,
            Y_train_list=Y_train_list,
            model_path=model_path,
            lr_value=0.1,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            patience=15
        )
    model = SingleLabelCNN(best_architecture, num_classes=9, pretrained=PRETRAINED)

    search_space = SingleArchitectureSearchSpace()
    nas_trainer = SingleLabelNASTrainer(search_space, None, model_path="", architecture_path="")

    if EVAL:
        # features, final_accuracy = nas_trainer.evaluate(model, X_test_list, Y_test_list, model_path)

        # evaluate_and_visualize_single(features, final_accuracy, model, X_test_list, Y_test_list, model_path, model_name='NAS_Single')

        svm_scores, svm_acc = svm_classifier_single(model, X_test_list, Y_test_list, model_path, name='NAS_Single')
        print(f"SVM Single Label Classification - Overall Accuracy: {svm_acc}")

if MODE == 'multi':  

    if TRAIN:  
        print("\nStarting extended training of best architecture...")

        best_accuracy = train_best_architecture_multi(
            architecture=best_architecture,
            X_train_list=X_train_list,
            Y_train_list=Y_train_list,
            lr_value=0.1,
            model_path=model_path,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            patience=15
        )
    model = FlexibleMultiLabelCNN(best_architecture, num_classes=9, pretrained=PRETRAINED)

    search_space = MultiArchitectureSearchSpace()
    nas_trainer = NASTrainer(search_space, None, model_path="", architecture_path="")

    if EVAL:
        svm_acc, class_acc = svm_classifier_multi(model, X_test_list, Y_test_list, test_batch_size=48, model_path=model_path, name='NAS_Multi')
        print(f"SVM Multi Label Classification - Overall Accuracy: {svm_acc}")
        print(f"Class-wise Accuracy: {class_acc}")