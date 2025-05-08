# FLAME_NAS

This repository was made as a part of our Computer Vision Project.

## A) Description

**FLAME_NAS** is a Neural Architecture Search (NAS) framework designed for automated architecture discovery in fire detection tasks using deep learning. The project leverages a controller-based NAS approach to search for optimal convolutional neural network architectures tailored for fire detection in images or video frames. It includes utilities for training, evaluation, and architecture search, making it easy to experiment with different datasets and configurations.

## B) Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/05kashyap/FLAME_NAS.git
   cd FLAME_NAS
   ```

2. **Install dependencies:**
   Make sure you have Python 3.7+ and pip installed. Then run:
   ```bash
   pip install torch torchvision numpy
   ```
3. **Prepare your dataset:**
   - Place your dataset in the appropriate directory (e.g., `Data/archive/`).
   - Ensure you have a labels file (e.g., `Frame_Pair_Labels.txt`).

4. **Run NAS training:**
   ```bash
   python NAS_trainer.py --labels-file <path_to_labels_file> --base-dir <dataset_base_dir> --batch-size 32 --pecs 3 --iters 20 --subset-fraction 0.2 --save_path <output_dir>
   ```

5. **Evaluate the best architecture:**
   ```bash
   python NAS_eval.py --labels-file <path_to_labels_file> --base-dir <dataset_base_dir> --batch-size 32 --epochs 1 --arch-path <path_to_best_architecture_json> --save_path <output_dir>
   ```

## C) Directory Structure

```
FLAME_NAS/
├── NAS_trainer.py           # Main script for NAS-based training
├── NAS_eval.py              # Script for evaluating the best architecture
├── dataloading.py           # Data loading utilities
├── NAS_utils/
│   └── multi/
│       ├── controller.py    # NAS controller implementation
│       ├── searchspace.py   # Search space definition
│       └── trainer.py       # NAS training logic
├── model_saving/            # Directory for saving trained models
├── Data/
│   └── archive/             # Place your dataset here
│       └── Frame_Pair_Labels.txt
├── flame-nas.ipynb          # Example Jupyter notebook
├── LICENSE
└── README.md
```

- **NAS_trainer.py**: Runs the NAS process to search for the best architecture.
- **NAS_eval.py**: Evaluates the selected/best architecture on the dataset.
- **NAS_utils/multi/**: Contains core NAS logic (controller, search space, trainer).
- **dataloading.py**: Handles dataset loading and preprocessing.
- **model_saving/**: Stores trained model checkpoints.
- **Data/archive/**: Expected location for your dataset and labels.
- **flame-nas.ipynb**: Example notebook for interactive usage.
- **patchand-cycle-gan.ipynb**: Example notebook for training the GAN model for minority class sample generation.

---
