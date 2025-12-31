# CNN from Scratch with PyTorch

This project implements a Convolutional Neural Network (CNN) using PyTorch for classifying the CIFAR-10 dataset. The CNN uses PyTorch's nn.Module for layers, with data augmentation and hyperparameter tuning.

## Features

- CNN model using PyTorch nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.Linear, nn.Dropout
- Data preprocessing: normalization and data augmentation (random crop, horizontal flip, random erasing, Gaussian noise)
- Hyperparameter search over learning rate, batch size, dropout rate, weight decay
- Training with Adam optimizer
- Evaluation on test set with confusion matrix and classification report
- GPU support if available

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Jupyter notebook `cnn_from_scratch.ipynb`.
2. The notebook will download CIFAR-10 automatically if not present.
3. It performs hyperparameter search on a subset of epochs to find the best combo.
4. Then trains the model with the best hyperparameters for 10 epochs.
5. Evaluates on the test set and plots results.

## Data Preprocessing

- **Normalization**: Images normalized using CIFAR-10 mean and std per channel.
- **Data Augmentation** (training only): Random crop (with padding), random horizontal flip, random erasing, Gaussian noise.

## Model Architecture

- Conv2D (3 -> filters1, 3x3, padding=1)
- BatchNorm2D
- ReLU
- MaxPool2D (2x2)
- Conv2D (filters1 -> filters2, 3x3, padding=1)
- BatchNorm2D
- ReLU
- MaxPool2D (2x2)
- Flatten
- Linear (filters2*8*8 -> 10)
- Dropout

Where filters1 and filters2 are configurable (default 32, 64).

## Hyperparameters

Tuned via 5-fold cross-validation grid search:

- Learning rate: [0.0001, 0.001, 0.01]
- Batch size: [64, 128, 256]
- Dropout rate: [0.3, 0.4, 0.5]
- Weight decay: [0, 1e-4, 1e-3]
- Optimizer: Adam

Training uses LR scheduler (StepLR), early stopping, mixed precision for GPU optimization, and label smoothing.

## Estimated Performance

With cross-validation tuning, LR scheduling, and optimizations, expect 75-85% test accuracy after 20-30 epochs.

## Notes

- Uses PyTorch for efficient computation and autograd.
- Supports GPU if available.
- Hyperparameter search trains each combo for 2 epochs to save time.
