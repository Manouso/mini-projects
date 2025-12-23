# Convolutional Neural Network for MNIST Digit Classification

A PyTorch implementation of a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset. This project demonstrates modern deep learning practices including data preprocessing, hyperparameter tuning, GPU acceleration, and model evaluation.

## Features

- **Modern CNN Architecture**: Lightweight CNN with batch normalization and dropout for stable training
- **Data Augmentation**: Random rotation and translation for improved generalization
- **GPU Acceleration**: Automatic GPU detection and utilization with CUDA support
- **Mixed Precision Training**: Uses PyTorch AMP for ~2x faster training on modern GPUs
- **Hyperparameter Tuning**: Random search over learning rate, batch size, dropout, and weight decay
- **Early Stopping**: Prevents overfitting with patience-based early stopping
- **Comprehensive Evaluation**: Classification reports, confusion matrices, and prediction examples

## Model Architecture

The CNN consists of:
- **Input**: 28x28 grayscale images
- **Conv Block 1**: 32 filters (3x3), BatchNorm, ReLU → 32 filters (3x3), BatchNorm, ReLU → MaxPool (2x2)
- **Conv Block 2**: 64 filters (3x3), BatchNorm, ReLU → 64 filters (3x3), BatchNorm, ReLU → Global Average Pooling
- **Classifier**: 128 neurons, BatchNorm, Dropout → 10 output classes
- **Regularization**: Dropout (0.4-0.5), L2 weight decay
- **Total Parameters**: ~75K

## Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA support for GPU acceleration)
- Torchvision 0.15+
- NumPy 1.21+
- Matplotlib 3.5+
- Scikit-learn 1.0+
- Seaborn 0.11+

## Installation

1. Clone or download the project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training Details

- **Dataset**: MNIST (60K train, 10K test)
- **Split**: 80/20 train/validation
- **Augmentation**: Random rotation (±10°) and translation (±10%)
- **Optimization**: Adam optimizer with ReduceLROnPlateau scheduling and mixed precision (FP16)
- **Tuning**: 3 trials × 5 epochs each (random search)
- **Final Training**: 20 epochs with early stopping (patience=5)
- **Hardware**: GPU preferred (automatic fallback to CPU)

## Results

- **Test Accuracy**: over 99% with validation accuracy larger than training indicating that the model truly learned well.


## Project Structure

```
convolutuonal-neural-network/
├── convolutional_neural_network.ipynb  # Main notebook
├── requirements.txt                    # Python dependencies
├── README.md                           # This file

```

## Key Hyperparameters

The model uses random search to optimize:
- Learning rate: [0.001, 0.01]
- Batch size: [32, 64]
- Dropout rate: [0.4, 0.5]
- Weight decay: 0.0001