# MNIST MLP Classification Project

This project implements a complete machine learning pipeline for MNIST digit classification using PyTorch Multi-Layer Perceptron (MLP) with hyperparameter tuning and comprehensive evaluation.

## Overview

The notebook demonstrates:
- Data loading and exploratory data analysis (EDA) of MNIST dataset
- MLP model definition with configurable hidden layers
- Hyperparameter tuning using RandomizedSearchCV with Stratified K-Fold cross-validation
- Training with validation monitoring and loss/accuracy curves
- Detailed evaluation metrics including accuracy, precision, recall, F1-score, and confusion matrix
- Sample predictions visualization

## Dataset

- **Training samples**: 60,000 (28x28 grayscale images)
- **Test samples**: 10,000 (28x28 grayscale images)
- **Classes**: 10 (digits 0-9)
- **Preprocessing**: Normalization to [-1, 1] range

## Model Architecture

MLP with:
- Input layer: 784 neurons (28x28 flattened)
- Hidden layers: Configurable (e.g., [128] or [128, 64])
- Output layer: 10 neurons (logits for 10 classes)
- Activation: ReLU for hidden layers
- Loss: CrossEntropyLoss

## Hyperparameter Tuning Results

Using RandomizedSearchCV with 5 iterations and 2-fold stratified cross-validation on 5,000 training samples:

- **Best Parameters**:
  - Optimizer: Adam
  - Learning Rate: 0.01
  - Hidden Sizes: [128]
  - Epochs: 3
  - Batch Size: 64

- **Best Cross-Validation Score**: 0.8504 (85.04% accuracy)

## Performance Metrics

### Cross-Validation Performance
- Accuracy: 85.04%
- Trained on subset of 5,000 samples for efficiency

### Final Performance

- Test accuracy : 90-92% (91.77%)
- Training includes validation monitoring with loss/accuracy curves

## Key Features

- **Optimized for CPU**: Reduced dataset sizes and parameters for practical execution on consumer hardware
- **Comprehensive Evaluation**: Classification report, confusion matrix, and sample predictions
- **Visualization**: Training curves, class distributions, sample images
- **Modular Design**: Sklearn-compatible wrapper for easy hyperparameter tuning

## Dependencies

- torch
- torchvision
- scikit-learn
- matplotlib
- seaborn
- numpy

## Results Summary

The optimized MLP achieves strong performance on MNIST with efficient training. The hyperparameter tuning identifies optimal settings for architecture and training parameters, resulting in reliable digit classification with high accuracy.
