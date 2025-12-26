# CNN from Scratch

This project implements a Convolutional Neural Network (CNN) from scratch using NumPy for classifying the CIFAR-10 dataset. The CNN includes convolutional layers, ReLU activation, max pooling, fully connected layers, and dropout.

## Features

- Custom implementation of CNN layers (Conv2D, ReLU, MaxPool2D, FC, Dropout)
- Manual forward and backward propagation
- Training with gradient descent and weight decay
- Data preprocessing: normalization (mean/std) and data augmentation (random crop, horizontal flip, random erasing, Gaussian noise)
- Evaluation on test set with confusion matrix and classification report

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure you have the CIFAR-10 dataset downloaded (the code handles automatic download).
2. Run the Jupyter notebook `cnn_from_scratch.ipynb` to train and evaluate the model.

## Data Preprocessing

- **Normalization**: Images are normalized using CIFAR-10 mean and std per channel.
- **Data Augmentation** (training only): Random crop (with padding), random horizontal flip, random erasing (occlusion simulation), and Gaussian noise to increase dataset diversity and reduce overfitting.

## Model Architecture

- Conv2D (3->32 filters, 3x3, padding=1)
- ReLU
- MaxPool2D (2x2)
- Conv2D (32->64 filters, 3x3, padding=1)
- ReLU
- MaxPool2D (2x2)
- FC (64*8*8 -> 10)
- Dropout (0.5)

## Hyperparameters

- Learning rate: 0.01
- Batch size: 128
- Dropout rate: 0.5
- Weight decay: 1e-4
- Epochs: 5

## Estimated Performance

In my opinion, with this basic architecture, normalization, data augmentation, and limited epochs, the model is expected to achieve a test accuracy of around 60-70% on CIFAR-10. Data augmentation helps reduce overfitting and improve generalization, potentially boosting performance compared to no augmentation.

## Notes

- The model runs on CPU using NumPy and on GPU using CuPy.
- Training may take time due to the manual loops in the layers.
- The above reason is why i estimated time since it would take many hours till the training and testing of the model.
- For better performance we could use PyTorch or TensorFlow (but that wasnt my concern here).
