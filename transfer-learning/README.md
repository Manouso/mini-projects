# Transfer Learning with ResNet18 on CIFAR-10

This project demonstrates advanced transfer learning using a pretrained ResNet18 model on the CIFAR-10 dataset with PyTorch.

## Features

- Load and preprocess CIFAR-10 dataset with appropriate transforms
- Use pretrained ResNet18 model from torchvision
- Replace the classifier head with a custom multi-layer network
- Freeze pretrained layers and train only the classifier initially
- Fine-tune the model by unfreezing later layers
- Evaluate performance on the test set

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

Run the Jupyter notebook `transfer_learning.ipynb` to execute the transfer learning pipeline.

## Results

The model achieves high accuracy on CIFAR-10 through transfer learning techniques.
