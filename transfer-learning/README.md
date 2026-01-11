# Transfer Learning with ResNet18 on CIFAR-10

This project demonstrates advanced transfer learning using a pretrained ResNet18 model on the CIFAR-10 dataset with PyTorch. The implementation includes extensive data augmentation, progressive unfreezing, and hyperparameter tuning to achieve optimal performance.

## Features

- Load and preprocess CIFAR-10 dataset with extensive data augmentation (random cropping, flipping, rotation, color jittering, affine transformations, perspective distortion, and random erasing)
- Use pretrained ResNet18 model from torchvision
- Replace the classifier head with a custom multi-layer network
- Implement progressive unfreezing: freeze pretrained layers initially, then unfreeze later layers for fine-tuning
- Hyperparameter optimization using random search over learning rates, batch sizes, and unfreezing strategies
- Evaluate performance on the validation set with detailed metrics

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure you have the CIFAR-10 dataset downloaded (the notebook will download it automatically if not present).
2. Run the Jupyter notebook `transfer_learning.ipynb` to execute the transfer learning pipeline.
3. The notebook includes:
   - Data loading and preprocessing
   - Model architecture definition
   - Training with progressive unfreezing
   - Hyperparameter tuning
   - Model evaluation and saving

## Results

After hyperparameter optimization, the model achieves a best validation accuracy of **58.56%** on the CIFAR-10 test set. The training process includes multiple trials with different configurations to find the optimal setup.

## Model Architecture

- **Base Model**: Pretrained ResNet18
- **Modified Classifier**: Custom head with dropout and batch normalization
- **Input Size**: 224x224 (resized from 32x32 with augmentation)
- **Output Classes**: 10 (CIFAR-10 categories)

## Training Strategy

1. **Phase 1**: Train only the classifier head with frozen ResNet layers
2. **Phase 2**: Unfreeze and fine-tune the last few ResNet layers
3. **Phase 3**: Hyperparameter optimization with random search

The saved model weights are available in `resnet18_cifar10_transfer_learning.pth`.
