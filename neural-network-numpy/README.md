# Neural Network from Scratch with NumPy

A complete implementation of a 2-layer neural network built entirely from scratch using NumPy. This project demonstrates the fundamentals of neural networks including forward propagation, backpropagation, and gradient descent optimization.

## Features

- **Custom Neural Network Class**: `TwoLayerNN` with configurable input, hidden, and output sizes
- **Activation Functions**: ReLU (hidden layer) and Sigmoid (output layer)
- **Proper Initialization**: He initialization for weights to ensure better convergence
- **Backpropagation**: Full gradient computation and parameter updates
- **Training Loop**: Configurable epochs with loss tracking
- **Prediction Methods**: Both binary predictions and probability outputs
- **Reproducibility**: Random state support for consistent results
- **Visualization**: Decision boundary plots and loss curves

## Architecture

- **Input Layer**: Accepts 2D input features
- **Hidden Layer**: Configurable number of neurons with ReLU activation
- **Output Layer**: Single neuron with Sigmoid activation for binary classification

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn (for sample dataset generation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/mini-projects.git
cd mini-projects/neural-network-numpy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook neural_network_numpy.ipynb
```

2. Run the cells in order to:
   - Import required libraries
   - Define the neural network class
   - Generate sample data (make_circles)
   - Train the model
   - Visualize results


## Key Components

### Forward Pass
- Linear transformation: `Z = X @ W + b`
- Activation: ReLU for hidden, Sigmoid for output

### Backward Pass
- Output layer gradients using chain rule
- Hidden layer gradients with ReLU derivative
- Parameter updates with gradient descent

### Training
- Binary cross-entropy loss
- Mini-batch gradient descent (batch size = full dataset)
- Loss monitoring every 100 epochs

## Results

The model achieves high accuracy on the make_circles dataset, demonstrating the effectiveness of the implementation. The decision boundary visualization shows how the neural network learns to separate the concentric circles.

### Suggestions for Improved Generalization

To minimize loss while maintaining accuracy in future projects:

1. **Train/Validation/Test Split**: Always split your data into training, validation, and test sets. Use validation loss to monitor overfitting and implement early stopping.

2. **Regularization Techniques**:
   - **L2 Regularization**: Add weight decay to the loss function to prevent large weights
   - **Dropout**: Randomly drop neurons during training to prevent co-adaptation
   - **Early Stopping**: Stop training when validation loss starts increasing

3. **Better Optimization**:
   - **Adam Optimizer**: Replace basic gradient descent with Adam for faster convergence
   - **Learning Rate Scheduling**: Use learning rate decay or cyclical learning rates
   - **Batch Training**: Implement mini-batch gradient descent instead of full-batch

4. **Data Augmentation**: Generate more diverse training samples to improve generalization

5. **Cross-Validation**: Use k-fold cross-validation to get more reliable performance estimates

6. **Model Architecture**: Experiment with different hidden layer sizes, add more layers, or try different activation functions

## Learning Objectives

This implementation helps understand:
- How neural networks process data through layers
- The mathematics behind backpropagation
- Importance of proper weight initialization
- Role of activation functions in non-linearity
- Gradient descent optimization
