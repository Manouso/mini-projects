# Neural Network with PyTorch on Titanic Dataset

A complete implementation of a neural network using PyTorch for binary classification on the Titanic survival dataset. This project demonstrates advanced preprocessing with sklearn pipelines, feature engineering, hyperparameter tuning, and PyTorch model training with sklearn compatibility.

## Features

- **PyTorch Neural Network**: Custom `nn.Module` class with configurable layers
- **Advanced Preprocessing**: sklearn pipelines with `FunctionTransformer` for missing values and feature engineering
- **Feature Engineering**: Family size, alone status, title extraction, age/fare binning
- **Data Pipeline**: Complete sklearn pipeline integrating pandas operations with PyTorch
- **Hyperparameter Tuning**: RandomizedSearchCV with stratified k-fold cross-validation
- **Multiple Loss Functions**: BCEWithLogitsLoss and BCELoss with proper sigmoid handling
- **Multiple Optimizers**: Adam, SGD, RMSprop with configurable parameters
- **sklearn Compatibility**: Custom estimator class for seamless sklearn integration
- **Reproducible Training**: Standardized random states across NumPy, PyTorch, and sklearn
- **Batch Training**: Mini-batch gradient descent with DataLoader
- **Evaluation**: Accuracy, classification report, and loss visualization

## Architecture

- **Input Layer**: Variable size based on engineered features (22 features after preprocessing)
- **Hidden Layer**: Configurable neurons (8-64) with ReLU activation
- **Output Layer**: Single neuron outputting raw logits for BCEWithLogitsLoss

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/mini-projects.git
cd mini-projects/neural-network-pytorch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook neural_network_pytorch.ipynb
```

2. Run the cells in order to:
   - Import libraries and set random seeds
   - Load and preprocess Titanic dataset
   - Define feature engineering functions
   - Build sklearn pipeline with FunctionTransformers
   - Perform hyperparameter tuning with RandomizedSearchCV
   - Train final model with best hyperparameters
   - Evaluate and visualize results

## Key Components

## Key Components

### Preprocessing Pipeline
- **Missing Values**: Median imputation for age, mode for embarked
- **Feature Engineering**: Creates FamilySize, IsAlone, Title, AgeBin, FareBin
- **Encoding**: One-hot encoding for categoricals, standardization for numerics
- **Integration**: Uses `FunctionTransformer` with wrappers to handle DataFrame operations in sklearn pipeline

### Neural Network Architecture
- **Forward Pass**: Linear → ReLU → Linear (outputs raw logits)
- **Loss Functions**: BCEWithLogitsLoss (preferred) and BCELoss with manual sigmoid
- **Optimizers**: Adam, SGD, RMSprop with configurable learning rates
- **Training**: Mini-batch with DataLoader, loss tracking, reproducible seeds

### Hyperparameter Tuning
- **Method**: RandomizedSearchCV with 20 iterations
- **Cross-Validation**: Stratified 5-fold CV for balanced class distribution
- **Parameters Tuned**:
  - Hidden layer size (8-64 neurons)
  - Learning rate (0.0001-0.01)
  - Epochs (50-200)
  - Optimizer type (adam, sgd, rmsprop)
  - Loss function type (bce_logits, bce)
- **Scoring**: Accuracy with stratified sampling

### sklearn Compatibility
- **PyTorchEstimator Class**: Custom BaseEstimator implementing fit/predict/predict_proba
- **Seamless Integration**: Works with sklearn's hyperparameter tuning tools
- **Proper Sigmoid Handling**: Different logic for BCEWithLogitsLoss vs BCELoss

### Data Flow
1. Raw DataFrame → sklearn Pipeline (preprocessing) → NumPy array
2. Array → Hyperparameter Tuning (RandomizedSearchCV) → Best Parameters
3. Best Params → Final Model Training → PyTorch tensors → DataLoader → Model training
4. Predictions → Evaluation metrics and classification report

## Results

### Hyperparameter Tuning Performance
- **Best Parameters Found**:
  - Hidden Size: 52 neurons
  - Learning Rate: 0.000846
  - Epochs: 64
  - Optimizer: RMSprop
  - Loss Function: BCEWithLogitsLoss
- **Cross-Validation Score**: 83.98% accuracy
- **Test Set Performance**: 81.56% accuracy

### Classification Report
```
              precision    recall  f1-score   support

           0       0.83      0.87      0.85       105
           1       0.80      0.74      0.77        74

    accuracy                           0.82       179
   macro avg       0.81      0.80      0.81       179
weighted avg       0.81      0.82      0.81       179
```

## Model Evaluation and Limitations

**Current Performance**: 81.56% test accuracy with optimized hyperparameters.

**Strengths**:
- Proper hyperparameter optimization with cross-validation
- Multiple loss functions and optimizers tested
- Reproducible results with fixed random seeds
- sklearn-compatible for easy integration

**Limitations**:
- Simple 2-layer architecture may underfit complex patterns
- No regularization (dropout, batch normalization) implemented
- Limited hyperparameter search space
- No early stopping or learning rate scheduling

## Future Enhancements

### Architecture Improvements
- Add dropout layers for regularization (20-30% dropout rate)
- Implement batch normalization for stable training
- Add more hidden layers (3-4 layers deep)
- Try different activation functions (LeakyReLU, ELU)

### Training Enhancements
- Implement early stopping to prevent overfitting
- Add learning rate scheduling (ReduceLROnPlateau)
- Use class weights for imbalanced classification
- Implement gradient clipping

### Advanced Techniques
- Ensemble methods (train multiple models, average predictions)
- Cross-validation with multiple metrics (F1, ROC-AUC)
- Feature selection and engineering improvements
- Model checkpointing and best model saving

## Learning Objectives

This implementation teaches:
- Integrating sklearn pipelines with PyTorch workflows
- Handling DataFrame operations in ML pipelines
- Proper random state management for reproducibility
- Hyperparameter tuning with RandomizedSearchCV
- sklearn-compatible PyTorch estimators
- Cross-validation strategies for imbalanced data
- Multiple loss functions and their proper usage
- Mini-batch training with PyTorch DataLoader
- Binary classification with BCE loss variants

## Comparison with NumPy Implementation

This PyTorch version offers:
- GPU acceleration potential
- Automatic differentiation
- Better scalability for larger datasets
- More flexible model architectures
- Integration with PyTorch ecosystem
- Hyperparameter tuning compatibility
- Advanced optimization algorithms

While the NumPy version provides deeper understanding of low-level operations, this PyTorch version is more practical for production use and advanced experimentation.
