# Machine Learning Projects Collection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

## Overview

This repository is a curated collection of machine learning mini-projects designed to build and reinforce fundamental concepts in data science and machine learning. Each project serves as a hands-on learning experience, allowing practitioners to apply theoretical knowledge through practical implementation. The primary motivation behind this collection is to **learn and practice** various algorithms, preprocessing techniques, model architectures, and deployment strategies in a structured, progressive manner. From basic linear regression to advanced transformer models, these projects demonstrate the journey of mastering machine learning through consistent practice and experimentation.

## Table of Contents

- [Projects](#projects)
- [Tools and Libraries](#tools-and-libraries)
- [Installation](#installation)
- [Goals](#goals)
- [Contributing](#contributing)

## Projects

### 1. Linear Regression Basics
- **Objective**: Implement linear regression from scratch using NumPy and compare with scikit-learn's implementation.
- **Learning Focus**: Understand model training, prediction, and evaluation metrics (MSE, R²).
- **Dataset**: Salary vs Experience data from Kaggle.
- **Key Skills**: Mathematical foundations, gradient descent, model comparison.

### 2. Exploratory Data Analysis (EDA) on Advertising Dataset
- **Objective**: Perform comprehensive exploratory data analysis on marketing data.
- **Learning Focus**: Identify correlations, create visualizations, and derive insights.
- **Dataset**: Kaggle Advertising dataset.
- **Key Skills**: Data visualization, correlation analysis, statistical exploration.

### 3. Data Preprocessing & Feature Engineering on Titanic Dataset
- **Objective**: Clean and preprocess the Titanic dataset for machine learning.
- **Learning Focus**: Handle missing values, encode categorical variables, scale features, and engineer new features.
- **Dataset**: Kaggle Titanic dataset.
- **Key Skills**: Data cleaning, feature engineering, preprocessing pipelines.

### 4. Logistic Regression on Pre-Processed Titanic Data
- **Objective**: Implement logistic regression with cross-validation and hyperparameter tuning.
- **Learning Focus**: Model evaluation, feature importance analysis, learning curves, error analysis.
- **Dataset**: Kaggle Titanic dataset.
- **Key Skills**: Classification algorithms, model validation, hyperparameter optimization.

### 5. Decision Trees on Pre-Processed Titanic Data
- **Objective**: Build and tune decision tree classifiers.
- **Learning Focus**: Explore tree parameters, visualize decision structures, analyze feature importance.
- **Dataset**: Kaggle Titanic dataset.
- **Key Skills**: Tree-based algorithms, model interpretability, parameter tuning.

### 6. ML Pipelines with Logistic Regression on Titanic Data
- **Objective**: Construct full scikit-learn pipelines with preprocessing and modeling.
- **Learning Focus**: Integrate feature engineering, preprocessing, and hyperparameter tuning in pipelines.
- **Dataset**: Kaggle Titanic dataset.
- **Key Skills**: Pipeline construction, cross-validation, data leakage prevention.

### 7. ML Pipelines with Decision Trees on Housing Data
- **Objective**: Build decision tree regression pipelines for housing price prediction.
- **Learning Focus**: Handle missing values, encode categoricals, scale features, and tune hyperparameters.
- **Dataset**: Ames Housing Dataset.
- **Key Skills**: Regression pipelines, feature preprocessing, model evaluation.

### 8. Random Forest Classifier with Pipelines on Heart Disease Data
- **Objective**: Implement random forest classification with advanced preprocessing.
- **Learning Focus**: Ensemble methods, feature engineering, stratified cross-validation.
- **Dataset**: Heart Failure Prediction Dataset.
- **Key Skills**: Ensemble learning, feature selection, robust evaluation.

### 9. Gradient Boosting Classifier with Pipelines on Heart Disease Data
- **Objective**: Apply gradient boosting with comprehensive EDA and feature engineering.
- **Learning Focus**: Boosting algorithms, feature importance analysis, threshold-based feature selection.
- **Dataset**: Heart Failure Prediction Dataset.
- **Key Skills**: Gradient boosting, advanced EDA, feature engineering.

### 10. XGBoost Classifier with Pipelines on Heart Disease Data
- **Objective**: Utilize XGBoost for classification with hyperparameter optimization.
- **Learning Focus**: Extreme gradient boosting, cross-validation, feature importance evaluation.
- **Dataset**: Heart Failure Prediction Dataset.
- **Key Skills**: XGBoost, hyperparameter tuning, model interpretation.

### 11. LightGBM Classifier with Advanced Practices on Heart Disease Data
- **Objective**: Implement LightGBM with domain-specific feature engineering.
- **Learning Focus**: Light gradient boosting, advanced EDA, healthcare domain knowledge application.
- **Dataset**: Heart Failure Prediction Dataset.
- **Key Skills**: LightGBM, domain expertise, advanced preprocessing.

### 12. Model Stacking with Pipelines on Heart Disease Data
- **Objective**: Build stacking ensembles with base and meta models.
- **Learning Focus**: Ensemble stacking, advanced feature engineering, overfitting prevention.
- **Dataset**: Heart Failure Prediction Dataset.
- **Key Skills**: Model stacking, ensemble methods, feature engineering.

### 13. Model Voting with Pipelines on Fraud Detection Data
- **Objective**: Create voting classifiers for fraud detection with error analysis.
- **Learning Focus**: Ensemble voting, hyperparameter optimization, error analysis, feature engineering.
- **Dataset**: Credit Card Fraud Detection Dataset.
- **Key Skills**: Voting ensembles, fraud detection, model diagnostics.

### 14. FastAPI Credit Card Fraud Prediction Service
- **Objective**: Deploy a fraud detection model as a REST API using FastAPI.
- **Learning Focus**: Model deployment, API development, input validation, error handling.
- **Dataset**: Credit Card Fraud Detection Dataset.
- **Key Skills**: API development, model serialization, production deployment.

### 15. Neural Network from Scratch with NumPy
- **Objective**: Implement a complete neural network using only NumPy.
- **Learning Focus**: Forward/backward propagation, gradient descent, activation functions, weight initialization.
- **Dataset**: Scikit-learn make_circles (synthetic).
- **Key Skills**: Neural network fundamentals, optimization algorithms, manual implementation.

### 16. Neural Network with PyTorch on Titanic Dataset
- **Objective**: Build PyTorch neural networks integrated with sklearn pipelines.
- **Learning Focus**: Deep learning frameworks, reproducible training, mini-batch processing.
- **Dataset**: Kaggle Titanic dataset.
- **Key Skills**: PyTorch, neural network training, pipeline integration.

### 17. Multi-Layer Perceptron on MNIST Dataset
- **Objective**: Implement MLP for digit classification with hyperparameter tuning.
- **Learning Focus**: Neural network architecture, regularization techniques, cross-validation.
- **Dataset**: MNIST.
- **Key Skills**: Deep learning, hyperparameter optimization, image classification.

### 18. Convolutional Neural Network on MNIST Dataset
- **Objective**: Build CNNs for handwritten digit recognition.
- **Learning Focus**: Convolutional layers, data augmentation, mixed precision training.
- **Dataset**: MNIST.
- **Key Skills**: Convolutional networks, GPU training, advanced optimization.

### 19. CNN from Scratch on CIFAR-10 Dataset
- **Objective**: Implement CNN from scratch for image classification.
- **Learning Focus**: Custom CNN architecture, data preprocessing, hyperparameter tuning.
- **Dataset**: CIFAR-10.
- **Key Skills**: CNN implementation, image processing, model optimization.

### 20. Transfer Learning with ResNet18 on CIFAR-10
- **Objective**: Apply transfer learning with progressive unfreezing.
- **Learning Focus**: Pretrained models, fine-tuning strategies, data augmentation.
- **Dataset**: CIFAR-10.
- **Key Skills**: Transfer learning, model adaptation, advanced augmentation.

### 21. EfficientNet Transfer Learning on CIFAR-10
- **Objective**: Utilize EfficientNet for efficient image classification.
- **Learning Focus**: Modern architectures, hyperparameter optimization, backbone fine-tuning.
- **Dataset**: CIFAR-10.
- **Key Skills**: EfficientNet, transfer learning, performance optimization.

### 22. Sentiment Analysis Preprocessing
- **Objective**: Implement text preprocessing for sentiment analysis using classical and neural approaches.
- **Learning Focus**: Text cleaning, vectorization (BoW/TF-IDF), vocabulary building, embedding preparation.
- **Dataset**: IMDB movie reviews.
- **Key Skills**: NLP preprocessing, classical ML for text, neural text preparation.

### 23. BERT Sentiment Classification
- **Objective**: Fine-tune BERT for sentiment analysis using Hugging Face Transformers.
- **Learning Focus**: Transformer models, fine-tuning, optimization techniques (FP16, early stopping).
- **Dataset**: IMDB movie reviews.
- **Key Skills**: Transformers, BERT, advanced NLP, model optimization.

## Tools and Libraries

- **Programming Language**: Python 3.8+
- **Core Libraries**: NumPy, Pandas, Matplotlib, Seaborn
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Deep Learning**: PyTorch, Hugging Face Transformers
- **Data Handling**: KaggleHub
- **Web Framework**: FastAPI
- **NLP**: spaCy, NLTK

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Manouso/mini-projects.git
   cd mini-projects
   ```

2. **Install Dependencies**:
   Each project folder contains a `requirements.txt` file. Install dependencies for specific projects:
   ```bash
   pip install -r <project-folder>/requirements.txt
   ```

3. **Additional Setup**:
   - For NLP projects, download required models (e.g., `python -m spacy download en_core_web_sm`)
   - Ensure GPU support for deep learning projects if available

## Goals

- **Build Strong Foundations**: Develop a structured understanding of data analysis and machine learning concepts.
- **Practical Application**: Apply theoretical knowledge through focused, hands-on projects.
- **Professional Preprocessing**: Master dataset preparation and feature engineering techniques.
- **Portfolio Development**: Create a comprehensive showcase of ML skills and project diversity.
- **Continuous Learning**: Emphasize the importance of practice and experimentation in mastering machine learning.

## Contributing

Contributions are welcome! If you'd like to add new projects, improve existing ones, or suggest enhancements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Author

**Manousos Kirkinis** - [GitHub](https://github.com/Manouso) | [LinkedIn](https://www.linkedin.com/in/manousos-kirkinis)

---

*This collection represents a journey of learning and practice in machine learning. Each project is designed to teach specific concepts while building towards more complex applications.*


