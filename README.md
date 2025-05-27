# CVFL
# Improved Coresets for Vertical Federated Learning: Regularized Linear and Logistic Regressions

This repository contains the implementation and experimental evaluation of various coreset-based sampling methods for Ridge Regression and Logistic Regression, comparing their performance against uniform sampling baselines.

## Overview

Coresets are small weighted subsets of large datasets that approximately preserve the cost function of machine learning problems. This work implements and evaluates several coreset construction algorithms for both regression and classification tasks.

## Implemented Methods

### Ridge Regression Methods
- **Uniform Sampling**: Baseline uniform random sampling
- **HLSZ Coreset**: Leverage score-based coreset construction
- **Regularized Leverage Scores (Lev)**: Improved leverage scores accounting for regularization

### Logistic Regression Methods
- **Uniform Sampling**: Baseline uniform random sampling  
- **HLSZ Coreset**: Leverage score-based coreset for logistic regression
- **Squared Leverage Scores (SqLev)**: Regularized leverage score sampling
- **Lewis Weights**: Lewis weight-based coreset construction
- **Augmented Lewis (AugLewis)**: Enhanced Lewis weights with regularization

## Requirements

```python
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
imbalanced-learn>=0.8.0
```

## Installation

```bash
pip install numpy pandas scikit-learn matplotlib imbalanced-learn
```

## Usage

### Ridge Regression Experiments

```python
python ridge_regression_experiment.py
```

This script:
1. Loads and preprocesses financial data
2. Runs experiments across different sample sizes
3. Compares uniform sampling vs. coreset methods
4. Generates performance plots and metrics

### Logistic Regression Experiments

```python
python logistic_regression_experiment.py
```

This script:
1. Loads credit card fraud detection data
2. Handles class imbalance using SMOTE
3. Evaluates different coreset construction methods
4. Compares accuracy, loss, and coefficient differences

## Experimental Setup

### Datasets

**Financial Data (Ridge Regression)**
- Features: Market indicators, currency rates, economic metrics
- Target: S&P 500 closing price
- Task: Regression with L2 regularization

**Credit Card Fraud (Logistic Regression)**  
- Features: 30 anonymized transaction features
- Target: Fraud detection (binary classification)
- Preprocessing: SMOTE for class balancing

### Evaluation Metrics

**Ridge Regression**
- Root Mean Square Error (RMSE) on train/test sets
- R² coefficient of determination  
- Relative coefficient difference from full model
- Training time comparison

**Logistic Regression**
- Balanced accuracy on train/test sets
- Log loss (cross-entropy)
- Relative coefficient difference from full model
- Training time comparison

### Sample Sizes

- **Ridge Regression**: [10, 25, 50, 75, 100, 200, 400, 500, 600, 800, 1000, 1500]
- **Logistic Regression**: [50, 150, 300, 500, 700, 900, 1200, 1500, 2000, 2500]

## Key Algorithms

### Leverage Score Computation

Leverage scores measure the importance of each data point for the optimization problem:

```
s_i = ||x_i||²_{(X^T X + λI)^{-1}}
```

### Coreset Construction

1. **Compute importance scores** (leverage scores, Lewis weights)
2. **Sample points** proportional to importance scores  
3. **Assign weights** to maintain unbiased estimation
4. **Train model** on weighted coreset

### Lewis Weight Iteration

For p-th power regularization:
```
w^{(t+1)}_i = (β · leverage_score_i)^{p/2}
```

## Results

The experiments demonstrate:

1. **Coreset methods consistently outperform uniform sampling** in terms of model quality
2. **Regularized methods (Lev, SqLev) show superior performance** on high-dimensional data
3. **Lewis weights provide excellent approximation** for logistic regression
4. **Training time benefits** scale with dataset size reduction
