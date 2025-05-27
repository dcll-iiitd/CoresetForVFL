"""
Coreset Methods for Ridge Regression: A Comparative Study

This module implements and compares different coreset sampling methods for Ridge regression,
including uniform sampling and various coreset construction algorithms based on leverage scores
and regularized leverage scores.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os


def uniform_ridge(m, D, y):
    """
    Uniform random sampling baseline method for Ridge regression.
    
    Args:
        m (int): Number of samples to select
        D (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        
    Returns:
        tuple: (sampled_data, sampled_targets)
    """
    D = np.hstack((D, y.reshape(-1, 1)))
    D_df = pd.DataFrame(D)
    
    # Random sampling
    C = D_df.sample(n=m, replace=False)
    C = C.to_numpy()
    
    data = C[:, :-1]
    target = C[:, -1]
    return data, target


def coreset_ridge_hlsz(m, D, y, alpha):
    """
    HLSZ (Huggins et al.) coreset method based on QR decomposition for Ridge regression.
    
    This method constructs importance sampling probabilities using QR decomposition
    on different parts of the feature matrix.
    
    Args:
        m (int): Number of samples to select
        D (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        alpha (float): Regularization parameter
        
    Returns:
        tuple: (sampled_data, sampled_targets, sample_weights)
    """
    num_of_data, _ = D.shape
    D = np.hstack((D, y.reshape(-1, 1)))
    
    # Split features into three parts for QR decomposition
    D1 = D[:, :15]
    D2 = D[:, 15:30]
    D3 = D[:, 30:]
    
    # Compute QR decomposition for each part
    q1, _ = np.linalg.qr(D1)
    q2, _ = np.linalg.qr(D2)
    q3, _ = np.linalg.qr(D3)
    
    Q = np.hstack((q1, q2, q3))
    
    # Compute importance sampling probabilities
    s = (np.sum(Q ** 2, axis=1)) + 1 / num_of_data
    
    D = np.hstack((D, (1/s).reshape(-1, 1)))
    D_df = pd.DataFrame(D)
    
    # Sample according to importance weights
    C = D_df.sample(n=m, replace=False, weights=s)
    C = C.to_numpy()
    
    data = C[:, :-2]
    target = C[:, -2]
    weight = C[:, -1]
    weight = weight / np.sum(weight) * m * 10
    
    return data, target, weight


def coreset_ridge_regularized(m, D, y, alpha):
    """
    Regularized leverage score coreset method for Ridge regression.
    
    This method incorporates regularization into the leverage score computation
    for more stable coreset construction.
    
    Args:
        m (int): Number of samples to select
        D (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        alpha (float): Regularization parameter
        
    Returns:
        tuple: (sampled_data, sampled_targets, sample_weights)
    """
    num_of_data, _ = D.shape
    
    # Add regularization term
    sqrt_lambda_I_d = np.sqrt(alpha) * np.eye(D.shape[1])
    D_hat = np.vstack((D, sqrt_lambda_I_d))
    y_hat = np.hstack((y, np.zeros(D.shape[1])))
    D_hat = np.hstack((D_hat, y_hat.reshape(-1, 1)))
    D = np.hstack((D, y.reshape(-1, 1)))
    
    # Split augmented matrix for QR decomposition
    D1 = D_hat[:, :15]
    D2 = D_hat[:, 15:30]
    D3 = D_hat[:, 30:]
    
    # Compute QR decomposition
    q1, _ = np.linalg.qr(D1)
    q1 = q1[:num_of_data, :]
    q2, _ = np.linalg.qr(D2)
    q2 = q2[:num_of_data, :]
    q3, _ = np.linalg.qr(D3)
    q3 = q3[:num_of_data, :]
    
    # Compute importance weights
    q1 = np.sqrt(np.sum(q1 ** 2, axis=1)) + 1/D1.shape[0]
    q2 = np.sqrt(np.sum(q2 ** 2, axis=1)) + 1/D2.shape[0]
    q3 = np.sqrt(np.sum(q3 ** 2, axis=1)) + 1/D3.shape[0]
    Q = q1 + q2 + q3
    
    D = np.hstack((D, (1/(Q/(np.sum(Q)))).reshape(-1, 1)))
    D_df = pd.DataFrame(D)
    
    # Sample with importance weights
    C = D_df.sample(n=m, replace=False, weights=Q / np.sum(Q))
    C = C.to_numpy()
    
    data = C[:, :-2]
    target = C[:, -2]
    weight = C[:, -1]
    weight = weight / m
    
    return data, target, weight


def run_experiment_with_comparison(X_train, y_train, X_test, y_test, 
                                 sample_sizes=[10, 25, 50, 75, 100, 200, 400, 500, 600, 800, 1000, 1500], 
                                 num_of_rep=3, ridge_alpha=1000):
    """
    Run comprehensive comparison experiments between different sampling methods for Ridge regression.
    
    Args:
        X_train (np.ndarray): Training feature matrix
        y_train (np.ndarray): Training targets
        X_test (np.ndarray): Test feature matrix
        y_test (np.ndarray): Test targets
        sample_sizes (list): List of sample sizes to evaluate
        num_of_rep (int): Number of repetitions per experiment
        ridge_alpha (float): Regularization parameter for Ridge regression
        
    Returns:
        tuple: (experiment_results, full_model)
    """
    methods = {
        'Uniform': uniform_ridge,
        'HLSZ': coreset_ridge_hlsz,
        'RegularizedLeverage': coreset_ridge_regularized,
    }
    
    # Train full model for comparison baseline
    print("Training full model for baseline comparison...")
    full_model = Ridge(alpha=ridge_alpha)
    start_full_time = time.time()
    full_model.fit(X_train, y_train)
    full_train_time = time.time() - start_full_time
    
    # Get full model predictions for baseline
    full_train_pred = full_model.predict(X_train)
    full_test_pred = full_model.predict(X_test)
    full_train_rmse = np.sqrt(mean_squared_error(y_train, full_train_pred))
    full_test_rmse = np.sqrt(mean_squared_error(y_test, full_test_pred))
    full_train_r2 = r2_score(y_train, full_train_pred)
    full_test_r2 = r2_score(y_test, full_test_pred)
    
    print(f"Full model performance - Train RMSE: {full_train_rmse:.4f}, Test RMSE: {full_test_rmse:.4f}")
    print(f"Full model performance - Train R²: {full_train_r2:.4f}, Test R²: {full_test_r2:.4f}")
    
    # Initialize results storage
    all_results = {size: {method: {
        'train_rmse': [],
        'test_rmse': [],
        'train_r2': [],
        'test_r2': [],
        'coef_diff': [],
        'train_time': []
    } for method in methods.keys()} for size in sample_sizes}
    
    # Run experiments for each sample size
    for size in sample_sizes:
        print(f"\n{'='*50}")
        print(f"Sample Size: {size}")
        print(f"{'='*50}")
        
        model = Ridge(alpha=ridge_alpha)
        
        for method_name, sampling_method in methods.items():
            print(f"Running {method_name} method...")
            
            for rep in range(num_of_rep):
                try:
                    # Sample data using the current method
                    if method_name == 'Uniform':
                        sample_X, sample_y = sampling_method(size, X_train, y_train)
                        weights = None
                    else:
                        sample_X, sample_y, weights = sampling_method(size, X_train, y_train, ridge_alpha)
                    
                    # Measure training time
                    start_time = time.time()
                    if weights is None:
                        model.fit(sample_X, sample_y)
                    else:
                        model.fit(sample_X, sample_y, sample_weight=weights)
                    elapsed_time = time.time() - start_time
                    
                    # Make predictions
                    pred_train = model.predict(X_train)
                    pred_test = model.predict(X_test)
                    
                    # Calculate evaluation metrics
                    train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
                    test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
                    train_r2 = r2_score(y_train, pred_train)
                    test_r2 = r2_score(y_test, pred_test)
                    
                    # Calculate relative coefficient difference from full model
                    coef_diff = (np.linalg.norm(model.coef_.flatten() - full_model.coef_.flatten()) / 
                               np.linalg.norm(full_model.coef_.flatten()))
                    
                    # Store results
                    all_results[size][method_name]['train_rmse'].append(train_rmse)
                    all_results[size][method_name]['test_rmse'].append(test_rmse)
                    all_results[size][method_name]['train_r2'].append(train_r2)
                    all_results[size][method_name]['test_r2'].append(test_r2)
                    all_results[size][method_name]['coef_diff'].append(coef_diff)
                    all_results[size][method_name]['train_time'].append(full_train_time / elapsed_time)
                    
                except Exception as e:
                    print(f"Error in {method_name} method, repetition {rep}: {e}")
                    continue
        
        # Print summary for current sample size
        print(f"\nSummary for Sample Size {size}:")
        print(f"{'Method':<18} {'Train RMSE':>12} {'Test RMSE':>12} {'Train R²':>10} {'Test R²':>10} {'Coef Diff':>11} {'Speedup':>10}")
        print('-' * 95)
        
        for method in methods.keys():
            if all_results[size][method]['train_rmse']:  # Check if we have results
                avg_train_rmse = np.mean(all_results[size][method]['train_rmse'])
                avg_test_rmse = np.mean(all_results[size][method]['test_rmse'])
                avg_train_r2 = np.mean(all_results[size][method]['train_r2'])
                avg_test_r2 = np.mean(all_results[size][method]['test_r2'])
                avg_coef_diff = np.mean(all_results[size][method]['coef_diff'])
                avg_speedup = np.mean(all_results[size][method]['train_time'])
                
                print(f"{method:<18} {avg_train_rmse:>12.4f} {avg_test_rmse:>12.4f} {avg_train_r2:>10.4f} "
                      f"{avg_test_r2:>10.4f} {avg_coef_diff:>11.4f} {avg_speedup:>10.2f}x")
    
    return all_results, full_model


def plot_results(results, sample_sizes, save_dir="plots_ridge", ridge_alpha=1000):
    """
    Generate comprehensive plots comparing different coreset methods for Ridge regression.
    
    Args:
        results (dict): Experimental results dictionary
        sample_sizes (list): List of sample sizes used in experiments
        save_dir (str): Directory to save plots
        ridge_alpha (float): Ridge regularization parameter value for filename
    """
    methods = list(results[sample_sizes[0]].keys())
    
    # Define consistent color scheme for methods
    colors = {
        'Uniform': '#1f77b4',              # Blue
        'HLSZ': '#ff7f0e',                 # Orange
        'RegularizedLeverage': '#2ca02c',  # Green
    }
    
    # Prepare data for plotting
    metrics = ['train_rmse', 'test_rmse', 'train_r2', 'test_r2', 'coef_diff', 'train_time']
    plot_data = {metric: {method: [] for method in methods} for metric in metrics}
    
    # Collect results
    for size in sample_sizes:
        for method in methods:
            if results[size][method]['train_rmse']:  # Check if we have results
                for metric in metrics:
                    plot_data[metric][method].append(np.mean(results[size][method][metric]))
            else:
                # Fill with NaN if no results
                for metric in metrics:
                    plot_data[metric][method].append(np.nan)
    
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create the figure with subplots
    plt.figure(figsize=(24, 6))
    
    plot_configs = [
        ('train_rmse', 'Training RMSE', 1),
        ('test_r2', 'Test R²', 2),
        ('coef_diff', 'Relative Coefficient Difference', 3),
        ('train_time', 'Training Speedup', 4)
    ]
    
    for metric, ylabel, subplot_idx in plot_configs:
        plt.subplot(1, 4, subplot_idx)
        
        for method in methods:
            data = plot_data[metric][method]
            # Filter out NaN values for plotting
            valid_indices = ~np.isnan(data)
            valid_sizes = np.array(sample_sizes)[valid_indices]
            valid_data = np.array(data)[valid_indices]
            
            if len(valid_data) > 0:
                plt.plot(valid_sizes, valid_data, marker='o', color=colors[method], 
                        label=method, linewidth=2, markersize=8)
        
        plt.xlabel('Sample Size', fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Adjust layout and save
    plt.tight_layout()
    filename = f"ridge_coreset_comparison_alpha_{ridge_alpha}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {os.path.join(save_dir, filename)}")


def main():
    """
    Main function to run the complete experimental pipeline for Ridge regression.
    """
    # Configuration
    DATA_PATH = "./financial_data.csv"
    REGULARIZATION_PARAMS = [100, 1000]
    SAMPLE_SIZES = [10, 25, 50, 75, 100, 200, 400, 500, 600, 800, 1000, 1500]
    NUM_REPETITIONS = 10
    
    # Load and prepare data
    print("Loading and preprocessing data...")
    try:
        data = pd.read_csv(DATA_PATH)
        print(f"Data loaded successfully. Shape: {data.shape}")
        print(data.info())
    except FileNotFoundError:
        print(f"Error: Could not find data file at {DATA_PATH}")
        print("Please ensure the financial_data.csv file is in the current directory.")
        return
    
    # Handle datetime column if present
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        
        # Extract temporal features
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month  
        data['day'] = data['date'].dt.day
        data['day_of_week'] = data['date'].dt.dayofweek
        data['quarter'] = data['date'].dt.quarter
        
        # Remove original date column
        data = data.drop(columns=['date'])
    
    # Handle missing values
    print("Handling missing values...")
    missing_before = data.isnull().sum().sum()
    
    for col in data.columns:
        if data[col].isnull().any():
            if col in ['usd_chf', 'eur_usd']:
                # Forward/backward fill for currency pairs
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
            else:
                # Mean imputation for other features
                data[col] = data[col].fillna(data[col].mean())
    
    missing_after = data.isnull().sum().sum()
    print(f"Missing values before: {missing_before}")
    print(f"Missing values after: {missing_after}")
    
    # Prepare features and target
    target_col = 'sp500 close'  # Adjust this based on your actual target column
    if target_col not in data.columns:
        # If the target column doesn't exist, use the last column as target
        target_col = data.columns[-1]
        print(f"Using '{target_col}' as target variable")
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Additional scaling for train/test splits
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train)
    X_test_scaled = scaler_final.transform(X_test)
    
    # Convert to numpy arrays
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    # Run experiments for different regularization parameters
    for alpha in REGULARIZATION_PARAMS:
        print(f"\n{'='*60}")
        print(f"Running experiments with Ridge alpha = {alpha}")
        print(f"{'='*60}")
        
        # Run comparison experiments
        results, full_model = run_experiment_with_comparison(
            X_train_scaled, 
            y_train, 
            X_test_scaled,
            y_test,
            sample_sizes=SAMPLE_SIZES,
            num_of_rep=NUM_REPETITIONS,
            ridge_alpha=alpha
        )
        
        # Generate and save plots
        save_directory = f"plots_ridge_research/alpha_{alpha}"
        plot_results(results, SAMPLE_SIZES, save_dir=save_directory, ridge_alpha=alpha)
        
        print(f"Results for alpha={alpha} completed and saved to {save_directory}")
    
    print("\nAll experiments completed successfully!")


if __name__ == "__main__":
    main()
