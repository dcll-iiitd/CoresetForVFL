"""
Coreset Methods for Logistic Regression: A Comparative Study

This module implements and compares different coreset sampling methods for logistic regression,
including uniform sampling and various coreset construction algorithms based on leverage scores
and Lewis weights.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os


def uniform_lr(m, D, y):
    """
    Uniform random sampling baseline method.
    
    Args:
        m (int): Number of samples to select
        D (np.ndarray): Feature matrix
        y (np.ndarray): Label vector
        
    Returns:
        tuple: (sampled_data, sampled_labels)
    """
    D = np.hstack((D, y.reshape(-1, 1)))
    D_df = pd.DataFrame(D)
    
    # Random sampling with one guaranteed positive class sample
    C = D_df.sample(n=m, replace=False)
    class_1_sample = D_df[D_df.iloc[:, -1] == 1].sample(n=1, random_state=42)
    C = pd.concat([C, class_1_sample])
    C = C.to_numpy()
    
    data = C[:, :-1]
    label = C[:, -1]
    return data, label


def coreset_lr_hlsz(m, D, y, alpha):
    """
    HLSZ (Huggins et al.) coreset method based on QR decomposition.
    
    This method constructs importance sampling probabilities using QR decomposition
    on different parts of the feature matrix.
    
    Args:
        m (int): Number of samples to select
        D (np.ndarray): Feature matrix
        y (np.ndarray): Label vector
        alpha (float): Regularization parameter
        
    Returns:
        tuple: (sampled_data, sampled_labels, sample_weights)
    """
    num_of_data, _ = D.shape
    D = np.hstack((D, y.reshape(-1, 1)))
    
    # Split features into three parts for QR decomposition
    D1 = D[:, :10]
    D2 = D[:, 10:20]
    D3 = D[:, 20:]
    
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
    
    # Ensure class balance
    class_1_sample = D_df[D_df.iloc[:, -2] == 1].sample(n=1, random_state=42)
    C = pd.concat([C, class_1_sample])
    C = C.to_numpy()
    
    data = C[:, :-2]
    label = C[:, -2]
    weight = C[:, -1]
    weight = weight / np.sum(weight) * m * 10
    
    return data, label, weight


def coreset_lr_regularized(m, D, y, alpha):
    """
    Squared leverage score coreset method with regularization.
    
    This method incorporates regularization into the leverage score computation
    for more stable coreset construction.
    
    Args:
        m (int): Number of samples to select
        D (np.ndarray): Feature matrix
        y (np.ndarray): Label vector
        alpha (float): Regularization parameter
        
    Returns:
        tuple: (sampled_data, sampled_labels, sample_weights)
    """
    num_of_data, _ = D.shape
    
    # Add regularization term
    sqrt_lambda_I_d = np.sqrt(10) * np.eye(D.shape[1])
    D_hat = np.vstack((D, sqrt_lambda_I_d))
    y_hat = np.hstack((y, np.zeros(D.shape[1])))
    D_hat = np.hstack((D_hat, y_hat.reshape(-1, 1)))
    D = np.hstack((D, y.reshape(-1, 1)))
    
    # Split augmented matrix for QR decomposition
    D1 = D_hat[:, :40]
    D2 = D_hat[:, 40:80]
    D3 = D_hat[:, 80:]
    
    # Compute QR decomposition
    q1, _ = np.linalg.qr(D1)
    q1 = q1[:num_of_data, :]
    q2, _ = np.linalg.qr(D2)
    q2 = q2[:num_of_data, :]
    q3, _ = np.linalg.qr(D3)
    q3 = q3[:num_of_data, :]
    
    # Compute importance weights
    q1 = (20 + 2 * 1) * (np.sqrt(np.sum(q1 ** 2, axis=1)) + 1/D1.shape[0])
    q2 = (20 + 2 * 1) * (np.sqrt(np.sum(q2 ** 2, axis=1)) + 1/D2.shape[0])
    q3 = (20 + 2 * 1) * (np.sqrt(np.sum(q3 ** 2, axis=1)) + 1/D3.shape[0])
    Q = q1 + q2 + q3
    
    D = np.hstack((D, (1/(Q/(np.sum(Q)))).reshape(-1, 1)))
    D_df = pd.DataFrame(D)
    
    # Sample with importance weights
    C = D_df.sample(n=m-1, replace=False, weights=Q / np.sum(Q))
    
    # Ensure class balance
    class_1_sample = D_df[D_df.iloc[:, -2] == 1].sample(n=1, random_state=42)
    class_0_sample = D_df[D_df.iloc[:, -2] == -1].sample(n=1, random_state=42)
    C = pd.concat([C, class_1_sample, class_0_sample])
    C = C.to_numpy()
    
    data = C[:, :-2]
    label = C[:, -2]
    weight = C[:, -1]
    weight = weight / m
    
    return data, label, weight


def lewis_iterate_initial(A, p, beta, w):
    """
    Single iteration of Lewis weight computation.
    
    Args:
        A (np.ndarray): Matrix for which to compute Lewis weights
        p (float): Lewis weight parameter
        beta (float): Scaling parameter
        w (np.ndarray): Current weight vector
        
    Returns:
        np.ndarray: Updated weight vector
    """
    w_half_inv_p = w ** (0.5 - (1 / p))
    WA = A * w_half_inv_p[:, np.newaxis]
    ATA_inv = np.linalg.pinv(WA.T @ WA)
    leverage_scores = np.einsum('ij,ji->i', A @ ATA_inv, A.T)
    leverage_scores_approx = beta * leverage_scores
    w = leverage_scores_approx ** (p / 2)
    return w


def approx_lewis_weights(A, p=1, beta=1, T=3):
    """
    Approximate Lewis weights computation using iterative method.
    
    Args:
        A (np.ndarray): Input matrix
        p (float): Lewis weight parameter (default: 1)
        beta (float): Scaling parameter (default: 1)
        T (int): Number of iterations (default: 3)
        
    Returns:
        np.ndarray: Approximate Lewis weights
    """
    num_rows = A.shape[0]
    w = np.ones(num_rows)
    for t in range(T):
        w = lewis_iterate_initial(A, p, beta, w)
    return w


def coreset_lr_lewis(m, D, y, alpha):
    """
    Lewis weight-based coreset method.
    
    This method uses Lewis weights to construct importance sampling probabilities
    for coreset selection in logistic regression.
    
    Args:
        m (int): Number of samples to select
        D (np.ndarray): Feature matrix
        y (np.ndarray): Label vector
        alpha (float): Regularization parameter
        
    Returns:
        tuple: (sampled_data, sampled_labels, sample_weights)
    """
    num_of_data, _ = D.shape
    D = np.hstack((D, y.reshape(-1, 1)))
    
    # Split features and apply label-dependent transformation
    D1 = D[:, :10]
    D2 = D[:, 10:20]
    D3 = D[:, 20:]
    
    # Adjust features based on labels for logistic regression
    D1_adjusted = D1 * np.where(y.reshape(-1, 1) == -1, -1, 1)
    D2_adjusted = D2 * np.where(y.reshape(-1, 1) == -1, -1, 1)
    D3_adjusted = D3[:, :-1] * np.where(y.reshape(-1, 1) == -1, -1, 1)
    
    # Compute Lewis weights for each part
    q1 = approx_lewis_weights(D1_adjusted)
    q2 = approx_lewis_weights(D2_adjusted)
    q3 = approx_lewis_weights(D3_adjusted)
    
    # Scale and combine weights
    q1 = (20 + 2 * 1) * (np.sqrt(3) * (q1**2))
    q2 = (20 + 2 * 1) * (np.sqrt(3) * (q2**2))
    q3 = (20 + 2 * 1) * (np.sqrt(3) * (q3**2))
    Q = q1 + q2 + q3
    
    D = np.hstack((D, (1/(Q/(np.sum(Q)))).reshape(-1, 1)))
    D_df = pd.DataFrame(D)
    
    # Sample with Lewis weights
    C = D_df.sample(n=m-1, replace=False, weights=Q / np.sum(Q))
    
    # Ensure class balance
    class_1_sample = D_df[D_df.iloc[:, -2] == 1].sample(n=1, random_state=42)
    class_0_sample = D_df[D_df.iloc[:, -2] == -1].sample(n=1, random_state=42)
    C = pd.concat([C, class_1_sample, class_0_sample])
    C = C.to_numpy()
    
    data = C[:, :-2]
    label = C[:, -2]
    weight = C[:, -1]
    weight = weight / m
    
    return data, label, weight


def coreset_lr_aug_lewis(m, D, y, alpha):
    """
    Augmented Lewis weight-based coreset method with regularization.
    
    This method combines Lewis weights with regularization for improved
    coreset construction in regularized logistic regression.
    
    Args:
        m (int): Number of samples to select
        D (np.ndarray): Feature matrix
        y (np.ndarray): Label vector
        alpha (float): Regularization parameter
        
    Returns:
        tuple: (sampled_data, sampled_labels, sample_weights)
    """
    p = 1
    num_of_data, _ = D.shape
    
    pth_root_alpha = (1/alpha) ** (1 / p)
    D = np.hstack((D, y.reshape(-1, 1)))
    
    # Process each feature block with regularization
    feature_blocks = [D[:, :10], D[:, 10:20], D[:, 20:]]
    lewis_weights = []
    
    for i, block in enumerate(feature_blocks):
        # Add regularization
        pth_lambda_I_d = pth_root_alpha * np.eye(block.shape[1])
        block_hat = np.vstack((block, pth_lambda_I_d))
        y_hat = np.hstack((y, np.zeros(block.shape[1])))
        block_hat = np.hstack((block_hat, y_hat.reshape(-1, 1)))
        
        # Apply label-dependent transformation
        block_adjusted = block_hat * np.where(y_hat.reshape(-1, 1) == -1, -1, 1)
        
        # Compute Lewis weights
        q = approx_lewis_weights(block_adjusted)
        q = q[:num_of_data]  # Keep only original data weights
        q = (20 + 2 * 1) * (np.sqrt(3) * (q**2))
        lewis_weights.append(q)
    
    # Combine all Lewis weights
    Q = sum(lewis_weights)
    
    D = np.hstack((D, (1/(Q/(np.sum(Q)))).reshape(-1, 1)))
    D_df = pd.DataFrame(D)
    
    # Sample with combined weights
    C = D_df.sample(n=m-2, replace=False, weights=Q / np.sum(Q))
    
    # Ensure class balance
    class_1_sample = D_df[D_df.iloc[:, -2] == 1].sample(n=1, random_state=42)
    class_0_sample = D_df[D_df.iloc[:, -2] == -1].sample(n=1, random_state=42)
    C = pd.concat([C, class_1_sample, class_0_sample])
    C = C.to_numpy()
    
    data = C[:, :-2]
    label = C[:, -2]
    weight = C[:, -1]
    weight = weight / m
    
    return data, label, weight


def run_experiment_with_comparison(X_train, y_train, X_test, y_test, 
                                 sample_sizes=[50, 150, 300, 500, 700, 900, 1200, 1500, 2000, 2500, 5000], 
                                 num_of_rep=3, C=1.0, alpha=0.1):
    """
    Run comprehensive comparison experiments between different sampling methods.
    
    Args:
        X_train (np.ndarray): Training feature matrix
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Test feature matrix
        y_test (np.ndarray): Test labels
        sample_sizes (list): List of sample sizes to evaluate
        num_of_rep (int): Number of repetitions per experiment
        C (float): Regularization parameter for logistic regression
        alpha (float): Alpha parameter for coreset methods
        
    Returns:
        tuple: (experiment_results, full_model)
    """
    methods = {
        'Uniform': uniform_lr,
        'HLSZ': coreset_lr_hlsz,
        'SqLev': coreset_lr_regularized,
        'Lewis': coreset_lr_lewis,
        'AugLewis': coreset_lr_aug_lewis,
    }
    
    # Train full model for comparison baseline
    print("Training full model for baseline comparison...")
    full_model = LogisticRegression(C=C, max_iter=5000)
    start_full_time = time.time()
    full_model.fit(X_train, y_train)
    full_train_time = time.time() - start_full_time
    
    # Initialize results storage
    all_results = {size: {method: {
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': [],
        'y_test': [],
        'y_pred': [],
        'weights_diff': [],
        'train_time': []
    } for method in methods.keys()} for size in sample_sizes}
    
    # Run experiments for each sample size
    for size in sample_sizes:
        print(f"\n{'='*50}")
        print(f"Sample Size: {size}")
        print(f"{'='*50}")
        
        model = LogisticRegression(C=C, max_iter=3000)
        
        for method_name, sampling_method in methods.items():
            print(f"Running {method_name} method...")
            
            for rep in range(num_of_rep):
                try:
                    # Sample data using the current method
                    if method_name == 'Uniform':
                        sample_X, sample_y = sampling_method(size, X_train, y_train)
                        weights = None
                    else:
                        sample_X, sample_y, weights = sampling_method(size, X_train, y_train, alpha)
                    
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
                    train_acc = balanced_accuracy_score(y_train, pred_train)
                    test_acc = balanced_accuracy_score(y_test, pred_test)
                    train_loss = log_loss(y_train, model.predict_proba(X_train))
                    test_loss = log_loss(y_test, model.predict_proba(X_test))
                    
                    # Calculate relative coefficient difference from full model
                    weights_diff = (np.linalg.norm(model.coef_.flatten() - full_model.coef_.flatten()) / 
                                  np.linalg.norm(full_model.coef_.flatten()))
                    
                    # Store results
                    all_results[size][method_name]['train_acc'].append(train_acc)
                    all_results[size][method_name]['test_acc'].append(test_acc)
                    all_results[size][method_name]['train_loss'].append(train_loss)
                    all_results[size][method_name]['test_loss'].append(test_loss)
                    all_results[size][method_name]['y_test'].extend(y_test.tolist())
                    all_results[size][method_name]['y_pred'].extend(pred_test.tolist())
                    all_results[size][method_name]['weights_diff'].append(weights_diff)
                    all_results[size][method_name]['train_time'].append(full_train_time / elapsed_time)
                    
                except Exception as e:
                    print(f"Error in {method_name} method, repetition {rep}: {e}")
                    continue
        
        # Print summary for current sample size
        print(f"\nSummary for Sample Size {size}:")
        print(f"{'Method':<15} {'Train Acc':>10} {'Test Acc':>10} {'Train Loss':>11} {'Test Loss':>10} {'Weight Diff':>11} {'Speedup':>10}")
        print('-' * 85)
        
        for method in methods.keys():
            if all_results[size][method]['train_acc']:  # Check if we have results
                avg_train_acc = np.mean(all_results[size][method]['train_acc'])
                avg_test_acc = np.mean(all_results[size][method]['test_acc'])
                avg_train_loss = np.mean(all_results[size][method]['train_loss'])
                avg_test_loss = np.mean(all_results[size][method]['test_loss'])
                avg_weight_diff = np.mean(all_results[size][method]['weights_diff'])
                avg_speedup = np.mean(all_results[size][method]['train_time'])
                
                print(f"{method:<15} {avg_train_acc:>10.4f} {avg_test_acc:>10.4f} {avg_train_loss:>11.4f} "
                      f"{avg_test_loss:>10.4f} {avg_weight_diff:>11.4f} {avg_speedup:>10.2f}x")
    
    return all_results, full_model


def plot_results(results, sample_sizes, save_dir="plots_logistic", c=1.0):
    """
    Generate comprehensive plots comparing different coreset methods.
    
    Args:
        results (dict): Experimental results dictionary
        sample_sizes (list): List of sample sizes used in experiments
        save_dir (str): Directory to save plots
        c (float): Regularization parameter value for filename
    """
    methods = list(results[sample_sizes[0]].keys())
    
    # Define consistent color scheme for methods
    colors = {
        'Uniform': '#1f77b4',    # Blue
        'HLSZ': '#ff7f0e',       # Orange
        'SqLev': '#2ca02c',      # Green
        'Lewis': '#d62728',      # Red
        'AugLewis': '#9467bd'    # Purple
    }
    
    # Prepare data for plotting
    metrics = ['train_acc', 'test_acc', 'train_loss', 'test_loss', 'weights_diff', 'train_time']
    plot_data = {metric: {method: [] for method in methods} for metric in metrics}
    
    # Collect results
    for size in sample_sizes:
        for method in methods:
            if results[size][method]['train_acc']:  # Check if we have results
                for metric in metrics:
                    if metric == 'weights_diff':
                        plot_data[metric][method].append(np.mean(results[size][method][metric]))
                    else:
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
        ('train_loss', 'Training Loss', 1),
        ('test_acc', 'Test Accuracy', 2),
        ('weights_diff', 'Relative Coefficient Difference', 3),
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
    filename = f"coreset_comparison_C_{c}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {os.path.join(save_dir, filename)}")


def main():
    """
    Main function to run the complete experimental pipeline.
    """
    # Configuration
    DATA_PATH = "./creditcard.csv"
    REGULARIZATION_PARAMS = [0.0001, 0.001, 0.01, 0.1]
    SAMPLE_SIZES = [50, 150, 300, 500, 700, 900, 1200, 1500, 2000, 2500]
    NUM_REPETITIONS = 10
    
    # Load and prepare data
    print("Loading and preprocessing data...")
    try:
        data = pd.read_csv(DATA_PATH)
        print(f"Data loaded successfully. Shape: {data.shape}")
        print(data.info())
    except FileNotFoundError:
        print(f"Error: Could not find data file at {DATA_PATH}")
        print("Please ensure the creditcard.csv file is in the current directory.")
        return
    
    # Prepare features and labels
    X = data.drop(columns=['Class'])
    y = data['Class']
    y = y.replace({0: -1, 1: 1})  # Convert to {-1, 1} labels for logistic regression
    
    print("Unique values in the target variable:", y.unique())
    print("Class distribution before balancing:")
    print(y.value_counts())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Balance classes using SMOTE
    print("Balancing classes using SMOTE...")
    sampler = SMOTE(random_state=52)
    X_balanced, y_balanced = sampler.fit_resample(X_scaled, y)
    
    print("Class distribution after balancing:")
    print(pd.Series(y_balanced).value_counts())
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=52, stratify=y_balanced
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Additional scaling for train/test splits
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train)
    X_test_scaled = scaler_final.transform(X_test)
    
    # Run experiments for different regularization parameters
    for c in REGULARIZATION_PARAMS:
        print(f"\n{'='*60}")
        print(f"Running experiments with C = {c}")
        print(f"{'='*60}")
        
        # Run comparison experiments
        results, full_model = run_experiment_with_comparison(
            X_train_scaled, 
            y_train, 
            X_test_scaled,
            y_test,
            sample_sizes=SAMPLE_SIZES,
            num_of_rep=NUM_REPETITIONS,
            C=c,
            alpha=1/c,  # Set alpha as inverse of C
        )
        
        # Generate and save plots
        save_directory = f"plots_logistic_research/C_{c}"
        plot_results(results, SAMPLE_SIZES, save_dir=save_directory, c=c)
        
        print(f"Results for C={c} completed and saved to {save_directory}")
    
    print("\nAll experiments completed successfully!")


if __name__ == "__main__":
    main()
