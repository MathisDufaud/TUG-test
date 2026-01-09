import os
from collections import Counter
import pickle

import pandas as pd
import numpy as np
import scipy.signal as signal
from scipy.stats import entropy

import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from SaraFolder.settings import classes, utils_darioalgo, running_settings, utils_evaluation, utils_plots, \
    utils_dataquality
from SaraFolder.settings.utils_parkaapp import utils_parkapp


def setup_manual_labelling_csv(all_tests, filename):
    # Read csv
    df_manual = pd.read_csv(running_settings.results_all + os.sep + filename)

    for test in all_tests:
        indexid = test.user_id + '_' + str(test.session_id)
        df_test = df_manual[df_manual['Unnamed: 0'] == indexid]
        secStart = df_test['secStart'].values[0]
        secEnd = df_test['secEnd'].values[0]
        msStart = secStart * 1000
        msEnd = secEnd * 1000

        test.processed_data['testBool'] = False
        test.processed_data.loc[(test.processed_data['msFromStart'] >= msStart) &
                                (test.processed_data['msFromStart'] <= msEnd), 'testBool'] = True
    return None


def prep_data(all_tests, fold_idx=None, n_splits=5, window_size=60, stride=30, input_type='triaxial_acc', output_steps=0):
    """
    Prepare data for ML training while maintaining test boundaries.

    Args:
        all_tests: List of TUG test objects
        fold_idx: Tuple of (train_indices, val_indices, holdout_indices) for current fold, or None for all data
        n_splits: Number of folds for cross-validation
        window_size: Size of the sliding window for sequences
        stride: Step size for sliding window (default=1 for no overlap)

    Returns:
        X, y, test_indices (to be split later by split_data)
    """
    target_col = 'testBool'
    if input_type == 'triaxial':
        cols = ['acc.x', 'acc.y', 'acc.z', 'rotRate.alpha', 'rotRate.beta',
                'rotRate.gamma', 'alpha', 'beta', 'gamma']
    elif input_type == 'magnitude_acc':
        cols = ['sqrt(X²+Y²+Z²)', 'rotRate.alpha', 'rotRate.beta',
                'rotRate.gamma', 'alpha', 'beta', 'gamma']
    elif input_type == 'sixaxial':
        cols = ['acc.x', 'acc.y', 'acc.z', 'rotRate.alpha', 'rotRate.beta', 'rotRate.gamma']

    X_list = []
    y_list = []
    test_indices = []  # Track which test each sequence belongs to

    for test_idx, test in enumerate(all_tests):
        test_data = test.processed_data[cols].values
        target_data = test.processed_data[target_col].values

        # Normalize data per test
        if False:
            mean = np.mean(test_data, axis=0, keepdims=True)
            std = np.std(test_data, axis=0, keepdims=True) + 1e-8
            test_data = (test_data - mean) / std

        # Create sliding windows for this test
        n_samples = len(test_data)
        if n_samples < window_size:
            print(f"Warning: Test {test_idx} has only {n_samples} samples, skipping...")
            continue

        # Create sequences with stride
        # for i in range(0, n_samples - window_size + 1, stride):
        #     X_list.append(test_data[i:i + window_size])
        #     y_list.append(target_data[i:i + window_size])
        #     test_indices.append(test_idx)

        if output_steps>0:
            target_start = window_size - output_steps
        else:
            target_start = 0

        for i in range(0, n_samples - window_size + 1, stride):
            X_list.append(test_data[i:i + window_size])
            y_list.append(target_data[i + target_start:i + window_size])
            test_indices.append(test_idx)

    X = np.array(X_list) # Shape: (n_sequences, window_size, n_features)
    if output_steps>0:
        y = np.expand_dims(np.array(y_list), axis=-1)  # (N, 15, 1)
    else:
        y = np.array(y_list)  # Shape: (n_sequences, window_size)
        # Add dimension for binary classification if needed
        if len(y.shape) == 2:
            y = np.expand_dims(y, axis=-1)  # Shape: (n_sequences, window_size, 1)

    test_indices = np.array(test_indices)

    return X, y, test_indices


def split_data(X, y, test_indices, test_index, fold_idx=None):
    """
    Split data into train/val sets based on test indices.

    Args:
        X: Feature array
        y: Target array
        test_indices: Array indicating which test each sequence belongs to
        fold_idx: Tuple of (train_test_indices, val_test_indices, holdout_test_indices) or None

    Returns:
        X_train, X_val, y_train, y_val
    """
    if fold_idx is not None:
        train_test_indices, val_test_indices = fold_idx

        # Get sequences belonging to training tests
        train_mask = np.isin(test_indices, train_test_indices)
        X_train = X[train_mask]
        y_train = y[train_mask]

        # Get sequences belonging to validation tests
        val_mask = np.isin(test_indices, val_test_indices)
        X_val = X[val_mask]
        y_val = y[val_mask]

        # Get sequences belonging to holdout tests
        # holdout_mask = np.isin(test_indices, holdout_test_indices)
        # X_holdout = X[holdout_mask]
        # y_holdout = y[holdout_mask]

    else:
        # Split by unique test indices (not by sequences)
        unique_tests = np.unique(test_indices)
        n_tests = len(unique_tests)

        # Split test indices: 70% train, 15% val, 15% holdout
        from sklearn.model_selection import train_test_split

        train_val_tests, holdout_tests = train_test_split(
            unique_tests,
            test_size=0.15,
            random_state=42,
            shuffle=True
        )

        train_tests, val_tests = train_test_split(
            train_val_tests,
            test_size=0.176,  # 0.15 / (1 - 0.15) ≈ 0.176 to get 15% of total
            random_state=42,
            shuffle=True
        )

        # Create masks based on test membership
        train_mask = np.isin(test_indices, train_tests)
        val_mask = np.isin(test_indices, val_tests)
        holdout_mask = np.isin(test_indices, holdout_tests)

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[val_mask]
        y_val = y[val_mask]

        print(f"Split by tests: {len(train_tests)} train, {len(val_tests)} val, {len(holdout_tests)} holdout")

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    if len(X_val) > 0:
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # Print class distributions
    print(f"Train class balance: {np.mean(y_train):.3f}")
    if len(y_val) > 0:
        print(f"Val class balance: {np.mean(y_val):.3f}")

    return X_train, X_val, y_train, y_val


"""def evaluate_cv(modelObj, X_val, y_val, fold, cv_results, fold_models, best_val_f1):
    # Evaluate on validation set

    y_pred = modelObj.fitted_model.predict(X_val, verbose=0)
    y_pred_binary = (y_pred > 0.5).astype(int)

    y_val_flat = y_val.reshape(-1)
    y_pred_flat = y_pred_binary.reshape(-1)

    val_loss, val_acc = modelObj.fitted_model.evaluate(X_val, y_val, verbose=0)
    precision = precision_score(y_val_flat, y_pred_flat)
    recall = recall_score(y_val_flat, y_pred_flat)
    f1 = f1_score(y_val_flat, y_pred_flat)

    cv_results.append({
        'fold': fold + 1,
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'val_precision': precision,
        'val_recall': recall,
        'val_f1_score': f1
    })
    fold_models.append(modelObj)

    # Track best fold by F1 score
    if f1 > best_val_f1:
        best_val_f1 = f1
        best_fold_idx = fold

    print(f"Fold {fold + 1} - Val Acc: {val_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return cv_results, fold_models, best_val_f1, best_fold_idx"""


def evaluate_holdout(best_model, best_scaler, X, y, test_indices, test_index, holdout_tests,
                     save=False):
    print(f"\n{'=' * 60}")
    print(f"HOLDOUT SET EVALUATION)")
    print(f"{'=' * 60}")

    X_holdout = X[np.isin(test_index, holdout_tests)]
    y_holdout = y[np.isin(test_index, holdout_tests)]

    X_holdout_flat = X_holdout.reshape(-1, X_holdout.shape[-1])
    X_holdout_scaled = best_scaler.fit_transform(X_holdout_flat)
    X_holdout = X_holdout_scaled.reshape(X_holdout.shape)

    print(f"Evaluation with X_holdout shaped: {X_holdout.shape}")

    # Predict on holdout
    y_holdout_pred = best_model.fitted_model.predict(X_holdout, verbose=2)
    y_holdout_pred_binary = (y_holdout_pred > 0.6).astype(int)

    y_holdout_flat = y_holdout.reshape(-1)
    y_holdout_pred_flat = y_holdout_pred_binary.reshape(-1)

    # Check for unknown or binary targets
    if y_holdout_flat.dtype == 'object':
        y_holdout_flat = y_holdout_flat.astype(int)

    holdout_acc = accuracy_score(y_holdout_flat, y_holdout_pred_flat)
    holdout_precision = precision_score(y_holdout_flat, y_holdout_pred_flat)
    holdout_recall = recall_score(y_holdout_flat, y_holdout_pred_flat)
    holdout_f1 = f1_score(y_holdout_flat, y_holdout_pred_flat)

    print(f"\nHoldout Results (TRUE generalization performance):")
    print(f"  Accuracy:  {holdout_acc:.4f}")
    print(f"  Precision: {holdout_precision:.4f}")
    print(f"  Recall:    {holdout_recall:.4f}")
    print(f"  F1-Score:  {holdout_f1:.4f}")

    # Save holdout results
    holdout_results = pd.DataFrame([{
        'holdout_accuracy': holdout_acc,
        'holdout_precision': holdout_precision,
        'holdout_recall': holdout_recall,
        'holdout_f1': holdout_f1,
        'n_holdout_tests': len(holdout_tests)
    }])

    if save:
        holdout_results.to_csv(running_settings.results_all + os.sep + 'holdout_results.csv', index=False)

    return holdout_results


def reconstruct_from_windows_output(predictions, window_size, stride, original_length, output_steps):
    # TODO: check
    """
    Reconstruct original sequence from overlapping window predictions
    when each window predicts only the last `output_steps` timesteps.

    Parameters
    ----------
    predictions : array-like, shape (n_windows, output_steps, n_features)
        Model predictions for each window (last 15 timesteps per 60-sample window).
    window_size : int
        Size of each input window (e.g., 60).
    stride : int
        Step size used when sliding windows (e.g., 30).
    original_length : int
        Length of the original full sequence.
    output_steps : int
        Number of timesteps predicted per window (e.g., 15).

    Returns
    -------
    reconstructed : array, shape (original_length, n_features)
        Sequence reconstructed by averaging overlapping predictions.
    counts : array, shape (original_length,)
        Number of predictions contributing to each timestep.
    """
    n_windows = len(predictions)
    n_features = predictions.shape[-1] if predictions.ndim == 3 else 1

    reconstructed = np.zeros((original_length, n_features) if n_features > 1 else (original_length,))
    counts = np.zeros(original_length)

    for window_idx in range(n_windows):
        # Position of this window in the original signal
        start_idx = window_idx * stride
        pred_start = start_idx + (window_size - output_steps)   # only last 15 samples
        pred_end = pred_start + output_steps

        # Handle boundary case (end of sequence)
        valid_end = min(pred_end, original_length)
        valid_steps = valid_end - pred_start
        if valid_steps <= 0:
            continue

        if n_features > 1:
            reconstructed[pred_start:valid_end] += predictions[window_idx, :valid_steps]
        else:
            reconstructed[pred_start:valid_end] += predictions[window_idx, :valid_steps].flatten()

        counts[pred_start:valid_end] += 1

    # Average overlapping predictions
    counts = np.maximum(counts, 1)
    if n_features > 1:
        reconstructed = reconstructed / counts[:, np.newaxis]
    else:
        reconstructed = reconstructed / counts

    return reconstructed, counts

def reconstruct_from_windows(predictions, window_size, stride, original_length):
    """
    Reconstruct original sequence from overlapping window predictions.

    Parameters:
    -----------
    predictions : array-like, shape (n_windows, window_size, n_features)
        Predictions from windowed data
    window_size : int
        Size of each window
    stride : int
        Stride used when creating windows
    original_length : int
        Length of the original sequence

    Returns:
    --------
    reconstructed : array, shape (original_length, n_features)
        Reconstructed predictions in original sequence length
    counts : array, shape (original_length,)
        Number of predictions averaged for each position
    """
    n_windows = len(predictions)
    n_features = predictions.shape[-1] if len(predictions.shape) > 2 else 1

    # Initialize arrays
    reconstructed = np.zeros((original_length, n_features) if n_features > 1 else (original_length,))
    counts = np.zeros(original_length)

    # Accumulate predictions
    for window_idx in range(n_windows):
        start_idx = window_idx * stride
        end_idx = start_idx + window_size

        # Handle edge case where window extends beyond original length
        valid_end = min(end_idx, original_length)
        valid_window_size = valid_end - start_idx

        if n_features > 1:
            reconstructed[start_idx:valid_end] += predictions[window_idx, :valid_window_size]
        else:
            #if not isinstance(predictions[0,0][0], np.int32):
                # Transform values into int
            #    predictions = predictions.astype(int)
            reconstructed[start_idx:valid_end] += predictions[window_idx, :valid_window_size].flatten()
            # Careful, here we are over-writing the part with stride overlap

        counts[start_idx:valid_end] += 1

    # Average overlapping predictions
    counts = np.maximum(counts, 1)  # Avoid division by zero
    if n_features > 1:
        reconstructed = reconstructed / counts[:, np.newaxis]
    else:
        reconstructed = reconstructed / counts

    return reconstructed, counts


def visualize_reconstruction(y_true, y_pred_windowed, y_pred_reconstructed,
                             window_size, stride, overlap_counts):
    """
    Visualize the reconstruction process and overlap regions.
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    # Plot 1: Original vs Reconstructed predictions
    ax = axes[0]
    x = np.arange(len(y_true))
    ax.plot(x, y_true, 'g-', label='Ground Truth', alpha=0.7, linewidth=2)
    ax.plot(x, y_pred_reconstructed, 'r-', label='Reconstructed Predictions', alpha=0.7, linewidth=2)
    ax.set_title('Original Size: Ground Truth vs Reconstructed Predictions', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Overlap counts (shows how many windows predicted each position)
    ax = axes[1]
    ax.bar(x, overlap_counts, color='steelblue', alpha=0.7)
    ax.set_title(f'Prediction Overlap (Window={window_size}, Stride={stride})',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Number of Overlapping Windows')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Prediction confidence based on overlap
    ax = axes[2]
    # More overlap = potentially more reliable
    confidence = overlap_counts / np.max(overlap_counts)
    ax.fill_between(x, 0, confidence, color='purple', alpha=0.5, label='Confidence')
    ax.plot(x, y_pred_reconstructed, 'r-', label='Predictions', linewidth=2)
    ax.set_title('Prediction Confidence Based on Overlap', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value / Confidence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def evaluate_holdout_per_test(modelObj, best_scaler, X_val, y_val, val_test_index, val_tests, fold, val_tests_original, output_steps=0, dataset='all'):
    """
    Evaluate model performance at the TEST level, aggregating predictions per test.

    Args:
        modelObj: Trained model object
        X: Full feature array
        y: Full target array
        test_index: Array indicating which test each sequence belongs to
        val_tests: List of test IDs in validation set
        fold: Current fold number

    Returns:
        test_results_df: DataFrame with per-test metrics
        test_level_metrics: Dictionary with aggregated test-level performance
    """

    # Calculate per-test metrics
    test_results = {}

    for x, test_id in enumerate(val_tests):
        print(f"\nEvaluating Test ID: {test_id}, Fold {fold + 1}, {x}/{len(val_tests)}")
        # Get all sequences belonging to this test
        test_mask = (val_test_index == test_id)
        original_test = val_tests_original[test_id]

        if test_mask.sum() == 0:
            continue

        X_val_test = X_val[test_mask]
        y_val_test = y_val[test_mask]

        X_val_test_flat = X_val_test.reshape(-1, X_val_test.shape[-1])
        X_val_test_scaled = best_scaler.transform(X_val_test_flat)
        X_val_test = X_val_test_scaled.reshape(X_val_test.shape)

        # Get predictions for validation set
        y_pred_test = modelObj.fitted_model.predict(X_val_test, verbose=0)

        # === RECONSTRUCTION TO ORIGINAL SIZE ===
        if output_steps==0:
            y_pred_reconstructed, overlap_counts = reconstruct_from_windows(
                predictions=y_pred_test,
                window_size=60,  # Your window size
                stride=running_settings.parameters['stride'],  # Your stride
                original_length=len(original_test.processed_data)
            )

            if dataset == 'matey_sanz':
                y_val_test = y_val_test.astype(int)
            # Also reconstruct ground truth
            y_true_original, overlap_counts_true = reconstruct_from_windows(
                predictions=y_val_test,
                window_size=60,
                stride=running_settings.parameters['stride'],
                original_length=len(original_test.processed_data)
            )
        else:
            y_pred_reconstructed, overlap_counts = reconstruct_from_windows_output(
                predictions=y_pred_test,
                window_size=60,  # Your window size
                stride=running_settings.parameters['stride'],  # Your stride
                original_length=len(original_test.processed_data),
                output_steps=output_steps
            )

            # Also reconstruct ground truth
            y_true_original, overlap_counts_true = reconstruct_from_windows_output(
                predictions=y_val_test,
                window_size=60,
                stride=running_settings.parameters['stride'],
                original_length=len(original_test.processed_data),
                output_steps=output_steps
            )


        # Now you have original-sized arrays
        print(f"Original test length: {len(original_test.processed_data)}")
        print(f"Reconstructed predictions shape: {y_pred_reconstructed.shape}")
        print(f"Reconstructed ground truth shape: {y_true_original.shape}")

        # Apply threshold for binary classification
        y_pred_binary_reconstructed = (y_pred_reconstructed > running_settings.parameters['classBinaryTresh']).astype(int)
        original_test.processed_data['predicted_testBool'] = y_pred_binary_reconstructed
        original_test.processed_data['predicted_testProba'] = y_pred_reconstructed

        # Calculate metrics on original-sized data
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(y_true_original, y_pred_binary_reconstructed)
        precision = precision_score(y_true_original, y_pred_binary_reconstructed, zero_division=0)
        recall = recall_score(y_true_original, y_pred_binary_reconstructed, zero_division=0)
        f1 = f1_score(y_true_original, y_pred_binary_reconstructed, zero_division=0)

        print(f"\nMetrics on reconstructed original-sized data:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        test_results[test_id] = {
            'test_id': test_id,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    test_results_df = pd.DataFrame(test_results).transpose()

    # Calculate aggregated test-level metrics (average across tests)
    test_level_metrics = {
        'fold': fold + 1,
        'n_tests': len(val_tests),
        'mean_test_accuracy': test_results_df['accuracy'].mean(),
        'std_test_accuracy': test_results_df['accuracy'].std(),
        'mean_test_precision': test_results_df['precision'].mean(),
        'std_test_precision': test_results_df['precision'].std(),
        'mean_test_recall': test_results_df['recall'].mean(),
        'std_test_recall': test_results_df['recall'].std(),
        'mean_test_f1': test_results_df['f1_score'].mean(),
        'std_test_f1': test_results_df['f1_score'].std(),
    }

    # Print test-level summary
    print(f"\nTest-Level Performance (averaging across {len(val_tests)} tests):")
    print(
        f"  Mean Accuracy:  {test_level_metrics['mean_test_accuracy']:.4f} ± {test_level_metrics['std_test_accuracy']:.4f}")
    print(
        f"  Mean Precision: {test_level_metrics['mean_test_precision']:.4f} ± {test_level_metrics['std_test_precision']:.4f}")
    print(
        f"  Mean Recall:    {test_level_metrics['mean_test_recall']:.4f} ± {test_level_metrics['std_test_recall']:.4f}")
    print(f"  Mean F1-Score:  {test_level_metrics['mean_test_f1']:.4f} ± {test_level_metrics['std_test_f1']:.4f}")

    return test_results_df, test_level_metrics, val_tests_original


def evaluate_cv_per_test(modelObj, X_val, y_val, val_test_index, val_tests, fold, val_tests_original, output_steps):
    """
    Evaluate model performance at the TEST level, aggregating predictions per test.

    Args:
        modelObj: Trained model object
        X: Full feature array
        y: Full target array
        test_index: Array indicating which test each sequence belongs to
        val_tests: List of test IDs in validation set
        fold: Current fold number

    Returns:
        test_results_df: DataFrame with per-test metrics
        test_level_metrics: Dictionary with aggregated test-level performance
    """

    # Calculate per-test metrics
    test_results = {}

    for x, test_id in enumerate(val_tests):
        print(f"\nEvaluating Test ID: {test_id}, Fold {fold + 1}, {x}/{len(val_tests)}")
        # Get all sequences belonging to this test
        test_mask = (val_test_index == test_id)
        original_test = val_tests_original[test_id]

        if test_mask.sum() == 0:
            continue

        X_val_test = X_val[test_mask]
        y_val_test = y_val[test_mask]

        # Get predictions for validation set
        y_pred_test = modelObj.fitted_model.predict(X_val_test, verbose=0)

        if output_steps == 0:
            y_pred_reconstructed, overlap_counts = reconstruct_from_windows(
                predictions=y_pred_test,
                window_size=60,  # Your window size
                stride=running_settings.parameters['stride'],           # Your stride
                original_length=len(original_test.processed_data)
            )

            y_true_original, overlap_counts_true = reconstruct_from_windows(
                predictions=y_val_test,
                window_size=60,
                stride=running_settings.parameters['stride'],
                original_length=len(original_test.processed_data)
            )
        else:
            y_pred_reconstructed, overlap_counts = reconstruct_from_windows_output(
                predictions=y_pred_test,
                window_size=60,  # Your window size
                stride=running_settings.parameters['stride'],  # Your stride
                original_length=len(original_test.processed_data),
                output_steps=output_steps
            )

            y_true_original, overlap_counts_true = reconstruct_from_windows_output(
                predictions=y_val_test,
                window_size=60,
                stride=running_settings.parameters['stride'],
                original_length=len(original_test.processed_data),
                output_steps=output_steps
            )

        # Now you have original-sized arrays
        print(f"Original test length: {len(original_test.processed_data)}")
        print(f"Reconstructed predictions shape: {y_pred_reconstructed.shape}")
        print(f"Reconstructed ground truth shape: {y_true_original.shape}")

        # Apply threshold for binary classification
        y_pred_binary_reconstructed = (y_pred_reconstructed > running_settings.parameters['classBinaryTresh']).astype(int)
        original_test.processed_data['predicted_testProba'] = y_pred_reconstructed
        original_test.processed_data['predicted_testBool'] = y_pred_binary_reconstructed

        # Calculate metrics on original-sized data
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(y_true_original, y_pred_binary_reconstructed)
        precision = precision_score(y_true_original, y_pred_binary_reconstructed, zero_division=0)
        recall = recall_score(y_true_original, y_pred_binary_reconstructed, zero_division=0)
        f1 = f1_score(y_true_original, y_pred_binary_reconstructed, zero_division=0)

        print(f"\nMetrics on reconstructed original-sized data:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        test_results[test_id] = {
            'test_id': test_id,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    test_results_df = pd.DataFrame(test_results).transpose()

    # Calculate aggregated test-level metrics (average across tests)
    test_level_metrics = {
        'fold': fold + 1,
        'n_tests': len(val_tests),
        'mean_test_accuracy': test_results_df['accuracy'].mean(),
        'std_test_accuracy': test_results_df['accuracy'].std(),
        'mean_test_precision': test_results_df['precision'].mean(),
        'std_test_precision': test_results_df['precision'].std(),
        'mean_test_recall': test_results_df['recall'].mean(),
        'std_test_recall': test_results_df['recall'].std(),
        'mean_test_f1': test_results_df['f1_score'].mean(),
        'std_test_f1': test_results_df['f1_score'].std(),
    }

    # Print test-level summary
    print(f"\nTest-Level Performance (averaging across {len(val_tests)} tests):")
    print(
        f"  Mean Accuracy:  {test_level_metrics['mean_test_accuracy']:.4f} ± {test_level_metrics['std_test_accuracy']:.4f}")
    print(
        f"  Mean Precision: {test_level_metrics['mean_test_precision']:.4f} ± {test_level_metrics['std_test_precision']:.4f}")
    print(
        f"  Mean Recall:    {test_level_metrics['mean_test_recall']:.4f} ± {test_level_metrics['std_test_recall']:.4f}")
    print(f"  Mean F1-Score:  {test_level_metrics['mean_test_f1']:.4f} ± {test_level_metrics['std_test_f1']:.4f}")

    return test_results_df, test_level_metrics, val_tests_original


def evaluate_cv(modelObj, X_val, y_val, fold, cv_results, fold_models, best_val_f1, best_fold_idx):
    """
    Evaluate model on validation set at sample level and update CV results.

    Args:
        modelObj: Trained model
        X_val: Validation features
        y_val: Validation targets
        fold: Current fold number
        cv_results: List of results dictionaries
        fold_models: List of model objects
        best_val_f1: Current best F1 score

    Returns:
        cv_results, fold_models, best_val_f1, best_fold_idx
    """
    from sklearn.metrics import f1_score

    # Evaluate on validation set (sample-level)
    val_loss, val_acc, val_prec, val_recall = modelObj.fitted_model.evaluate(X_val, y_val, verbose=0)

    y_pred = modelObj.fitted_model.predict(X_val, verbose=0)
    y_pred_binary = (y_pred > running_settings.parameters['classBinaryTresh']).astype(int)

    y_val_flat = y_val.reshape(-1)
    y_pred_flat = y_pred_binary.reshape(-1)

    f1 = f1_score(y_val_flat, y_pred_flat, zero_division=0)

    cv_results.append({
        'fold': fold + 1,
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'val_precision': val_prec,
        'val_recall': val_recall,
        'val_f1_score': f1
    })

    fold_models.append(modelObj)

    # Track best fold
    if f1 > best_val_f1:
        best_val_f1 = f1
        best_fold_idx = fold


    print(f"\nFold {fold + 1} Sample-Level Validation Results:")
    print(f"  Loss:      {val_loss:.4f}")
    print(f"  Accuracy:  {val_acc:.4f}")
    print(f"  Precision: {val_prec:.4f}")
    print(f"  Recall:    {val_recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    return cv_results, fold_models, best_val_f1, best_fold_idx




def plot_ml_prediction(test, error=None):
    """
    Plot binary classification predictions with ground truth markers and sensor data.

    Parameters:
    -----------
    test : object
        Test object containing processed_data with columns:
        - msFromStart: timestamps
        - predicted_testProba: prediction probabilities
        - predicted_testBool: binary predictions
        - testBool: ground truth labels
        - sensor data: rotRate.alpha/beta/gamma, alpha/beta/gamma, sqrt(X²+Y²+Z²)
    """
    # Get the data
    df = test.processed_data

    if 'mlproba_stats' not in test.quality.keys():
        test.quality['mlproba_stats'] = probability_quality(test)
    
    # Get ground truth start and end
    gt_indices = df[df['testBool'] == True]
    if len(gt_indices) > 0:
        gt_start = gt_indices['msFromStart'].values[0]
        gt_end = gt_indices['msFromStart'].values[-1]
        gtmltot = (gt_end - gt_start)/1000
    else:
        gt_start = None
        gt_end = None

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Prediction Probabilities
    ax1.plot(df['msFromStart'], df['predicted_testProba'],
             label='Predicted Probability', color='blue', linewidth=2)
    ax1.axhline(y=running_settings.parameters['classBinaryTresh'], color='gray', linestyle='--', linewidth=1, alpha=0.7, label=f"Threshold ({running_settings.parameters['classBinaryTresh']})")

    # Add ground truth region
    if gt_start is not None and gt_end is not None:
        ax1.axvline(x=gt_start, color='green', linestyle='-', linewidth=2, label='GT Start')
        ax1.axvline(x=gt_end, color='green', linestyle='-', linewidth=2, label='GT End')
        ax1.axvspan(gt_start, gt_end, alpha=0.2, color='green', label='GT Region')

    # Add estimation region
    if 'ml' in test.results.keys():
        ax1.axvline(x=test.results['ml']['t_start']*1000, color='red', linestyle='-', linewidth=2, label='Estimation Start')
        ax1.axvline(x=test.results['ml']['t_end']*1000, color='red', linestyle='-', linewidth=2, label='Estimation End')
        ax1.axvspan(test.results['ml']['t_start']*1000, test.results['ml']['t_end']*1000, alpha=0.2, color='red', label='Estimation Region')

    ax1.set_ylabel('Prediction Probability', fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    gtm = np.round(test.gt_total_manual,2) if test.gt_total_manual is not None else 0
    if error is not None: 
        ax1.set_title(f'Classification Predictions, test: {test.user_id}_{test.session_id}, '
                    f'error ml: {np.round(error, 2)}, gtgwalk: {np.round(test.gt_total_gwalk,2)}, '
                    f'gtmanual: {gtm}, '
                    f'gtml: {np.round(gtmltot,2)}', fontsize=14)
    else:
        ax1.set_title(f'Classification Predictions, test: {test.user_id}_{test.session_id}, '
                    f'error ml: {error}, gtgwalk: {np.round(test.gt_total_gwalk,2)}, '
                    f'gtmanual: {gtm}, '
                    f'gtml: {np.round(gtmltot,2)}', fontsize=14)

    # Plot 2: Binary Predictions
    ax2.plot(df['msFromStart'], df['predicted_testBool'],
             label='Predicted Class', color='blue', linewidth=2, drawstyle='steps-post')

    # Add ground truth region
    if gt_start is not None and gt_end is not None:
        ax2.axvline(x=gt_start, color='green', linestyle='-', linewidth=2, label='GT Start')
        ax2.axvline(x=gt_end, color='green', linestyle='-', linewidth=2, label='GT End')
        ax2.axvspan(gt_start, gt_end, alpha=0.2, color='green', label='GT Region')

    if 'ml' in test.results.keys():
        ax2.axvline(x=test.results['ml']['t_start']*1000, color='red', linestyle='-', linewidth=2, label='Estimation Start')
        ax2.axvline(x=test.results['ml']['t_end']*1000, color='red', linestyle='-', linewidth=2, label='Estimation End')
        ax2.axvspan(test.results['ml']['t_start']*1000, test.results['ml']['t_end']*1000, alpha=0.2, color='red', label='Estimation Region')

    ax2.set_ylabel('Predicted Class', fontsize=12)
    ax2.set_ylim(-0.1, 1.35)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Negative (0)', 'Positive (1)'])
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Sensor Data (Acceleration and Rotation)
    # Primary y-axis for acceleration
    ax3.plot(df['msFromStart'], df['sqrt(X²+Y²+Z²)'],
             label='Motion (m/s²)', color='blue', linestyle='-', linewidth=2)
    ax3.set_ylabel('Acceleration (m/s²)', color='blue', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='blue')

    # Secondary y-axis for rotation data
    ax3_right = ax3.twinx()

    # Orientation angles
    if 'alpha' in df.columns:
        ax3_right.plot(df['msFromStart'], df['alpha'],
                       label='Alpha (°)', color='red', linestyle='--', linewidth=2)
        ax3_right.plot(df['msFromStart'], df['beta'],
                       label='Beta (°)', color='green', linestyle='-.', linewidth=2)
        ax3_right.plot(df['msFromStart'], df['gamma'],
                       label='Gamma (°)', color='purple', linestyle=':', linewidth=2)

    # Rotation rates
    ax3_right.plot(df['msFromStart'], df['rotRate.alpha'],
                   label='RotRate Alpha (°/s)', color='darkred', linestyle='--', alpha=0.7)
    ax3_right.plot(df['msFromStart'], df['rotRate.beta'],
                   label='RotRate Beta (°/s)', color='darkgreen', linestyle='-.', alpha=0.7)
    ax3_right.plot(df['msFromStart'], df['rotRate.gamma'],
                   label='RotRate Gamma (°/s)', color='indigo', linestyle=':', alpha=0.7)

    ax3_right.set_ylabel('Rotation (°) / Rotation Rate (°/s)', fontsize=12)

    # Add ground truth region
    if gt_start is not None and gt_end is not None:
        ax3.axvline(x=gt_start, color='green', linestyle='-', linewidth=2.5)
        ax3.axvline(x=gt_end, color='green', linestyle='-', linewidth=2.5)
        ax3.axvspan(gt_start, gt_end, alpha=0.2, color='green')

    if 'ml' in test.results.keys():
        ax3.axvline(x=test.results['ml']['t_start']*1000, color='red', linestyle='-', linewidth=2, label='Estimation Start')
        ax3.axvline(x=test.results['ml']['t_end']*1000, color='red', linestyle='-', linewidth=2, label='Estimation End')
        ax3.axvspan(test.results['ml']['t_start']*1000, test.results['ml']['t_end']*1000, alpha=0.2, color='red', label='Estimation Region')

    ax3.set_xlabel('Time (ms from start)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    ax3_right.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    return fig


def iteratetimebtw(estimation, test, main_threshold_ms=15000, right_threshold_ms=5000, verbose=True):
    """
    Given a dataframe with 'msFromStart' and 'predicted_testBool' columns,
    intelligently identify the main test block and discard isolated intervals.

    Logic:
    1. Find the center (halfway point) of the test
    2. Select the closest True interval to the right of center as the MAIN block or the block that overlaps the center.
    3. Keep intervals to the left of main block if gap >= 15000 ms
    4. Keep intervals to the right of main block only if they start within 5000 ms of main block end

    Args:
        estimation: DataFrame with columns 'msFromStart' and 'predicted_testBool'
        main_threshold_ms: Threshold for intervals to the left of main block (default: 15000)
        right_threshold_ms: Threshold for intervals to the right of main block (default: 5000)
        verbose: Print detailed information about filtering decisions

    Returns:
        DataFrame with isolated True intervals removed (set to 0)
    """
    if 'parkapp' == test.dataset_id:
        print('parkapp')

    if verbose:
        fig = plot_ml_prediction(test, 0)

    result = estimation.copy()

    # Identify continuous True segments
    result['group'] = (result['predicted_testBool'] != result['predicted_testBool'].shift()).cumsum()

    # Get only True segments
    true_segments = result[result['predicted_testBool'] == 1].groupby('group').agg({
        'msFromStart': ['first', 'last', 'count']
    }).reset_index()

    true_segments.columns = ['group', 'start_time', 'end_time', 'count']

    # Segments that are shorter than XXms are ignored and discarded
    true_segments = true_segments[(true_segments['end_time'] - true_segments['start_time']) >= 100].copy()

    if len(true_segments) == 0:
        print("No true intervals above probability threshold.")
        t_start = result.iloc[0]['msFromStart']
        t_end = result.iloc[-1]['msFromStart']
        return t_start, t_end

    # Find the center (halfway point) of the entire test
    test_start = result['msFromStart'].min()
    test_end = result['msFromStart'].max()
    test_center = (test_start + test_end) / 2

    if verbose:
        print(f"\nTest duration: {test_start:.0f} - {test_end:.0f} ms")
        print(f"Test center: {test_center:.0f} ms")
        print(f"\nFound {len(true_segments)} True interval(s):")

    # Check if any interval overlaps the center
    overlapping_intervals = true_segments[
        (true_segments['start_time'] <= test_center) &
        (true_segments['end_time'] >= test_center)
    ].copy()

    if len(overlapping_intervals) > 0:
        # If there's an interval overlapping the center, select it as main block
        main_idx = overlapping_intervals.index[0]
        selection_method = "OVERLAPS CENTER"
    else:
        # Otherwise, find intervals to the right of center
        right_intervals = true_segments[true_segments['start_time'] >= test_center].copy()

        if len(right_intervals) == 0:
            if verbose:
                print("\nNo intervals found to the right of center!")
                print("Selecting the rightmost interval as main block...")
            # If no intervals to the right, select the last (rightmost) interval
            main_idx = true_segments['start_time'].idxmax()
            selection_method = "RIGHTMOST (fallback)"
        else:
            # Select the closest interval to the right of center (minimum start_time)
            main_idx = right_intervals['start_time'].idxmin()
            selection_method = "CLOSEST TO RIGHT OF CENTER"

    main_block = true_segments.loc[main_idx]
    main_group = main_block['group']

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"MAIN BLOCK identified: Group {main_group} ({selection_method})")
        print(f"  Time: {main_block['start_time']:.0f} - {main_block['end_time']:.0f} ms")
        print(f"  Duration: {main_block['end_time'] - main_block['start_time']:.0f} ms")
        if selection_method == "OVERLAPS CENTER":
            print(f"  ✓ This interval contains the test center ({test_center:.0f} ms)")
        print(f"{'=' * 80}\n")

    # Evaluate each interval
    true_segments['keep'] = False
    true_segments['reason'] = ''

    for idx, row in true_segments.iterrows():
        if row['group'] == main_group:
            # Always keep the main block
            true_segments.loc[idx, 'keep'] = True
            true_segments.loc[idx, 'reason'] = 'MAIN BLOCK'
        elif row['end_time'] < main_block['start_time']:
            # Interval to the LEFT of main block
            gap_to_main = main_block['start_time'] - row['end_time']
            if gap_to_main >= main_threshold_ms:
                true_segments.loc[idx, 'keep'] = False
                true_segments.loc[idx, 'reason'] = f'LEFT: gap to main = {gap_to_main:.0f} ms >= {main_threshold_ms} ms (DISCARD)'
            else:
                true_segments.loc[idx, 'keep'] = True
                true_segments.loc[
                    idx, 'reason'] = f'LEFT: gap to main = {gap_to_main:.0f} ms < {main_threshold_ms} ms'
        else:
            # Interval to the RIGHT of main block
            gap_from_main = row['start_time'] - main_block['end_time']
            if gap_from_main <= right_threshold_ms:
                true_segments.loc[idx, 'keep'] = True
                true_segments.loc[
                    idx, 'reason'] = f'RIGHT: gap from main = {gap_from_main:.0f} ms <= {right_threshold_ms} ms'
            else:
                true_segments.loc[idx, 'keep'] = False
                true_segments.loc[
                    idx, 'reason'] = f'RIGHT: gap from main = {gap_from_main:.0f} ms > {right_threshold_ms} ms (DISCARD)'

    if verbose:
        print("Interval Analysis:")
        print("-" * 80)
        for idx, row in true_segments.iterrows():
            status = "✓ KEEP" if row['keep'] else "✗ DISCARD"
            print(f"{status} - Interval at {row['start_time']:.0f} - {row['end_time']:.0f} ms")
            print(f"       {row['reason']}")
        print("-" * 80)

    # Keep only groups that are to keep
    groups_to_keep = true_segments[true_segments['keep']]['group'].values
    result['predicted_testBool'] = result['group'].isin(groups_to_keep).astype(int)

    # Return cleaned result
    final_result = result[['msFromStart', 'predicted_testBool']].copy()

    # Extract t_start and t_end from final result
    true_values = final_result[final_result['predicted_testBool'] == 1]

    if len(true_values) > 0:
        t_start = true_values['msFromStart'].iloc[0]
        t_end = true_values['msFromStart'].iloc[-1]
    else:
        t_start = None
        t_end = None

    if verbose:
        print(f"\nSummary:")
        print(f"  Original True predictions: {estimation['predicted_testBool'].sum()}")
        print(f"  Filtered True predictions: {final_result['predicted_testBool'].sum()}")
        print(f"  Intervals kept: {true_segments['keep'].sum()}/{len(true_segments)}")
        if t_start is not None and t_end is not None:
            ax1 = fig.axes[0]
            ax2 = fig.axes[1]
            ax3 = fig.axes[2]
            # Add dark green line for final predictions
            true_predictions = final_result[final_result['predicted_testBool'] == 1]
            if len(true_predictions) > 0:
                ax2.scatter(true_predictions['msFromStart'],
                            [1.2] * len(true_predictions),
                            color='darkgreen', s=10, alpha=0.6, label='Final Predictions', marker='|')
            ax1.axvline(x=t_start, color='red', linestyle='--', linewidth=2, label='Estimated Start')
            ax1.axvline(x=t_end, color='red', linestyle='--', linewidth=2, label='Estimated End')
            ax2.axvline(x=t_start, color='red', linestyle='--', linewidth=2, label='Estimated Start')
            ax2.axvline(x=t_end, color='red', linestyle='--', linewidth=2, label='Estimated End')
            ax3.axvline(x=t_start, color='red', linestyle='--', linewidth=2, label='Estimated Start')
            ax3.axvline(x=t_end, color='red', linestyle='--', linewidth=2, label='Estimated End')
            ax1.legend(loc='upper right')
            ax2.legend(loc='upper right')
            ax3.legend(loc='upper left')
            plt.show()
        else:
            print("No True predictions remain after filtering.")

    return t_start, t_end


def basic_approach(processed_data):
    estimation = processed_data[processed_data['predicted_testBool'] == True]
    if len(estimation) > 0:
        msStart = estimation['msFromStart'].values[0]
        msEnd = estimation['msFromStart'].values[-1]
    else:
        msStart = processed_data['msFromStart'].values[0]
        msEnd = processed_data['msFromStart'].values[-1]
    return msStart, msEnd


def compute_error(msStart, msEnd, test):
    gt_indices = test.processed_data[test.processed_data['testBool'] == True]
    if len(gt_indices) > 0:
        gt_start = gt_indices['msFromStart'].values[0]
        gt_end = gt_indices['msFromStart'].values[-1]
        gt = (gt_end - gt_start) / 1000

    duration = (msEnd - msStart) 
    if duration > 1000: 
        duration = duration / 1000
    error = duration - gt
    return error


def approach_duration_estimation(test, method=''):
    try:
        processed_data = test.processed_data
        print("Using basic approach for duration estimation")
        msStart, msEnd = basic_approach(processed_data)
        error = compute_error(msStart, msEnd, test)

        if abs(error) > 10:
            # plot_ml_prediction(test, error)
            verbose=True
        else:
            verbose=False

        if method == 'timebtwbool':
            print("Using time between peaks for duration estimation")
            estimation = processed_data[['msFromStart', 'predicted_testBool']]
            msStart, msEnd = iteratetimebtw(estimation, test, main_threshold_ms=9000, right_threshold_ms=5000, verbose=verbose)
            error = compute_error(msStart, msEnd, test)
        if msStart is not None and msEnd is not None:
            return msStart/1000, msEnd/1000
        else: 
            return None, None
    
    except:
        processed_data = test.processed_data
        print(f"Issues with normal approaches, taking first and last raw sample. {test.user_id + '_' + str(test.session_id)}")
        print(msStart, msEnd)
        return processed_data['msFromStart'].values[0]/1000, processed_data['msFromStart'].values[-1]/1000


def evaluate_duration_tests(tests_original, method):
    for test_id, test in tests_original.items():
        print(f"\nEvaluating duration for Test ID: {test_id}")
        if test.user_id + '_' + str(test.session_id) == '2_parkapp_8' or test.user_id + '_' + str(test.session_id) == '4_parkapp_8' or test.user_id + '_' + str(test.session_id) == '12_parkapp_1':
            print("to check better here")

        if test.processed_data.shape[0] > 0:
            msStart, msEnd = approach_duration_estimation(test, method='timebtwbool')

            test.results[method] = {'t_start': msStart, 't_end': msEnd}
            error = compute_error(msStart, msEnd, test)
            test.error[method] = error

            if abs(error) > 20:
                print("Error > 20")
                plot_ml_prediction(test, error)
        else:
            test.results[method] = None


    return tests_original


def observe_performance_per_test(original_tests_fold, holdout_tests_original, method, modelname):
    # Observe performances across tests
    if len(holdout_tests_original) > 0:
        holdout_tests_original = evaluate_duration_tests(holdout_tests_original, method)

        utils_evaluation.evaluate_results(list(holdout_tests_original.values()), eval_type='duration',
                                          method=method, gttype='gwalk',
                                          dataset='holdout', title=modelname + '_' + method, logging=True)

        all_holdout_remained = utils_dataquality.plot_tests_witherror(holdout_tests_original.values(), error_threshold=4, method='ml')

    if len(original_tests_fold) > 0:
        all_folds_tests = []
        for f in original_tests_fold.keys():
            val_tests_original = original_tests_fold[f]
            val_tests_original = evaluate_duration_tests(val_tests_original, method)
            all_folds_tests.extend(list(val_tests_original.values()))

        all_fold_remained = utils_dataquality.plot_tests_witherror(all_folds_tests, error_threshold=30, method='ml')
        utils_evaluation.evaluate_results(all_folds_tests, eval_type='duration',
                                          method=method, gttype='gwalk',
                                          dataset='cvfolds', title=modelname + '_' + method, logging=True)

        return all_folds_tests


def lopo_validation(all_tests):
    # Implement leave one participant out validation based on test.user_id for each element of all_tests
    from sklearn.model_selection import LeaveOneGroupOut

    # Extract user_id from each test
    user_ids = np.array([test.user_id for test in all_tests])
    unique_users = np.unique(user_ids)
    n_users = len(unique_users)

    print(f"Total tests: {len(all_tests)}")
    print(f"Unique participants: {n_users}")
    print(f"Tests per participant: {np.bincount([list(unique_users).index(uid) for uid in user_ids])}")

    # Hold out 15% of PARTICIPANTS for final evaluation
    from sklearn.model_selection import train_test_split

    train_val_users, holdout_users = train_test_split(
        unique_users,
        test_size=0.15,
        random_state=42,
        shuffle=True
    )

    # Get test indices for train/val and holdout
    train_val_mask = np.isin(user_ids, train_val_users)
    holdout_mask = np.isin(user_ids, holdout_users)

    train_val_tests = np.where(train_val_mask)[0]
    holdout_tests = np.where(holdout_mask)[0]

    # Store holdout tests
    holdout_tests_original = {i: test for i, test in enumerate(all_tests) if i in holdout_tests}

    print(f"Train+Val participants: {len(train_val_users)}, Holdout participants: {len(holdout_users)}")
    print(f"Train+Val tests: {len(train_val_tests)}, Holdout tests: {len(holdout_tests)}")

    # Filter user_ids to only include train_val users
    train_val_user_ids = user_ids[train_val_mask]

    # Create custom splits generator for LOPO
    def lopo_splits():
        """Generate LOPO splits for train/val participants"""
        for user in train_val_users:
            # Train: all participants except current one
            # Val: current participant
            train_mask = train_val_user_ids != user
            val_mask = train_val_user_ids == user

            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]

            yield train_idx, val_idx

    # Override kf with our custom generator
    split_loop = lopo_splits()
    n_splits = len(train_val_users)  # Update n_splits for loop iteration

    return split_loop, n_splits, holdout_tests_original, holdout_tests

def kfold_validation(all_tests, n_splits, holdout=False):
    n_tests = len(all_tests)
    test_indices = np.arange(n_tests)

    # Hold out 15% of tests for final evaluation (never seen during CV)
    from sklearn.model_selection import train_test_split

    unique_tests = np.unique(test_indices)

    if not holdout:
        if False:
            train_val_tests, holdout_tests = train_test_split(
                unique_tests,
                test_size=0, #0.15
                random_state=42,
                shuffle=True
            )
        else:
            holdout_tests = []
            train_val_tests = unique_tests
        holdout_tests_original = {i: test for i, test in enumerate(all_tests) if i in holdout_tests}
    else:
        holdout_tests = test_indices
        holdout_tests_original = {i: test for i, test in enumerate(all_tests) if i in holdout_tests}
        train_val_tests = []

    print(f"Train+Val tests: {len(train_val_tests)}, Holdout tests: {len(holdout_tests)}")
    if len(train_val_tests)>0:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_loop = kf.split(train_val_tests)
    else:
        split_loop = None

    return split_loop, holdout_tests_original, holdout_tests

def kfold_validation_equalsplit(all_tests, n_splits):
    n_splits=5

    # Group tests by dataset_id
    train_val_groups = {}
    for idx, test in enumerate(all_tests):
        dataset_id = test.dataset_id
        if dataset_id not in train_val_groups:
            train_val_groups[dataset_id] = []
        train_val_groups[dataset_id].append(idx)

    n_datasets = len(train_val_groups)
    print(f"\n{'=' * 60}")
    print(f"Train+Val Tests Distribution:")
    print(f"{'=' * 60}")
    for dataset_id, indices in train_val_groups.items():
        print(f"Dataset {dataset_id}: {len(indices)} tests")
    print(f"Total datasets: {n_datasets}")
    print(f"Total train+val tests: {len(all_tests)}")

    # Create stratified splits
    folds = [[] for _ in range(n_splits)]
    for dataset_id, indices in train_val_groups.items():
        indices = np.array(indices)
        np.random.seed(42)
        np.random.shuffle(indices)

        # Distribute tests from this dataset across folds as evenly as possible
        for i, idx in enumerate(indices):
            fold_num = i % n_splits
            folds[fold_num].append(idx)

    # Shuffle each fold to mix datasets
    np.random.seed(42)
    for fold in folds:
        np.random.shuffle(fold)

    # Print distribution for each fold
    for fold_num, fold_indices in enumerate(folds):
        fold_tests = [all_tests[i] for i in fold_indices]
        fold_dataset_counts = Counter([test.dataset_id for test in fold_tests])
        print(f"\nFold {fold_num + 1}:")
        for dataset_id in sorted(train_val_groups.keys()):
            count = fold_dataset_counts.get(dataset_id, 0)
            print(f"  Dataset {dataset_id}: {count} tests")
        print(f"  Total: {len(fold_indices)} tests")

    # Create generator for splits
    def split_generator():
        for val_fold_num in range(n_splits):
            val_indices = np.array(folds[val_fold_num])
            train_indices = np.array([idx for i, fold in enumerate(folds)
                                      if i != val_fold_num for idx in fold])

            print(f"\n{'=' * 60}")
            print(f"Current Split: Fold {val_fold_num + 1} as Validation")
            print(f"{'=' * 60}")

            train_tests = [all_tests[i] for i in train_indices]
            val_tests = [all_tests[i] for i in val_indices]

            train_dataset_counts = Counter([test.dataset_id for test in train_tests])
            val_dataset_counts = Counter([test.dataset_id for test in val_tests])

            print(f"Training set:")
            for dataset_id in sorted(train_val_groups.keys()):
                count = train_dataset_counts.get(dataset_id, 0)
                print(f"  Dataset {dataset_id}: {count} tests")
            print(f"  Total: {len(train_indices)} tests")

            print(f"Validation set:")
            for dataset_id in sorted(train_val_groups.keys()):
                count = val_dataset_counts.get(dataset_id, 0)
                print(f"  Dataset {dataset_id}: {count} tests")
            print(f"  Total: {len(val_indices)} tests")

            yield train_indices, val_indices

    split_loop = split_generator()

    return split_loop


def verify_stratification(all_tests, split_loop, n_splits=5):
    """
    Verify that the stratification worked correctly.
    """
    print(f"\n{'=' * 60}")
    print(f"VERIFICATION: Checking Dataset Distribution Across Folds")
    print(f"{'=' * 60}")

    # Count dataset distribution in full train_val set
    full_dataset_counts = Counter([test.dataset_id for test in all_tests])
    expected_per_fold = {ds: count / n_splits for ds, count in full_dataset_counts.items()}

    print(f"\nExpected tests per fold (average):")
    for dataset_id, count in sorted(expected_per_fold.items()):
        print(f"  Dataset {dataset_id}: {count:.1f} tests")

    # Check each fold
    fold_num = 1
    max_deviation = {}

    for train_idx, val_idx in split_loop:
        val_tests = [all_tests[i] for i in val_idx]
        val_dataset_counts = Counter([test.dataset_id for test in val_tests])

        print(f"\nFold {fold_num} validation set:")
        for dataset_id in sorted(full_dataset_counts.keys()):
            actual = val_dataset_counts.get(dataset_id, 0)
            expected = expected_per_fold[dataset_id]
            deviation = abs(actual - expected)

            if dataset_id not in max_deviation:
                max_deviation[dataset_id] = 0
            max_deviation[dataset_id] = max(max_deviation[dataset_id], deviation)

            print(f"  Dataset {dataset_id}: {actual} tests (expected: {expected:.1f}, deviation: {deviation:.1f})")

        fold_num += 1

    print(f"\n{'=' * 60}")
    print(f"Maximum Deviation from Expected:")
    print(f"{'=' * 60}")
    for dataset_id, deviation in sorted(max_deviation.items()):
        print(f"Dataset {dataset_id}: {deviation:.1f} tests")

    if all(dev <= 1 for dev in max_deviation.values()):
        print(f"\n✓ Stratification is well-balanced (max deviation ≤ 1 test)")
    else:
        print(f"\n⚠ Some imbalance detected (this is normal for small datasets)")

def apply_data_augmentation(X_train, y_train, X_val, y_val):
    return X_train, y_train, X_val, y_val

def load_pretrained_model(model_name, best_fold_idx):
    print(f"\n{'=' * 60}")
    print(f"Loading model at {best_fold_idx} fold")

    fold_model_name = f"{model_name.replace('.h5', '')}_fold{best_fold_idx}.h5"
    modelObj = classes.MlModel(model_name=fold_model_name)
    modelObj.load_model()
    return modelObj

def save_bestscaler(best_scaler, title):
    scaler_filename = running_settings.models_path + os.sep + title+'_scaler.pickle'
    pickle.dump(best_scaler, open(scaler_filename, "wb"))

def load_bestscaler(title):
    scaler_filename = running_settings.models_path + os.sep + title+'_scaler.pickle'
    best_scaler = pickle.load(open(scaler_filename, "rb"))
    return best_scaler

def fittingmodel(fold, n_splits, X_train, y_train, X, X_val, y_val, model_name, modelcomments,
                 architecture, training_epochs, save_model, output_steps):
    print(f"\n{'=' * 60}")
    print(f"Training Fold {fold + 1}/{n_splits}")
    print(f"{'=' * 60}")

    fold_model_name = f"{model_name.replace('.h5', '')}_fold{fold + 1}.h5"
    modelObj = classes.MlModel(model_name=fold_model_name)
    modelObj.define_model(save_model=save_model, n_features=X.shape[2], architecture=architecture, output_steps=output_steps)
    modelObj.model_fit(
        X_train, y_train, X_val, y_val,
        plot=False,
        epochs=training_epochs,
        information=f"{modelcomments} - Fold {fold + 1}",
        save_model=save_model)
    return modelObj

def ML_pipeline(all_tests, model_name="best_model.h5", use_cv=True,
                n_splits=5, architecture='', training_epochs=5, save_model=False, 
                input_type='triaxial', method='ml', output_steps=0, 
                load_existing = False, evaluation = True):
    """
    Train ML model with optional cross-validation.

    Args:
        all_tests: TUG classes objects with processed_data containing sensor data
        model_name: Name for saving the model
        use_cv: Whether to use cross-validation
        n_splits: Number of folds for cross-validation

    Returns:
        Trained model object (or list of models if using CV)
    """

    modelcomments = running_settings.model_comments

    X, y, test_index = prep_data(all_tests, stride=running_settings.parameters['stride'], input_type=input_type, output_steps=output_steps)

    if use_cv:
        if isinstance(n_splits, int):
            split_loop, holdout_tests_original, holdout_tests = kfold_validation(all_tests, n_splits=n_splits)

        elif n_splits == 'equalcvsplit':
            split_loop = kfold_validation_equalsplit(all_tests, n_splits=5)
            holdout_tests_original, holdout_tests = [], []

        elif n_splits == 'lopo':
            split_loop, n_splits, holdout_tests_original, holdout_tests = lopo_validation(all_tests)

        cv_results = []
        cv_results_per_test_all = []
        cv_test_level_metrics = []
        original_tests_fold = {}
        fold_models = []
        scalers = []
        best_fold_idx = 0
        best_val_f1 = 0

        for fold, (train_fold_idx, val_fold_idx) in enumerate(split_loop):
            val_tests = val_fold_idx
            val_tests_original = {i: test for i, test in enumerate(all_tests) if i in val_tests}

            X_train = X[np.isin(test_index, train_fold_idx)]
            y_train = y[np.isin(test_index, train_fold_idx)]

            scaler = StandardScaler()
            X_train_flat = X_train.reshape(-1, X_train.shape[-1])
            X_train_scaled = scaler.fit_transform(X_train_flat)
            X_train = X_train_scaled.reshape(X_train.shape)
            scalers.append(scaler)

            X_val = X[np.isin(test_index, val_fold_idx)]
            y_val = y[np.isin(test_index, val_fold_idx)]

            if X_val.shape[0]>0:
                X_val_flat = X_val.reshape(-1, X_val.shape[-1])
                X_val_scaled = scaler.transform(X_val_flat)
                X_val = X_val_scaled.reshape(X_val.shape)

                val_test_index = test_index[np.isin(test_index, val_fold_idx)]

                X_train, y_train, X_val, y_val = apply_data_augmentation(X_train, y_train, X_val, y_val)

                # Define and train model
                if not load_existing:
                    modelObj = fittingmodel(fold, n_splits, X_train, y_train, X, X_val, y_val, model_name, modelcomments,
                 architecture, training_epochs, save_model, output_steps)
                else:
                    print(f"\n{'=' * 60}")
                    print(f"Loading model at {fold + 1}/{n_splits}")
                    modelObj = load_pretrained_model(model_name, fold+1)

                # Sample-level evaluation
                cv_results, fold_models, best_val_f1, best_fold_idx = evaluate_cv(
                    modelObj, X_val, y_val, fold, cv_results, fold_models, best_val_f1, best_fold_idx
                )
                cv_results_per_test_all.append(test_results_df)

                if evaluation:
                    # Test-level evaluation
                    test_results_df, test_level_metrics, val_tests_original = evaluate_cv_per_test(
                        modelObj, X_val, y_val, val_test_index, val_tests, fold, val_tests_original, output_steps=output_steps
                    )
                    original_tests_fold[fold] = val_tests_original
                    cv_test_level_metrics.append(test_level_metrics)

            else:
                print("No data in this batch")
                continue

        if not load_existing:
            utils_plots.plot_all_training_history(fold_models, title=model_name.strip('.h5') + '_allfoldsresults.jpg')

        # Print CV summary - Sample Level
        if evaluation: 
            print(f"\n{'=' * 60}")
            print("Cross-Validation Summary - SAMPLE LEVEL:")
            print(f"{'=' * 60}")
            cv_df = pd.DataFrame(cv_results)
            for metric in ['val_accuracy', 'val_precision', 'val_recall', 'val_f1_score']:
                mean_val = cv_df[metric].mean()
                std_val = cv_df[metric].std()
                print(f"{metric.replace('val_', '').replace('_', ' ').title():15s}: {mean_val:.4f} ± {std_val:.4f}")

            # Print CV summary - Test Level
            print(f"\n{'=' * 60}")
            print("Cross-Validation Summary - TEST LEVEL (averaged per test):")
            print(f"{'=' * 60}")
            cv_test_df = pd.DataFrame(cv_test_level_metrics)
            for metric in ['mean_test_accuracy', 'mean_test_precision', 'mean_test_recall', 'mean_test_f1']:
                mean_val = cv_test_df[metric].mean()
                std_val = cv_test_df[metric].std()
                print(f"{metric.replace('mean_test_', '').replace('_', ' ').title():15s}: {mean_val:.4f} ± {std_val:.4f}")

        # Save results
        if False:
            cv_df.to_csv(running_settings.results_all + os.sep + 'cv_results_sample_level.csv', index=False)
            cv_test_df.to_csv(running_settings.results_all + os.sep + 'cv_results_test_level.csv', index=False)

            # Save all per-test results
            all_test_results = pd.concat(cv_results_per_test_all, ignore_index=True)
            all_test_results.to_csv(running_settings.results_all + os.sep + 'cv_per_test_detailed.csv', index=False)

        best_model = fold_models[best_fold_idx]
        best_scaler = scalers[best_fold_idx]
        # Save best_scaler: 
        save_bestscaler(best_scaler)

        

        if len(holdout_tests)>0:
            # Evaluate on holdout set
            test_indeces = np.arange(len(all_tests))

            # Evaluate on holdout set
            holdout_results = evaluate_holdout(best_model, best_scaler,
                                               X, y, test_indeces, test_index, holdout_tests, save=False)

            # modelObj, X_val, y_val, val_test_index, val_tests, fold, val_tests_original
            holdout_results_df, holdout_level_metrics, holdout_tests_original = evaluate_holdout_per_test(best_model,
                                                                                                          best_scaler, X, y,
                                                                                                          test_index,
                                                                                                          holdout_tests,
                                                                                                          best_fold_idx,
                                                                                                          holdout_tests_original)

        all_folds_tests = observe_performance_per_test(original_tests_fold, holdout_tests_original, method=method, modelname=model_name.strip('.h5'))

        return best_model, best_scaler, best_fold_idx, all_folds_tests

    else:
        # Single model training (no CV) or loading existing model
        if not load_existing:
            # Train on all data
            X_train, X_val, y_train, y_val, X_holdout, y_holdout = split_data(X, y, test_index, fold_idx=None)

            modelObj = classes.MlModel(model_name=model_name)
            modelObj.define_model()
            modelObj.model_fit(X_train, y_train, X_val, y_val, plot=True, information=modelcomments)
            modelObj.save_model(title=modelObj.model_name)
        else:
            modelObj = classes.MlModel(model_name=model_name)
            modelObj.load_model(title=modelObj.model_name)

        return modelObj

def investigate_probastats_error(all_fold_tests):

    """
    # Relate the statistics obtained from mlproba_stats to the error in test.error['ml']
    # Find a way that we can use the predicted probabilities as a quality metric. Don't make uip random scores, use statisitcs and a deterministic approach.
    # Use correlations and scatter plots to identify relationships.

    Args:
        all_fold_tests:

    Returns:

    """

    qualityproba_all = utils_dataquality.get_stats_all(all_fold_tests, method='mlproba_stats')
    error_all, indiv_error_duration = utils_dataquality.compute_error_tests(all_fold_tests, method='ml')

    df = utils_dataquality.prepare_quality_error_dataframe1(qualityproba_all, error_all)


    fig1, df = utils_dataquality.plot_error_by_quality(df, proba=True)

    pass
def investigate_predictedproba(all_fold_tests):
    for test in all_fold_tests:
        test.quality['mlproba_stats'] = probability_quality(test)

    investigate_probastats_error(all_fold_tests)
    return None

def probability_quality(test):
    """
    Compute deterministic, statistically meaningful quality metrics based on
    predicted probabilities in test.processed_data['predicted_testProba'].
    Returns a dictionary.
    """

    df = test.processed_data
    proba = df['predicted_testProba'].values

    # Safety filtering
    if len(proba) == 0:
        return {"error": "empty_signal"}

    stats = {}

    # --- (1) Probability distribution overall ---
    stats["mean"] = float(np.mean(proba))
    stats["median"] = float(np.median(proba))
    stats["std"] = float(np.std(proba))
    stats["min"] = float(np.min(proba))
    stats["max"] = float(np.max(proba))
    stats["autocorr_180lag"] = float(pd.Series(proba).autocorr(lag=180))
    stats["autocorr_60lag"] = float(pd.Series(proba).autocorr(lag=60))
    stats["autocorr_30lag"] = float(pd.Series(proba).autocorr(lag=30))

    # --- (3) Probability entropy (higher = worse) ---
    # Convert proba to binary distribution [p, 1-p]
    eps = 1e-12
    entropy_vals = entropy(np.vstack([proba + eps, 1 - proba + eps]).T, base=2, axis=1)
    stats["mean_entropy"] = float(np.mean(entropy_vals))

    # --- (6) Smoothness of probability time series ---
    # Higher variance of derivative = noisier model
    deriv = np.diff(proba)
    stats["derivative_std"] = float(np.std(deriv))

    return stats


def holdout_external_testing(model, scaler, best_fold, input_type, output_steps, all_tests_matey):

    modelcomments = running_settings.model_comments

    X, y, test_index = prep_data(all_tests_matey, stride=running_settings.parameters['stride'],
                                 input_type=input_type, output_steps=output_steps)

    split_loop, holdout_tests_original, holdout_tests = kfold_validation(all_tests_matey, n_splits=None, holdout=True)

    if len(holdout_tests) > 0:
        # Evaluate on holdout set
        test_indeces = np.arange(len(all_tests_matey))

        # Evaluate on holdout set
        holdout_results = evaluate_holdout(model, scaler, X, y, test_indeces,
                                                        test_index, holdout_tests, save=False)

        # modelObj, X_val, y_val, val_test_index, val_tests, fold, val_tests_original
        holdout_results_df, holdout_level_metrics, holdout_tests_original = evaluate_holdout_per_test(model, scaler, X, y,
                                                                                                      test_index,
                                                                                                      holdout_tests,
                                                                                                      best_fold,
                                                                                                      holdout_tests_original,
                                                                                                      dataset='matey_sanz')


        all_folds_tests = observe_performance_per_test([], holdout_tests_original,
                                                       method='ml',
                                                       modelname=model.model_name.strip('.h5'))

    return None