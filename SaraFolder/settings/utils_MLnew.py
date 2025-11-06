import os
from collections import Counter

import pandas as pd
import numpy as np
import scipy.signal as signal

import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

matplotlib.use('TkAgg')
from SaraFolder.settings import classes, utils_darioalgo, running_settings, utils_evaluation, utils_plots
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


def evaluate_holdout(best_fold_idx, best_scaler, fold_models, X, y, test_indices, test_index, holdout_tests,
                     save=False):
    print(f"\n{'=' * 60}")
    print(f"HOLDOUT SET EVALUATION (Best Model: Fold {best_fold_idx + 1})")
    print(f"{'=' * 60}")

    best_model = fold_models[best_fold_idx]

    X_holdout = X[np.isin(test_index, holdout_tests)]
    y_holdout = y[np.isin(test_index, holdout_tests)]

    X_holdout_flat = X_holdout.reshape(-1, X_holdout.shape[-1])
    X_holdout_scaled = best_scaler.fit_transform(X_holdout_flat)
    X_holdout = X_holdout_scaled.reshape(X_holdout.shape)

    print(f"Evaluation with X_holdout shaped: {X_holdout.shape}")

    # Predict on holdout
    y_holdout_pred = best_model.fitted_model.predict(X_holdout, verbose=2)
    y_holdout_pred_binary = (y_holdout_pred > 0.5).astype(int)

    y_holdout_flat = y_holdout.reshape(-1)
    y_holdout_pred_flat = y_holdout_pred_binary.reshape(-1)

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
        'best_fold': best_fold_idx + 1,
        'holdout_accuracy': holdout_acc,
        'holdout_precision': holdout_precision,
        'holdout_recall': holdout_recall,
        'holdout_f1': holdout_f1,
        'n_holdout_tests': len(holdout_tests)
    }])

    if save:
        holdout_results.to_csv(running_settings.results_all + os.sep + 'holdout_results.csv', index=False)

    return fold_models, holdout_results


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
            reconstructed[start_idx:valid_end] += predictions[window_idx, :valid_window_size].flatten()

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


def evaluate_holdout_per_test(modelObj, best_scaler, X_val, y_val, val_test_index, val_tests, fold, val_tests_original, output_steps=0):
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
        y_pred_binary_reconstructed = (y_pred_reconstructed > 0.5).astype(int)
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
                stride=running_settings.parameters['stride'],  # Your stride
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
        y_pred_binary_reconstructed = (y_pred_reconstructed > 0.5).astype(int)
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
    y_pred_binary = (y_pred > 0.5).astype(int)

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


def evaluate_duration_tests(holdout_tests_original, method):
    for test_id, test in holdout_tests_original.items():
        print(f"\nEvaluating duration for Test ID: {test_id}")
        gt = test.gt_total_gwalk
        if not gt:
            gt = test.gt_total_manual
        try:
            if test.processed_data.shape[0] > 0:
                estimation = test.processed_data[test.processed_data['predicted_testBool'] == True]
                msStart = estimation['msFromStart'].values[0]
                msEnd = estimation['msFromStart'].values[-1]

                test.results[method] = {'t_start': msStart / 1000, 't_end': msEnd / 1000}
            else:
                test.results[method] = None
        except:
            print(1)

    return holdout_tests_original


def observe_performance_per_test(original_tests_fold, holdout_tests_original, method, modelname):
    # Observe performances across tests
    if len(holdout_tests_original)>0:
        holdout_tests_original = evaluate_duration_tests(holdout_tests_original, method)

        utils_evaluation.evaluate_results(list(holdout_tests_original.values()), eval_type='duration',
                                          method=method, gttype='gwalk',
                                          dataset='holdout', title=modelname + '_' + method, logging=True)

    all_folds_tests = []
    for f in original_tests_fold.keys():
        val_tests_original = original_tests_fold[f]
        val_tests_original = evaluate_duration_tests(val_tests_original, method)
        all_folds_tests.extend(list(val_tests_original.values()))

    utils_evaluation.evaluate_results(all_folds_tests, eval_type='duration',
                                      method=method, gttype='gwalk',
                                      dataset='cvfolds', title=modelname + '_' + method, logging=True)

    return 0


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

def kfold_validation(all_tests, n_splits):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_tests = len(all_tests)
    test_indices = np.arange(n_tests)

    # Hold out 15% of tests for final evaluation (never seen during CV)
    from sklearn.model_selection import train_test_split

    unique_tests = np.unique(test_indices)

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

    print(f"Train+Val tests: {len(train_val_tests)}, Holdout tests: {len(holdout_tests)}")
    split_loop = kf.split(train_val_tests)

    return split_loop, holdout_tests_original, holdout_tests

def kfold_validation_equalsplit(all_tests, n_splits):
    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_tests = len(all_tests)
    # test_indices = np.arange(n_tests)
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

            # Show dataset distribution for this split
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


def ML_pipeline(all_tests, model_name="best_model.h5", use_cv=True,
                n_splits=5, architecture='', training_epochs=5, save_model=False, input_type='triaxial', method='', output_steps=0):
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

    load_existing = running_settings.load_existing_model
    modelcomments = running_settings.model_comments

    X, y, test_index = prep_data(all_tests, stride=running_settings.parameters['stride'], input_type=input_type, output_steps=output_steps)

    if use_cv and not load_existing:
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
        best_fold_idx = None
        best_val_f1 = 0

        for fold, (train_fold_idx, val_fold_idx) in enumerate(split_loop):
            print(f"\n{'=' * 60}")
            print(f"Training Fold {fold + 1}/{n_splits}")
            print(f"{'=' * 60}")

            val_tests = val_fold_idx
            val_tests_original = {i: test for i, test in enumerate(all_tests) if i in val_tests}

            X_train = X[np.isin(test_index, train_fold_idx)]
            y_train = y[np.isin(test_index, train_fold_idx)]

            try:
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
                    fold_model_name = f"{model_name.replace('.h5', '')}_fold{fold + 1}.h5"
                    modelObj = classes.MlModel(model_name=fold_model_name)
                    modelObj.define_model(save_model=save_model, n_features=X.shape[2], architecture=architecture, output_steps=output_steps)
                    modelObj.model_fit(
                        X_train, y_train, X_val, y_val,
                        plot=False,
                        epochs=training_epochs,
                        information=f"{modelcomments} - Fold {fold + 1}",
                        save_model=False
                    )

                    # Sample-level evaluation
                    cv_results, fold_models, best_val_f1, best_fold_idx = evaluate_cv(
                        modelObj, X_val, y_val, fold, cv_results, fold_models, best_val_f1, best_fold_idx
                    )

                    # Test-level evaluation
                    test_results_df, test_level_metrics, val_tests_original = evaluate_cv_per_test(
                        modelObj, X_val, y_val, val_test_index, val_tests, fold, val_tests_original, output_steps=output_steps
                    )
                    original_tests_fold[fold] = val_tests_original

                    cv_results_per_test_all.append(test_results_df)
                    cv_test_level_metrics.append(test_level_metrics)
                else:
                    print("No data in this batch")
                    continue
            except:
                print("bug here")

        utils_plots.plot_all_training_history(fold_models, title=model_name.strip('.h5') + '_allfoldsresults.jpg')

        # Print CV summary - Sample Level
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

        # Evaluate on holdout set
        best_model = fold_models[best_fold_idx]
        best_scaler = scalers[best_fold_idx]

        test_indeces = np.arange(len(all_tests))

        if len(holdout_tests)>0:
            # Evaluate on holdout set
            fold_models, holdout_results = evaluate_holdout(best_fold_idx, best_scaler, fold_models, X, y, test_indeces,
                                                            test_index, holdout_tests, save=False)
            # modelObj, X_val, y_val, val_test_index, val_tests, fold, val_tests_original
            holdout_results_df, holdout_level_metrics, holdout_tests_original = evaluate_holdout_per_test(best_model,
                                                                                                          best_scaler, X, y,
                                                                                                          test_index,
                                                                                                          holdout_tests,
                                                                                                          best_fold_idx,
                                                                                                          holdout_tests_original)

        observe_performance_per_test(original_tests_fold, holdout_tests_original, method=method.strip('.h5'), modelname=model_name.strip('.h5'))
        return best_model

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
