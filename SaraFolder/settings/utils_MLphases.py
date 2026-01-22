import os
from collections import Counter
import pickle

import pandas as pd
import numpy as np
import scipy.signal as signal
from scipy.stats import entropy

import matplotlib
from matplotlib import pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from SaraFolder.settings import classes, utils_MLnew, utils_darioalgo, running_settings, utils_evaluation, utils_plots, \
    utils_dataquality
from SaraFolder.settings.utils_parkaapp import utils_parkapp

###################################################################################################

PHASE_MAP = {
    "totalDuration": ("msEnd", "msStart"),
    "Sit-to-stand": ("msWalking1", "msStart"),
    "Walking1": ("msTurn1", "msWalking1"),
    "Turn1": ("msWalking2", "msTurn1"),
    "Walking2": ("msTurn2", "msWalking2"),
    "Turn2+Stand-to-sit": ("msEnd", "msTurn2"),
}

PHASE_LABELS = {
    '0:No test': 'No test',
    '1:Sit-to-stand': 'Sit-to-stand',
    '2:Turn1': 'Turn1',
    '3:Turn2+Stand-to-sit': 'Turn2+Stand-to-sit',
    '4:Walking1': 'Walking1',
    '5:Walking2': 'Walking2',
}

PHASE_ORDER = [
    'No test',
    'Sit-to-stand',
    'Walking1',
    'Turn1',
    'Walking2',
    'Turn2+Stand-to-sit',
]

class_names = {0: 'No test', 1: 'Sit-to-stand', 2: 'Turn1', 3: 'Turn2+Stand-to-sit', 4: 'Walking1', 5:'Walking2'}

###################################################################################################

def evaluation_level(title, search_terms, cv_results):
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    cv_df = pd.DataFrame(cv_results)
    for metric in search_terms:
        mean_val = cv_df[metric].mean()
        std_val = cv_df[metric].std()
        print(f"{metric.replace('val_', '').replace('_', ' ').title():15s}: {mean_val:.4f} ± {std_val:.4f}")

def ML_pipeline_phases(all_tests, model_name="mdl_strongbs_sixax.h5", # let's keep the bigger kernels!
                        architecture='strongbs', # strongbs, 'cnn_bilstm', # strongbs # bs_predictbatch, 'tcn'
                        use_cv=True,
                        n_splits='equalcvsplit',  # lopo # equalcvsplit # int number # type: ignore
                        training_epochs=30,
                        save_model=True,
                        input_type='sixaxial', # or 'magnitude_acc' # triaxial
                        output_steps=0,
                        load_existing=True, 
                        evaluation = False, 
                        method='ml'
                        ):
    """Train ML model with optional cross-validation.

    Args:
        all_tests: TUG classes objects with processed_data containing sensor data
        model_name: Name for saving the model
        use_cv: Whether to use cross-validation
        n_splits: Number of folds for cross-validation

    Returns:
        Trained model object (or list of models if using CV)
    """

    modelcomments = running_settings.model_comments

    X, y, test_index = prep_data_phases(all_tests, stride=running_settings.parameters['stride'], input_type=input_type, output_steps=output_steps)
    n_classes = len(np.unique(y))
    if use_cv:
        holdout_tests_original, holdout_tests, split_loop = utils_MLnew.defining_fold(all_tests, n_splits)

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

            X_train, scaler = utils_MLnew.scale_data(X_train, StandardScaler(), fit=True)
            scalers.append(scaler)

            X_val = X[np.isin(test_index, val_fold_idx)]
            y_val = y[np.isin(test_index, val_fold_idx)]

            if X_val.shape[0]>0:
                X_val, scaler = utils_MLnew.scale_data(X_val, scaler=scaler, fit=False)


                val_test_index = test_index[np.isin(test_index, val_fold_idx)]

                # X_train, y_train, X_val, y_val = apply_data_augmentation(X_train, y_train, X_val, y_val)

                # Define and train model
                if not load_existing:
                    modelObj = fittingmodel_phases(fold, n_splits, X_train, y_train, X, X_val, y_val, model_name, modelcomments,
                    architecture, training_epochs, save_model, output_steps, n_classes=n_classes)
                else:
                    print(f"\n{'=' * 60}")
                    print(f"Loading model at {fold + 1}/{n_splits}")
                    modelObj = load_pretrained_model(model_name, fold+1)

                # Sample-level evaluation - OKAY
                cv_results, fold_models, best_val_f1, best_fold_idx = evaluate_cv_phases(
                    modelObj, X_val, y_val, fold, cv_results, fold_models, best_val_f1, best_fold_idx, num_classes=n_classes
                )

                if evaluation:
                    # Test-level evaluation
                    test_results_df, test_level_metrics, val_tests_original = evaluate_cv_per_test_phases(
                        modelObj, X_val, y_val, val_test_index, val_tests, fold, val_tests_original, output_steps=output_steps, num_classes=n_classes
                    )
                    cv_results_per_test_all.append(test_results_df)
                    cv_test_level_metrics.append(test_level_metrics)
                    original_tests_fold[fold] = val_tests_original
            else:
                print("No data in this batch")
                continue

        if not load_existing and 'crf' not in architecture:
            utils_plots.plot_all_training_history(fold_models, title=model_name.strip('.h5') + '_allfoldsresults.jpg')

        if evaluation: 
            evaluation_level(title ="Cross-Validation Summary - SAMPLE LEVEL:", 
                                   search_terms = ['val_accuracy', 'val_precision', 'val_recall', 'val_f1_score'], 
                                   cv_results= cv_results)
            evaluation_level(title = "Cross-Validation Summary - TEST LEVEL (averaged per test):", 
                                   search_terms=['mean_test_accuracy', 'mean_test_precision', 'mean_test_recall', 'mean_test_f1'],
                                   cv_results = cv_test_level_metrics)
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
        save_bestscaler(title='', best_scaler=best_scaler)

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

        all_folds_tests = observe_performance_per_test_phases(original_tests_fold, holdout_tests_original, method=method, modelname=model_name.strip('.h5'))

        return best_model, best_scaler, best_fold_idx, all_folds_tests

    else:
        # Single model training (no CV) or loading existing model
        if not load_existing:
            # Train on all data
            X_train, X_val, y_train, y_val, X_holdout, y_holdout = split_data(X, y, test_index, fold_idx=None) # type: ignore

            modelObj = classes.MlModel(model_name=model_name)
            modelObj.define_model()
            modelObj.model_fit(X_train, y_train, X_val, y_val, plot=True, information=modelcomments)
            modelObj.save_model(title=modelObj.model_name)
        else:
            modelObj = classes.MlModel(model_name=model_name)
            modelObj.load_model(title=modelObj.model_name)

        return modelObj

def transform_target_phases(target_data):
    """
    Docstring for transform_target_phases
    
    :param target_data: array of string values for each sample
    return: array of codes mapped to the string values
    """
    label_encoder = LabelEncoder()
    target_data = label_encoder.fit_transform(target_data)
    print("Encoded classes: ")
    print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))) # type: ignore
    return target_data
    

def prep_data_phases(all_tests, fold_idx=None, n_splits=5, window_size=60, stride=30, input_type='triaxial_acc', output_steps=0):
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
    target_col = 'testPhases'
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
        if target_col in test.processed_data.columns:
            target_data = test.processed_data[target_col].values # In the binary part this is an array of booleans (True/False)
            target_data = transform_target_phases(target_data)

            # Create sliding windows for this test
            n_samples = len(test_data)
            if n_samples < window_size:
                print(f"Warning: Test {test_idx} has only {n_samples} samples, skipping...")
                continue

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

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"test_indices shape: {test_indices.shape}")
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


def evaluate_cv_phases(modelObj, X_val, y_val, fold, cv_results, fold_models, best_val_f1, best_fold_idx, num_classes=1):
    """
    Evaluate model on validation set at sample level for the specific fold and update CV results.
    Handles both binary and multiclass classification.

    Args:
        modelObj: Trained model
        X_val: Validation features
        y_val: Validation targets
        fold: Current fold number
        cv_results: List of results dictionaries
        fold_models: List of model objects
        best_val_f1: Current best F1 score
        best_fold_idx: Index of best fold
        num_classes: 1 for binary, >1 for multiclass

    Returns:
        cv_results, fold_models, best_val_f1, best_fold_idx
    """
    from sklearn.metrics import f1_score, classification_report, confusion_matrix

        # Get predictions
    y_pred = modelObj.fitted_model.predict(X_val, verbose=0)
    # Check if the prediction has different classes or only 1 outcome: 
    if len(np.unique(y_pred)) == 1: 
        print("Warning: Model predictions have only one unique value. Check model training.")
    
    # Evaluate on validation set (sample-level)
    if num_classes == 1:
        # Binary: returns [loss, acc, prec, recall]
        eval_results = modelObj.fitted_model.evaluate(X_val, y_val, verbose=0)
        val_loss = eval_results[0]
        val_acc = eval_results[1]
        val_prec = eval_results[2] if len(eval_results) > 2 else None
        val_recall = eval_results[3] if len(eval_results) > 3 else None

        # Binary classification
        y_pred_binary = (y_pred > running_settings.parameters['classBinaryTresh']).astype(int)
        y_val_flat = y_val.reshape(-1)
        y_pred_flat = y_pred_binary.reshape(-1)
        
        # Compute F1 score
        f1 = f1_score(y_val_flat, y_pred_flat, zero_division=0)
        
        # Compute precision/recall if not from model
        if val_prec is None:
            from sklearn.metrics import precision_score, recall_score
            val_prec = precision_score(y_val_flat, y_pred_flat, zero_division=0)
            val_recall = recall_score(y_val_flat, y_pred_flat, zero_division=0)
        
    else:
        # Multiclass: returns [loss, acc]
        eval_results = modelObj.fitted_model.evaluate(X_val, y_val, verbose=0)
        val_loss = eval_results[0]
        val_acc = eval_results[1]
        val_prec = None  # Will compute from sklearn
        val_recall = None

        # Multiclass classification
        # y_pred shape: (samples, timesteps, num_classes)
        if 'crf' not in modelObj.model_name:
            y_pred_classes = np.argmax(y_pred, axis=-1) # Get class with highest probability
        else:  
            y_pred_classes = y_pred  # For CRF, predictions are already class labels
            # Extract accuracy value
            val_acc = val_acc['accuracy'].numpy()
            val_loss = val_loss.numpy()

        # For CRF, predictions are already class labels
        y_val_flat = y_val.reshape(-1).astype(int)
        y_pred_flat = y_pred_classes.reshape(-1).astype(int)

        # Compute macro F1 score (average of F1 for each class)
        f1 = f1_score(y_val_flat, y_pred_flat, average='macro', zero_division=0)
        
        # Compute precision and recall
        from sklearn.metrics import precision_score, recall_score
        val_prec = precision_score(y_val_flat, y_pred_flat, average='macro', zero_division=0)
        val_recall = recall_score(y_val_flat, y_pred_flat, average='macro', zero_division=0)

    # Store results
    result_dict = {
        'fold': fold + 1,
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'val_precision': val_prec,
        'val_recall': val_recall,
        'val_f1_score': f1
    }
    
    # Add per-class metrics for multiclass
    if num_classes > 1:
        # Compute per-class F1 scores
        f1_per_class = f1_score(y_val_flat, y_pred_flat, average=None, zero_division=0)
        for i, f1_class in enumerate(f1_per_class): # type: ignore
            result_dict[f'val_f1_class_{i}'] = f1_class
    
    cv_results.append(result_dict)
    fold_models.append(modelObj)

    # Track best fold
    if f1 > best_val_f1:
        best_val_f1 = f1
        best_fold_idx = fold

    # Print results
    print(f"\nFold {fold + 1} Sample-Level Validation Results:")
    print(f"  Loss:      {val_loss:.4f}")
    print(f"  Accuracy:  {val_acc:.4f}")
    print(f"  Precision: {val_prec:.4f}")
    print(f"  Recall:    {val_recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    if num_classes > 1:
        print(f"\n  Per-Class F1 Scores:")
        for i, f1_class in enumerate(f1_per_class): # type: ignore
            print(f"    Class {i}: {f1_class:.4f}")
        
        # Optional: Print confusion matrix
        cm = confusion_matrix(y_val_flat, y_pred_flat)
        print(f"\n  Confusion Matrix:")
        print(cm)

    return cv_results, fold_models, best_val_f1, best_fold_idx

def plot_ml_prediction_phases(test, est_timestamps_startend, gt_timestamps_startend, errors):
    """
    Plot binary classification predictions with ground truth markers and sensor data.
    """
    PHASE_COLORS = {
    'No test': 'black',
    'Sit-to-stand': '#1f77b4',          # blue
    'Walking1': '#2ca02c',               # green
    'Turn1': '#ff7f0e',                  # orange
    'Walking2': '#9467bd',               # purple
    'Turn2+Stand-to-sit': '#d62728',     # red
    }

    df = test.processed_data
    if 'processed_testProba_phases' in test.__dict__.keys():
        proba_df = test.processed_testProba_phases
        proba=True
    else:
        proba=False

    # # ---------- ground truth ----------
    gt_indices = df[df['testBool'] == True]
    if len(gt_indices) > 0:
        gt_start = gt_indices['msFromStart'].values[0]
        gt_end = gt_indices['msFromStart'].values[-1]
    else:
        gt_start = None
        gt_end = None

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True
    )
    if proba:
        # ======================================================================
        # Plot 1: Prediction probabilities (multiclass)
        # ======================================================================
        for col in proba_df.columns:
            label = PHASE_LABELS.get(col, col)
            ax1.plot(
                df['msFromStart'],
                proba_df[col],
                linewidth=2,
                label=label, 
                color=PHASE_COLORS[label]
            )

        ax1.set_ylabel('Class probability')
        ax1.set_ylim(-0.02, 1.02)
        ax1.legend(loc='upper left', ncol=2)
        ax1.grid(True, alpha=0.3)
    else:
        print("No phase probabilities found for plotting.")
    # ======================================================================
    # Plot 2: Multiclass phase timelines (GT vs EST)
    # ======================================================================
    y_gt = 1.0
    y_est = 0.3
    band_height = 0.15
    for phase in PHASE_ORDER:
        color = PHASE_COLORS.get(phase, 'gray')

        # ---------------- GT ----------------
        if phase in gt_timestamps_startend:
            start, end = gt_timestamps_startend[phase]

            ax2.axvline(start, ymin=y_gt - band_height, ymax=y_gt + band_height,
                        color=color, linewidth=2)
            ax2.axvline(end, ymin=y_gt - band_height, ymax=y_gt + band_height,
                        color=color, linewidth=2)

            ax2.fill_betweenx(
                [y_gt - band_height, y_gt + band_height],
                start, end,
                color=color,
                alpha=0.3
            )

            ax2.text(start, y_gt + band_height + 0.05, phase,
                    fontsize=9, color=color, va='bottom')

        # ---------------- EST ----------------
        if phase in est_timestamps_startend:
            start, end = est_timestamps_startend[phase]

            ax2.axvline(start, ymin=y_est - band_height, ymax=y_est + band_height,
                        color=color, linestyle='--', linewidth=2, label=f"{phase} start")
            ax2.axvline(end, ymin=y_est - band_height, ymax=y_est + band_height,
                        color=color, linestyle='--', linewidth=2, label=f"{phase} end")

            ax2.fill_betweenx(
                [y_est - band_height, y_est + band_height],
                start, end,
                color=color,
                alpha=0.15
            )

    # Axis formatting
    ax2.set_yticks([y_gt, y_est])
    ax2.set_yticklabels(['Ground truth', 'Estimated'])
    ax2.set_ylabel('Phase timelines')
    ax2.grid(True, axis='x', alpha=0.3)

    # ======================================================================
    # Plot 3: Sensor data (UNCHANGED, as requested)
    # ======================================================================
    ax3.plot(df['msFromStart'], df['sqrt(X²+Y²+Z²)'], linewidth=2)
    ax3.set_ylabel('Acceleration (m/s²)')

    ax3_right = ax3.twinx()

    if 'alpha' in df.columns:
        ax3_right.plot(df['msFromStart'], df['alpha'], linestyle='--')
        ax3_right.plot(df['msFromStart'], df['beta'], linestyle='-.')
        ax3_right.plot(df['msFromStart'], df['gamma'], linestyle=':')

    ax3_right.plot(df['msFromStart'], df['rotRate.alpha'], alpha=0.7)
    ax3_right.plot(df['msFromStart'], df['rotRate.beta'], alpha=0.7)
    ax3_right.plot(df['msFromStart'], df['rotRate.gamma'], alpha=0.7)

    ax3.set_xlabel('Time (ms from start)')
    ax3.grid(True, alpha=0.3)

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

def iteratetimebtw_phases(estimation, test, main_threshold_ms=15000, right_threshold_ms=5000, verbose=True):
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
    if 'predicted_testBool' in processed_data.columns: 
        estimation = processed_data[processed_data['predicted_testBool'] == True]
    else: 
        estimation = processed_data[processed_data['predicted_testClass']!= 0]

    if len(estimation) > 0:
        msStart = estimation['msFromStart'].values[0]
        msEnd = estimation['msFromStart'].values[-1]
        if 'predicted_testBool' not in processed_data.columns:
            if len(estimation['msFromStart'][processed_data['predicted_testPhases'] == 'Walking1'])>0:
                msWalking1 = estimation['msFromStart'][processed_data['predicted_testPhases'] == 'Walking1'].values[0]
            else: 
                msWalking1 = msStart
            if len(estimation['msFromStart'][processed_data['predicted_testPhases'] == 'Turn1'])>0:
                msTurn1 = estimation['msFromStart'][processed_data['predicted_testPhases'] == 'Turn1'].values[0]
            else: 
                msTurn1 = msWalking1
            if len(estimation['msFromStart'][processed_data['predicted_testPhases'] == 'Walking2'])>0:
                msWalking2 = estimation['msFromStart'][processed_data['predicted_testPhases'] == 'Walking2'].values[0]
            else:
                msWalking2 = msTurn1
            if len(estimation['msFromStart'][processed_data['predicted_testPhases'] == 'Turn2+Stand-to-sit'])>0:
                msTurn2 = estimation['msFromStart'][processed_data['predicted_testPhases'] == 'Turn2+Stand-to-sit'].values[0]
            else:
                msTurn2 = msEnd

            return {'msStart': msStart, 'msEnd': msEnd, 'msWalking1': msWalking1, 'msTurn1': msTurn1, 'msWalking2': msWalking2, 'msTurn2': msTurn2}
        else:
            return msStart, msEnd
    else:
        msStart = processed_data['msFromStart'].values[0]
        msEnd = processed_data['msFromStart'].values[-1]
        return msStart, msEnd

def compute_error_phases(est_timestamps, test):
    """
    Returns a dict:
        { phase_name : error_in_ms }
    where error = estimated_duration - ground_truth_duration
    """

    df = test.processed_data
    errors = {}
    gt_timestamps_startend = {}
    est_timestamps_startend = {}
    # ---------- per-phase errors ----------
    for phase, (end_key, start_key) in PHASE_MAP.items():

        # ---------- ground truth duration ----------
        if phase == "totalDuration":
            not_no_test = df["testPhases"] != "No test"

            if not not_no_test.any():
                errors[phase] = np.nan
                continue

            gt_start = df.loc[not_no_test, "msFromStart"].iloc[0]
            gt_end   = df.loc[not_no_test, "msFromStart"].iloc[-1]
            gt_duration = gt_end - gt_start
        else: 
            gt_rows = df[df["testPhases"] == phase]

            if gt_rows.empty:
                errors[phase] = np.nan
                continue

            gt_start = gt_rows["msFromStart"].iloc[0]
            gt_end   = gt_rows["msFromStart"].iloc[-1]
            gt_duration = gt_end - gt_start
        
        gt_timestamps_startend[phase] = (gt_start, gt_end) 

        # ---------- estimated duration ----------
        if end_key not in est_timestamps or start_key not in est_timestamps:
            est_duration = np.nan
        else:
            est_duration = est_timestamps[end_key] - est_timestamps[start_key]
            est_timestamps_startend[phase] = (est_timestamps[start_key],  est_timestamps[end_key])
        # ---------- error ----------
        errors[phase] = est_duration - gt_duration
    
    return errors, est_timestamps_startend, gt_timestamps_startend


def approach_duration_estimation_phases(test, method=''):
    try:
        if 'testPhases' in test.processed_data.columns:
            processed_data = test.processed_data
            print("Using basic approach for duration estimation")
            # msStart, msEnd, msWalking1, msTurn1, msWalking2, msTurn2 = basic_approach(processed_data) # type: ignore
            est_timestamps = basic_approach(processed_data)
            errors, est_timestamps_startend, gt_timestamps_startend = compute_error_phases(est_timestamps, test)

            #if errors['totalDuration']>1000:
            if False:
                plot_ml_prediction_phases(test, est_timestamps_startend, gt_timestamps_startend, errors)
            # if abs(error) > 10:
            #     # plot_ml_prediction(test, error)
            #     verbose=True
            # else:
            #     verbose=False

            #if method == 'timebtwbool':
            if False:
                print("Using time between peaks for duration estimation")
                estimation = processed_data[['msFromStart', 'predicted_testPhases']]
                est_timestamps = iteratetimebtw_phases(estimation, test, main_threshold_ms=9000, right_threshold_ms=5000, verbose=1)
                errors = compute_error_phases(est_timestamps, test)

            # if msStart is not None and msEnd is not None:
            #     return msStart/1000, msEnd/1000
            # else: 
            #     return None, None
            return est_timestamps_startend, gt_timestamps_startend, errors
        else:
            return None
    except:
        processed_data = test.processed_data
        print(f"Issues with normal approaches, taking first and last raw sample. {test.user_id + '_' + str(test.session_id)}")
        return processed_data['msFromStart'].values[0]/1000, processed_data['msFromStart'].values[-1]/1000


def evaluate_duration_tests_phases(tests_original, method):
    for test_id, test in tests_original.items():
        print(f"\nEvaluating duration for Test ID: {test_id}")
        if test.user_id + '_' + str(test.session_id) == '2_parkapp_8' or test.user_id + '_' + str(test.session_id) == '4_parkapp_8' or test.user_id + '_' + str(test.session_id) == '12_parkapp_1':
            print("to check better here")

        if test.processed_data.shape[0] > 0 and 'testPhases' in test.processed_data.columns:
            est_timestamps_startend, gt_timestamps_startend, errors = approach_duration_estimation_phases(test, method='timebtwbool') # type: ignore

            test.results[method] = est_timestamps_startend
            test.gt_phases = gt_timestamps_startend
            test.error[method] = errors

            # if abs(error) > 20:
            #     print("Error > 20")
            #     plot_ml_prediction(test, error)
        else:
            test.results[method] = None
            test.gt_phases = None

    return tests_original


def observe_performance_per_test_phases(original_tests_fold, holdout_tests_original, method, modelname):
    # Observe performances across tests
    if len(holdout_tests_original) > 0:
        holdout_tests_original = evaluate_duration_tests_phases(holdout_tests_original, method)

        utils_evaluation.evaluate_results_phases_pt2(list(holdout_tests_original.values()), eval_type='duration',
                                          method=method, gttype='gwalk',
                                          dataset='holdout', title=modelname + '_' + method, logging=True)

        all_holdout_remained = utils_dataquality.plot_tests_witherror(holdout_tests_original.values(), error_threshold=4, method='ml')

    if len(original_tests_fold) > 0:
        all_folds_tests = []
        for f in original_tests_fold.keys():
            val_tests_original = original_tests_fold[f]
            val_tests_original = evaluate_duration_tests_phases(val_tests_original, method)
            all_folds_tests.extend(list(val_tests_original.values()))

        if False: 
            all_fold_remained = utils_dataquality.plot_tests_witherror(all_folds_tests, error_threshold=30, method='ml')

        utils_evaluation.evaluate_results_phases_pt2(all_folds_tests, eval_type='phases',
                                          method=method, gttype='manual',
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

def fittingmodel_phases(fold, n_splits, X_train, y_train, X, X_val, y_val, model_name, modelcomments,
                 architecture, training_epochs, save_model, output_steps, n_classes):
    print(f"\n{'=' * 60}")
    print(f"Training Fold {fold + 1}/{n_splits}")
    print(f"{'=' * 60}")

    fold_model_name = f"{model_name.replace('.h5', '')}_fold{fold + 1}.h5"
    modelObj = classes.MlModelphases(model_name=fold_model_name)
    modelObj.define_model(save_model=save_model, n_features=X.shape[2], architecture=architecture, output_steps=output_steps, num_classes= n_classes)
    modelObj.model_fit(
        X_train, y_train, X_val, y_val,
        plot=False,
        epochs=training_epochs,
        information=f"{modelcomments} - Fold {fold + 1}",
        save_model=save_model)
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

def reconstruct_from_windows_phases(predictions, window_size, stride, original_length, num_classes=1, crf=False):
    """
    Reconstruct original sequence from overlapping window predictions.
    Handles both binary and multiclass classification.

    Parameters:
    -----------
    predictions : array-like, shape (n_windows, window_size, n_features) or (n_windows, window_size, num_classes)
        Predictions from windowed data
    window_size : int
        Size of each window
    stride : int
        Stride used when creating windows
    original_length : int
        Length of the original sequence
    num_classes : int
        1 for binary classification, >1 for multiclass

    Returns:
    --------
    reconstructed : array, shape (original_length,) for binary or (original_length, num_classes) for multiclass
        Reconstructed predictions in original sequence length
    counts : array, shape (original_length,)
        Number of predictions averaged for each position
    """
    if crf:
                # Shape handling: (n_windows, window) or (n_windows, window, 1)
        if predictions.ndim == 3:
            predictions = predictions.squeeze(-1)

        n_windows = predictions.shape[0]

        # votes[t, c] = number of windows voting class c at time t
        votes = np.zeros((original_length, num_classes), dtype=np.int32)
        counts = np.zeros(original_length, dtype=np.int32)

        for w in range(n_windows):
            start = w * stride
            end = min(start + window_size, original_length)
            valid_len = end - start

            labels = predictions[w, :valid_len].astype(int)

            for i, label in enumerate(labels):
                votes[start + i, label] += 1
                counts[start + i] += 1

        # Final label = majority vote
        reconstructed = np.argmax(votes, axis=1)

        return reconstructed, counts

    n_windows = len(predictions)
    
    if num_classes == 1:
        # Binary classification
        n_features = predictions.shape[-1] if len(predictions.shape) > 2 else 1
        reconstructed = np.zeros((original_length, n_features) if n_features > 1 else (original_length,))
    else:
        # Multiclass classification - predictions shape: (n_windows, window_size, num_classes)
        reconstructed = np.zeros((original_length, num_classes))
    
    counts = np.zeros(original_length)

    # Accumulate predictions
    for window_idx in range(n_windows):
        start_idx = window_idx * stride
        end_idx = start_idx + window_size

        # Handle edge case where window extends beyond original length
        valid_end = min(end_idx, original_length)
        valid_window_size = valid_end - start_idx

        if num_classes == 1:
            # Binary classification
            n_features = predictions.shape[-1] if len(predictions.shape) > 2 else 1
            if n_features > 1:
                reconstructed[start_idx:valid_end] += predictions[window_idx, :valid_window_size]
            else:
                reconstructed[start_idx:valid_end] += predictions[window_idx, :valid_window_size].flatten()
        else:
            # Multiclass classification - accumulate probability distributions
            if not crf:
                reconstructed[start_idx:valid_end] += predictions[window_idx, :valid_window_size, :]
            else: 
                reconstructed[start_idx:valid_end] = predictions[window_idx, :valid_window_size, :]

        counts[start_idx:valid_end] += 1

    # Average overlapping predictions
    counts = np.maximum(counts, 1)  # Avoid division by zero
    
    if num_classes == 1:
        # Binary classification
        n_features = predictions.shape[-1] if len(predictions.shape) > 2 else 1
        if n_features > 1:
            reconstructed = reconstructed / counts[:, np.newaxis]
        else:
            reconstructed = reconstructed / counts
    else:
        # Multiclass classification - average probabilities
        if not crf:
            reconstructed = reconstructed / counts[:, np.newaxis]

    return reconstructed, counts


def evaluate_cv_per_test_phases(modelObj, X_val, y_val, val_test_index, val_tests, fold, 
                         val_tests_original, output_steps, num_classes=1):
    """
    Evaluate model performance at the TEST level, aggregating predictions per test.
    Handles both binary and multiclass classification.

    Args:
        modelObj: Trained model object
        X_val: Validation features
        y_val: Validation targets
        val_test_index: Array indicating which test each sequence belongs to
        val_tests: List of test IDs in validation set
        fold: Current fold number
        val_tests_original: Dictionary of original test data
        output_steps: Number of output steps (0 for full window)
        num_classes: 1 for binary, >1 for multiclass

    Returns:
        test_results_df: DataFrame with per-test metrics
        test_level_metrics: Dictionary with aggregated test-level performance
        val_tests_original: Original test data dictionary
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    # Calculate per-test metrics
    test_results = {}
    if 'crf' in modelObj.model_name:
        crf = True
    else:
        crf = False

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

        # Reconstruct from windows based on output_steps
        if output_steps == 0:
            y_pred_reconstructed, overlap_counts = reconstruct_from_windows_phases(
                predictions=y_pred_test,
                window_size=60,
                stride=running_settings.parameters['stride'],
                original_length=len(original_test.processed_data),
                num_classes=num_classes, 
                crf=crf
            )

            y_true_original, overlap_counts_true = reconstruct_from_windows_phases(
                predictions=y_val_test,
                window_size=60,
                stride=running_settings.parameters['stride'],
                original_length=len(original_test.processed_data),
                num_classes=num_classes, 
                crf=crf
            )

        print(f"Original test length: {len(original_test.processed_data)}")
        print(f"Reconstructed predictions shape: {y_pred_reconstructed.shape}")
        print(f"Reconstructed ground truth shape: {y_true_original.shape}")

        # Convert predictions to class labels
        if num_classes == 1:
            # Binary classification
            y_pred_binary_reconstructed = (y_pred_reconstructed > running_settings.parameters['classBinaryTresh']).astype(int)
            y_true_labels = y_true_original.astype(int)
            
            # Store predictions
            original_test.processed_data['predicted_testProba'] = y_pred_reconstructed
            original_test.processed_data['predicted_testBool'] = y_pred_binary_reconstructed
            
            # Calculate metrics
            accuracy = accuracy_score(y_true_labels, y_pred_binary_reconstructed)
            precision = precision_score(y_true_labels, y_pred_binary_reconstructed, zero_division=0)
            recall = recall_score(y_true_labels, y_pred_binary_reconstructed, zero_division=0)
            f1 = f1_score(y_true_labels, y_pred_binary_reconstructed, zero_division=0)
            
            print(f"\nBinary Classification Metrics:")
            
        else:
            # Multiclass classification
            # y_pred_reconstructed shape: (original_length, num_classes)
            if not crf:
                y_pred_classes = np.argmax(y_pred_reconstructed, axis=-1)
                # Keep only the first value of each block of 4 elements
                y_true_labels = y_true_original[:, 0].astype(int)  
            else: # Shape: (952,)
                y_true_labels = y_true_original.astype(int)
                y_pred_classes = y_pred_reconstructed.astype(int)
            
            # Store predictions
            original_test.processed_data['predicted_testClass'] = y_pred_classes
            original_test.processed_data['predicted_testPhases'] = original_test.processed_data['predicted_testClass'].map(class_names)
            if not crf: 
                original_test.processed_testProba_phases = pd.DataFrame(
                y_pred_reconstructed,
                columns=[str(i)+':'+class_names[i] for i in range(len(class_names))]
            )
                        
            # Calculate macro-averaged metrics
            accuracy = accuracy_score(y_true_labels, y_pred_classes)
            precision = precision_score(y_true_labels, y_pred_classes, average='macro', zero_division=0)
            recall = recall_score(y_true_labels, y_pred_classes, average='macro', zero_division=0)
            f1 = f1_score(y_true_labels, y_pred_classes, average='macro', zero_division=0)
            
            # Calculate per-class metrics
            f1_per_class = f1_score(y_true_labels, y_pred_classes, average=None, zero_division=0)
            
            print(f"\nMulticlass Classification Metrics (Macro-averaged):")

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        # Store results
        test_results[test_id] = {
            'test_id': test_id,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Add per-class F1 for multiclass
        if num_classes > 1:
            for i, f1_class in enumerate(f1_per_class):
                test_results[test_id][f'f1_class_{i}'] = f1_class
                print(f"  Class {i} F1: {f1_class:.4f}")
            
            # Print confusion matrix
            cm = confusion_matrix(y_true_labels, y_pred_classes)
            print(f"\nConfusion Matrix:")
            print(cm)

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
    
    # Add per-class metrics for multiclass
    if num_classes > 1:
        for i in range(num_classes):
            col_name = f'f1_class_{i}'
            if col_name in test_results_df.columns:
                test_level_metrics[f'mean_f1_class_{i}'] = test_results_df[col_name].mean()
                test_level_metrics[f'std_f1_class_{i}'] = test_results_df[col_name].std()

    # Print test-level summary
    print(f"\nTest-Level Performance (averaging across {len(val_tests)} tests):")
    print(f"  Mean Accuracy:  {test_level_metrics['mean_test_accuracy']:.4f} ± {test_level_metrics['std_test_accuracy']:.4f}")
    print(f"  Mean Precision: {test_level_metrics['mean_test_precision']:.4f} ± {test_level_metrics['std_test_precision']:.4f}")
    print(f"  Mean Recall:    {test_level_metrics['mean_test_recall']:.4f} ± {test_level_metrics['std_test_recall']:.4f}")
    print(f"  Mean F1-Score:  {test_level_metrics['mean_test_f1']:.4f} ± {test_level_metrics['std_test_f1']:.4f}")
    
    if num_classes > 1:
        print(f"\n  Per-Class Mean F1 Scores:")
        for i in range(num_classes):
            if f'mean_f1_class_{i}' in test_level_metrics:
                print(f"    Class {i}: {test_level_metrics[f'mean_f1_class_{i}']:.4f} ± {test_level_metrics[f'std_f1_class_{i}']:.4f}")

    return test_results_df, test_level_metrics, val_tests_original

import seaborn as sns

def plot_phase_evaluation_summary(df, title=None, figsize=(16, 10)):
    """
    Creates a single figure summarizing phase-level performance:
    - Boundary errors
    - Duration errors
    - IoU
    - Coverage
    """

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # --------------------------------------------------
    # 1. Start / End boundary errors
    # --------------------------------------------------
    df_boundary = df.melt(
        id_vars=["id", "phase"],
        value_vars=["start_error_ms", "end_error_ms"],
        var_name="error_type",
        value_name="error_ms"
    )

    sns.boxplot(
        data=df_boundary,
        x="phase",
        y="error_ms",
        hue="error_type",
        order=PHASE_ORDER,
        ax=ax1
    )
    ax1.axhline(0, linestyle="--", color="black", linewidth=1)
    ax1.set_title("Phase boundary errors")
    ax1.set_ylabel("Error (ms)")
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(title="", loc="upper right")

    # --------------------------------------------------
    # 2. Absolute duration error
    # --------------------------------------------------
    df_dur = df.copy()
    df_dur["abs_duration_error_ms"] = df_dur["duration_error_ms"].abs()

    sns.violinplot(
        data=df_dur,
        x="phase",
        y="abs_duration_error_ms",
        order=PHASE_ORDER,
        inner="quartile",
        cut=0,
        ax=ax2
    )
    ax2.set_title("Phase duration accuracy")
    ax2.set_ylabel("Absolute error (ms)")
    ax2.set_xlabel("")
    ax2.tick_params(axis="x", rotation=30)

    # --------------------------------------------------
    # 3. IoU distribution
    # --------------------------------------------------
    sns.boxplot(
        data=df,
        x="phase",
        y="iou",
        order=PHASE_ORDER,
        ax=ax3
    )
    ax3.set_ylim(0, 1)
    ax3.set_title("Temporal overlap (IoU)")
    ax3.set_ylabel("IoU")
    ax3.set_xlabel("")
    ax3.tick_params(axis="x", rotation=30)

    # --------------------------------------------------
    # 4. Coverage analysis
    # --------------------------------------------------
    df_cov = df.melt(
        id_vars=["id", "phase"],
        value_vars=["gt_coverage", "pred_coverage"],
        var_name="coverage_type",
        value_name="coverage"
    )

    sns.boxplot(
        data=df_cov,
        x="phase",
        y="coverage",
        hue="coverage_type",
        order=PHASE_ORDER,
        ax=ax4
    )
    ax4.set_ylim(0, 1)
    ax4.set_title("Coverage analysis")
    ax4.set_ylabel("Coverage ratio")
    ax4.set_xlabel("")
    ax4.tick_params(axis="x", rotation=30)
    ax4.legend(title="", loc="lower right")

    # --------------------------------------------------
    # Global formatting
    # --------------------------------------------------
    if title is not None:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, Bidirectional, LSTM, Concatenate, Lambda, Dense, TimeDistributed, Layer
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

# Phase labels and ordering
PHASE_LABELS = {
    '0:No test': 'No test',
    '1:Sit-to-stand': 'Sit-to-stand',
    '2:Turn1': 'Turn1',
    '3:Turn2+Stand-to-sit': 'Turn2+Stand-to-sit',
    '4:Walking1': 'Walking1',
    '5:Walking2': 'Walking2',
}

PHASE_ORDER = [
    'Sit-to-stand',
    'Walking1',
    'Turn1',
    'Walking2',
    'Turn2+Stand-to-sit',
]

# Create label to index mapping
ALL_PHASES = ['No test'] + PHASE_ORDER
PHASE_TO_IDX = {phase: idx for idx, phase in enumerate(ALL_PHASES)}
IDX_TO_PHASE = {idx: phase for phase, idx in PHASE_TO_IDX.items()}
NUM_CLASSES = len(ALL_PHASES)


class CRFLayer(Layer):
    """
    Conditional Random Field layer with transition constraints.
    Enforces that phases can only transition forward according to PHASE_ORDER.
    """
    
    def __init__(self, num_classes, **kwargs):
        super(CRFLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        
    def build(self, input_shape):
        # Transition matrix: [from_state, to_state]
        # Initialize with very negative values (impossible transitions)
        initial_transitions = np.full((self.num_classes, self.num_classes), -1e10, dtype=np.float32)
        
        # Build allowed transitions based on PHASE_ORDER
        # 'No test' (idx 0) can transition to itself or 'Sit-to-stand' (idx 1)
        initial_transitions[0, 0] = 0.0  # No test -> No test
        initial_transitions[0, 1] = 0.0  # No test -> Sit-to-stand
        
        # Each phase can stay in itself or move to the next phase
        for i in range(1, self.num_classes):
            initial_transitions[i, i] = 0.0  # Stay in current phase
            if i < self.num_classes - 1:
                initial_transitions[i, i + 1] = 0.0  # Move to next phase
        
        # Last phase can transition back to 'No test'
        initial_transitions[self.num_classes - 1, 0] = 0.0
        
        self.transitions = self.add_weight(
            name='transitions',
            shape=(self.num_classes, self.num_classes),
            initializer=tf.constant_initializer(initial_transitions),
            trainable=True
        )
        
        # Start and end transitions
        self.start_transitions = self.add_weight(
            name='start_transitions',
            shape=(self.num_classes,),
            initializer='zeros',
            trainable=True
        )
        
        self.end_transitions = self.add_weight(
            name='end_transitions',
            shape=(self.num_classes,),
            initializer='zeros',
            trainable=True
        )
        
        super(CRFLayer, self).build(input_shape)
    
    def call(self, inputs, mask=None, training=None):
        """
        During training: return emissions for loss computation
        During inference: return viterbi decoded sequence
        """
        if training:
            # Return emissions during training
            return inputs
        else:
            # Return decoded sequence during inference
            return self.viterbi_decode(inputs, mask)
    
    def viterbi_decode(self, emissions, mask=None):
        """
        Viterbi algorithm for finding most likely sequence.
        """
        batch_size = tf.shape(emissions)[0]
        seq_length = tf.shape(emissions)[1]
        
        # Initialize with start transitions
        score = emissions[:, 0, :] + self.start_transitions  # (batch, num_classes)
        
        # Store backpointers
        backpointers = []
        
        # Forward pass
        for i in range(1, emissions.shape[1]):
            # Expand dimensions for broadcasting
            score_expanded = tf.expand_dims(score, 2)  # (batch, num_classes, 1)
            emission_i = emissions[:, i, :]  # (batch, num_classes)
            
            # Calculate scores for all transitions
            next_score = score_expanded + self.transitions  # (batch, from_class, to_class)
            next_score = next_score + tf.expand_dims(emission_i, 1)  # (batch, from_class, to_class)
            
            # Find best previous state for each current state
            backpointer = tf.argmax(next_score, axis=1, output_type=tf.int32)  # (batch, num_classes)
            score = tf.reduce_max(next_score, axis=1)  # (batch, num_classes)
            
            backpointers.append(backpointer)
        
        # Add end transitions
        score = score + self.end_transitions
        
        # Backward pass to get best path
        best_last_tag = tf.argmax(score, axis=1, output_type=tf.int32)  # (batch,)
        
        # Decode the best path
        best_tags = [best_last_tag]
        
        for backpointer in reversed(backpointers):
            best_last_tag = tf.gather_nd(
                backpointer,
                tf.stack([tf.range(batch_size), best_last_tag], axis=1)
            )
            best_tags.append(best_last_tag)
        
        # Reverse to get correct order and convert to float for consistency
        best_tags = tf.stack(list(reversed(best_tags)), axis=1)  # (batch, seq_length)
        
        return tf.cast(best_tags, tf.float32)
    
    def compute_loss(self, emissions, tags):
        """
        Compute CRF loss (negative log-likelihood).
        emissions: (batch_size, seq_length, num_classes)
        tags: (batch_size, seq_length) - ground truth labels
        """
        # Calculate score for the gold sequence
        gold_score = self.score_sentence(emissions, tags)
        
        # Calculate normalization (all possible sequences)
        norm_score = self.log_norm(emissions)
        
        # Loss is negative log-likelihood
        loss = norm_score - gold_score
        
        return tf.reduce_mean(loss)
    
    def score_sentence(self, emissions, tags):
        """Score of a given tag sequence."""
        batch_size = tf.shape(emissions)[0]
        seq_length = tf.shape(emissions)[1]
        tags = tf.cast(tags, tf.int32)
        
        # Start transitions
        score = tf.gather(self.start_transitions, tags[:, 0])
        score += tf.gather_nd(emissions[:, 0, :], 
                             tf.stack([tf.range(batch_size), tags[:, 0]], axis=1))
        
        # Transitions and emissions
        for i in range(1, emissions.shape[1]):
            indices = tf.stack([tags[:, i-1], tags[:, i]], axis=1)
            transition_score = tf.gather_nd(self.transitions, indices)
            
            emission_indices = tf.stack([tf.range(batch_size), tags[:, i]], axis=1)
            emission_score = tf.gather_nd(emissions[:, i, :], emission_indices)
            
            score += transition_score + emission_score
        
        # End transitions
        score += tf.gather(self.end_transitions, tags[:, -1])
        
        return score
    
    def log_norm(self, emissions):
        """Log-sum-exp of all possible sequences (partition function)."""
        seq_length = tf.shape(emissions)[1]
        
        # Initialize with start transitions
        score = emissions[:, 0, :] + self.start_transitions
        
        # Forward algorithm
        for i in range(1, emissions.shape[1]):
            score_expanded = tf.expand_dims(score, 2)
            emission_i = tf.expand_dims(emissions[:, i, :], 1)
            
            next_score = score_expanded + self.transitions + emission_i
            score = tf.reduce_logsumexp(next_score, axis=1)
        
        # Add end transitions
        score = score + self.end_transitions
        
        return tf.reduce_logsumexp(score, axis=1)
    
    def get_config(self):
        config = super(CRFLayer, self).get_config()
        config.update({'num_classes': self.num_classes})
        return config

class CRFModel(Model):
    """
    Custom Model class that handles CRF loss computation.
    """
    def __init__(self, inputs, outputs, crf_layer, emissions_output, **kwargs):
        super(CRFModel, self).__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.crf_layer = crf_layer
        self.emissions_output = emissions_output
        
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            # Get emissions (forward pass with training=True)
            emissions = self.emissions_output(x, training=True)
            
            # Compute CRF loss
            loss = self.crf_layer.compute_loss(emissions, y)
            
            # Add regularization losses if any
            if self.losses:
                loss += tf.add_n(self.losses)
        
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Get predictions for metrics (viterbi decode)
        y_pred = self.crf_layer.viterbi_decode(emissions)
        
        # Update metrics
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x, y = data
        
        # Get emissions
        emissions = self.emissions_output(x, training=False)
        
        # Compute CRF loss
        loss = self.crf_layer.compute_loss(emissions, y)
        
        # Get predictions (viterbi decode)
        y_pred = self.crf_layer.viterbi_decode(emissions)
        
        # Update metrics
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
    
    def call(self, inputs, training=None):
        """
        Forward pass.
        During training: returns emissions
        During inference: returns viterbi decoded sequence
        """
        emissions = self.emissions_output(inputs, training=training)
        if training:
            return emissions
        else:
            return self.crf_layer.viterbi_decode(emissions)

def build_crf_model(window_size=60, n_features=9, output_steps=15):
    """
    Build the CRF-enhanced model for phase prediction.
    Use this exactly like your original model!
    """
    inp = Input(shape=(window_size, n_features))
    
    # Multi-scale convolutions
    c1 = Conv1D(64, 3, padding='same', activation='relu')(inp)
    c1 = BatchNormalization()(c1)
    
    c2 = Conv1D(64, 5, padding='same', activation='relu')(inp)
    c2 = BatchNormalization()(c2)
    
    c3 = Conv1D(64, 7, padding='same', activation='relu')(inp)
    c3 = BatchNormalization()(c3)
    
    x = Concatenate()([c1, c2, c3])  # (batch, 60, 192)
    x = Dropout(0.2)(x)
    
    # Temporal modeling
    x = Bidirectional(LSTM(64, return_sequences=True))(x)  # (batch, 60, 128)
    x = Dropout(0.3)(x)
    
    # Keep only the last output_steps timesteps
    x = Lambda(lambda t: t[:, -output_steps:, :])(x)  # (batch, 15, 128)
    
    # Emission scores (unnormalized log probabilities)
    emissions = TimeDistributed(Dense(NUM_CLASSES, activation='linear'))(x)  # (batch, 15, num_classes)
    
    # Create emissions model (for computing emissions)
    emissions_model = Model(inp, emissions)
    
    # CRF layer for constrained decoding
    crf_layer = CRFLayer(NUM_CLASSES)
    predictions = crf_layer(emissions)  # (batch, 15)
    
    # Create the full model
    model = CRFModel(
        inputs=inp,
        outputs=predictions,
        crf_layer=crf_layer,
        emissions_output=emissions_model
    )
    
    return model

