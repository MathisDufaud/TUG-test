import os
import sys
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.ion()

from SaraFolder.settings import classes, running_settings


def error_duration_compute(result, gt, phases):
    errors_duration = {}
    for i in range(len(phases) - 1):
        phase_start = phases[i]
        phase_end = phases[i + 1]
        if phase_start in result and phase_end in result and phase_start in gt and phase_end in gt:
            duration_result = result[phase_end] - result[phase_start]
            duration_gt = gt[phase_end] - gt[phase_start]
            errors_duration[f"{phase_start}_to_{phase_end}"] = duration_result - duration_gt

    test_start = phases[0]
    test_end = phases[-1]
    if test_start in result and test_end in result and test_start in gt and test_end in gt:
        total_duration_result = result[test_end] - result[test_start]
        total_duration_gt = gt[test_end] - gt[test_start]
        errors_duration["total_duration"] = total_duration_result - total_duration_gt
    return errors_duration


def phases_eval(all_results, gt_dict):
    """
    Evaluate phase-by-phase timing errors.
    Returns dict with errors per phase and individual.
    """
    indiv_errors = defaultdict(lambda: defaultdict(list))
    indiv_errors_duration = defaultdict(lambda: defaultdict(list))
    phases = ['t_start', 't_end_stand', 't_start_turn',
              't_end_turn', 't_start_turn2', 't_start_sit', 't_end']

    for key, result in all_results.items():
        indiv_id, test_iter, clean_key = parse_key(key)
        if key not in gt_dict or isinstance(result, str):
            continue

        # remove unwanted key
        result = {k: v for k, v in result.items() if k != "t_end_turn2"}
        if isinstance(gt_dict[key], dict):
            gt = gt_dict[key]

            # compute errors per phase (timestamp)
            errors = {phase: result[phase] - gt[phase] for phase in result.keys() if phase in gt}
            # compute errors per phase (duration)
            errors_duration = error_duration_compute(result, gt, phases)

            indiv_errors[indiv_id][test_iter].append(errors)
            indiv_errors_duration[indiv_id][test_iter].append(errors_duration)
        else:
            gt = gt_dict[key]
            if gt is not None:
                errors_duration = {'total_duration': (result['t_end'] - result['t_start']) - gt/1000}
                indiv_errors_duration[indiv_id][test_iter].append(errors_duration)

    return indiv_errors, indiv_errors_duration


def define_res_gts(all_tests, gttype):
    all_results = {}
    all_gts = {}
    for test in all_tests:
        if test.results is not None and test.results['labelling'] is not None:
            all_results[str(test.user_id) + '_' + str(test.session_id) + '_' + test.context[0]] = test.results[
                'labelling']
            if test.gt_phases.t_end is not None:
                all_gts[
                    str(test.user_id) + '_' + str(test.session_id) + '_' + test.context[0]] = test.gt_phases.to_dict()
            else:
                # TODO Careful here, provide both analysis!!!
                if gttype == 'gwalk':
                    all_gts[str(test.user_id) + '_' + str(test.session_id) + '_' + test.context[0]] = test.gt_total_gwalk
                elif gttype == 'manual':
                    all_gts[str(test.user_id) + '_' + str(test.session_id) + '_' + test.context[0]] = test.gt_total_manual

    return all_results, all_gts


def evaluate_results(all_tests, eval_type, method, dataset, gttype, title):
    # TODO: A lot of skipped tests, some of them maybe are not that wrong?
    title = title+'_' + gttype
    all_results, all_gts = define_res_gts(all_tests, gttype=gttype)
    indiv_errors, indiv_errors_duration = phases_eval(all_results, all_gts)

    if eval_type == 'phases':

        if dataset == 'parkapp':
            res_path = running_settings.results_parkapp + \
                       os.sep + 'results'+title+'.txt'
        elif dataset == 'synergy':
            res_path = running_settings.results_synergy + \
                                os.sep + 'results'+title+'.txt'
        elif dataset:
            res_path = running_settings.results_pisatug + \
                                os.sep + 'results'+title+'.txt'
        elif dataset == 'all':
            res_path = running_settings.results_all + \
                                os.sep + 'results'+title+'.txt'

        # Logger start
        lg = classes.Logger(res_path)
        sys.stdout = lg

        print(f"Total individuals evaluated: {len(indiv_errors_duration)}")
        total_tests = sum(len(tests) for tests in indiv_errors_duration.values())
        print(f"Total tests evaluated: {total_tests}")

        # Run the aggregation
        results = aggregate_errors(indiv_errors_duration)

        # Convert to DataFrame for easier analysis
        df = create_error_dataframe(indiv_errors_duration)
        df.dropna(inplace=True)

        # Analyze patterns
        patterns = analyze_error_patterns(df)

        # Create visualizations
        plot_error_distributions(df, method=method, title=title)

        lg.stop_logging()
        sys.stdout = sys.__stdout__

    if eval_type == 'duration':
        # Run the aggregation
        results = aggregate_errors(indiv_errors_duration)

        # Convert to DataFrame for easier analysis
        df = create_error_dataframe(indiv_errors_duration)

        # Analyze patterns
        patterns = analyze_error_patterns(df)


        # Create visualizations
        plot_error_distributions(df, method=method, title=title)

        print(1)

    return None


def analyze_error_patterns(df):
    """
    Analyze error patterns and potential issues.
    """
    print("=== ERROR PATTERN ANALYSIS ===")

    # 1. Check for systematic bias by phase
    print("\n1. Systematic Bias by Phase:")
    phase_bias = df.groupby('phase')['error'].mean().sort_values(key=abs, ascending=False)
    for phase, bias in phase_bias.head().items():
        direction = "overestimation" if bias > 0 else "underestimation"
        print(f"  {phase}: {bias:.3f} ({direction})")

    # 2. Check for high variability phases
    print("\n2. Phases with Highest Variability:")
    phase_std = df.groupby('phase')['error'].std().sort_values(ascending=False)
    for phase, std in phase_std.head().items():
        print(f"  {phase}: std = {std:.3f}")

    # 3. Check for problematic individuals
    print("\n3. Individuals with Highest Error Rates:")
    individual_mae = df.groupby('individual_id')['abs_error'].mean().sort_values(ascending=False)
    for ind_id, mae in individual_mae.head().items():
        print(f"  Individual {ind_id}: MAE = {mae:.3f}")

    # 4. Check for iteration effects
    print("\n4. Iteration Effects (Learning/Fatigue):")
    iteration_mae = df.groupby('iteration_id')['abs_error'].mean().sort_values(ascending=False)
    for iter_id, mae in iteration_mae.head().items():
        print(f"  Iteration {iter_id}: MAE = {mae:.3f}")

    return {
        'phase_bias': phase_bias,
        'phase_variability': phase_std,
        'individual_errors': individual_mae,
        'iteration_effects': iteration_mae
    }


def export_error_analysis_to_excel(df, filename="error_labelling.xlsx"):
    """
    Analyze error patterns and save a detailed Excel report, combining phase bias and variability.

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain columns: 'phase', 'individual_id', 'iteration_id', 'error', 'abs_error'.
    filename : str
        Output Excel file path.

    Returns:
    --------
    None
    """
    # 1. Phase bias (mean error) and direction
    phase_bias = df.groupby('phase')['error'].mean()
    phase_bias_df = phase_bias.reset_index().rename(columns={'error': 'mean_error'})
    phase_bias_df['direction'] = phase_bias_df['mean_error'].apply(
        lambda x: 'overestimation' if x > 0 else 'underestimation')

    # 2. Phase variability (std) and limits of agreement
    phase_std = df.groupby('phase')['error'].std().reset_index().rename(columns={'error': 'std_error'})

    # Merge bias and std to get one combined table
    phase_summary_df = phase_bias_df.merge(phase_std, on='phase')
    phase_summary_df['LoA_lower'] = phase_summary_df['mean_error'] - 1.96 * phase_summary_df['std_error']
    phase_summary_df['LoA_upper'] = phase_summary_df['mean_error'] + 1.96 * phase_summary_df['std_error']

    # 3. Individual errors
    individual_mae = df.groupby('individual_id')['abs_error'].mean().sort_values(ascending=False)
    individual_mae_df = individual_mae.reset_index().rename(columns={'abs_error': 'MAE'})

    # 4. Iteration effects
    iteration_mae = df.groupby('iteration_id')['abs_error'].mean().sort_values(ascending=False)
    iteration_mae_df = iteration_mae.reset_index().rename(columns={'abs_error': 'MAE'})

    # Optional: full iteration stats including counts
    iteration_stats = df.groupby('iteration_id').agg(
        mean_error=('error', 'mean'),
        mean_abs_error=('abs_error', 'mean'),
        count=('iteration_id', 'count')
    ).reset_index()

    # Write to Excel
    with pd.ExcelWriter(running_settings.results_path + os.sep + filename) as writer:
        phase_summary_df.to_excel(writer, sheet_name="Phase Summary", index=False)
        individual_mae_df.to_excel(writer, sheet_name="Individual Errors", index=False)
        iteration_mae_df.to_excel(writer, sheet_name="Iteration MAE", index=False)
        iteration_stats.to_excel(writer, sheet_name="Iteration Stats", index=False)

    print(f"Error analysis exported to {filename}")

def plot_error_distributions(df, method, title):
    """
    Create visualizations of error distributions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Error distribution by phase
    df.boxplot(column='error', by='phase', ax=axes[0, 0], rot=45)
    axes[0, 0].set_title('Error Distribution by Phase')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('')
    axes[0, 0].set_ylabel('MAE')

    # 2. Absolute error by phase
    phase_mae = df.groupby('phase')['abs_error'].mean().sort_values(ascending=False)
    axes[0, 1].bar(range(len(phase_mae)), phase_mae.values)
    axes[0, 1].set_xticks(range(len(phase_mae)))
    axes[0, 1].set_xticklabels(phase_mae.index, rotation=45)
    axes[0, 1].set_title('Mean Absolute Error by Phase')
    axes[0, 1].set_ylabel('MAE')

    # 3. Individual error patterns
    df.boxplot(column='abs_error', by='individual_id', ax=axes[1, 0], rot=45)
    # Boxplot of individual MAE
    axes[1, 0].set_title('Distribution of Individual MAE')
    axes[1, 0].set_xlabel('Participants')
    axes[1, 0].set_ylabel('MAE')

    # 4. Error vs iteration
    iteration_stats = df.groupby('iteration_id').agg({
        'error': 'mean',
        'abs_error': 'mean',
        'iteration_id': 'count'  # count number of samples
    }).rename(columns={'iteration_id': 'count'}).reset_index()

    # Use scatter with size scaled by count
    df.boxplot(column='abs_error', by='dataset', ax=axes[1, 1], rot=45)

    # Horizontal black dashed line at 0
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.7)
    axes[1, 1].set_title('Distribution of dataset-MAE')
    axes[1, 1].set_xlabel('Dataset')
    axes[1, 1].set_ylabel('MAE')

    plt.tight_layout()
    plt.suptitle('')
    plt.savefig(running_settings.figures_parkapp + os.sep + 'eval' + title+'.jpg', dpi=400)
    plt.show()


def parse_key(key: str):
    """
    Extract (individual_id, test_iteration, base_key) from result key.
    Result keys may have suffix to remove (last 2 characters).
    """
    # remove suffix for comparison
    clean_key = key[:-2]
    parts = clean_key.split("_")
    if len(parts) < 2:
        warnings.warn(f"Unexpected key format: {key}", UserWarning)
        return None, None, clean_key
    indiv_id  = parts[0] + '_' + parts[1]
    test_iter = parts[2]
    return indiv_id, test_iter, clean_key


def aggregate_errors(indiv_errors_duration):
    """
    Comprehensive error aggregation for nested duration error dictionary.

    Args:
        indiv_errors_duration: Nested dict with structure:
            {individual_id: {test_iteration: [phase_errors_dict]}}

    Returns:
        Dictionary containing various aggregated error metrics
    """

    # Initialize results dictionary
    results = {}

    # 1. PHASE-LEVEL AGGREGATION
    print("=== PHASE-LEVEL ERROR AGGREGATION ===")
    phase_errors = defaultdict(list)

    for individual_id, iterations in indiv_errors_duration.items():
        for iteration_id, phase_data_list in iterations.items():
            for phase_data in phase_data_list:
                for phase_name, error_value in phase_data.items():
                    phase_errors[phase_name].append(error_value)

    # Calculate phase-level statistics
    phase_stats = {}
    for phase, errors in phase_errors.items():
        phase_stats[phase] = {
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors),
            'mae': np.mean(np.abs(errors)),  # Mean Absolute Error
            'rmse': np.sqrt(np.mean(np.array(errors) ** 2)),  # Root Mean Square Error
            'bias': np.mean(errors),  # Same as mean, but conceptually different
            'precision': np.std(errors),  # Variability around bias
            'count': len(errors),
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'q25': np.percentile(errors, 25),
            'q75': np.percentile(errors, 75)
        }

    results['phase_level'] = phase_stats

    # Print phase-level results
    for phase, stats in phase_stats.items():
        print(f"\n{phase}:")
        print(f"  Mean Error (Bias): {stats['mean_error']:.3f}")
        print(f"  MAE: {stats['mae']:.3f}")
        print(f"  RMSE: {stats['rmse']:.3f}")
        print(f"  Std (Precision): {stats['precision']:.3f}")

    # 2. INDIVIDUAL-LEVEL AGGREGATION
    print("\n=== INDIVIDUAL-LEVEL ERROR AGGREGATION ===")
    individual_stats = {}

    for individual_id, iterations in indiv_errors_duration.items():
        individual_errors = []
        individual_phase_errors = defaultdict(list)

        for iteration_id, phase_data_list in iterations.items():
            for phase_data in phase_data_list:
                for phase_name, error_value in phase_data.items():
                    individual_errors.append(error_value)
                    individual_phase_errors[phase_name].append(error_value)

        individual_stats[individual_id] = {
            'overall_mae': np.mean(np.abs(individual_errors)),
            'overall_rmse': np.sqrt(np.mean(np.array(individual_errors) ** 2)),
            'overall_bias': np.mean(individual_errors),
            'total_duration_errors': [phase_data.get('total_duration', 0)
                                      for iterations_dict in iterations.values()
                                      for phase_data in iterations_dict],
            'phase_specific': {phase: {
                'mae': np.mean(np.abs(errors)),
                'bias': np.mean(errors),
                'count': len(errors)
            } for phase, errors in individual_phase_errors.items()}
        }

    results['individual_level'] = individual_stats

    # Print top 5 individuals with highest errors
    individual_maes = [(ind_id, stats['overall_mae'])
                       for ind_id, stats in individual_stats.items()]
    individual_maes.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 5 individuals with highest MAE:")
    for ind_id, mae in individual_maes[:5]:
        print(f"  Individual {ind_id}: MAE = {mae:.3f}")

    # 3. ITERATION-LEVEL AGGREGATION
    print("\n=== ITERATION-LEVEL ERROR AGGREGATION ===")
    iteration_stats = defaultdict(lambda: defaultdict(list))

    for individual_id, iterations in indiv_errors_duration.items():
        for iteration_id, phase_data_list in iterations.items():
            for phase_data in phase_data_list:
                for phase_name, error_value in phase_data.items():
                    iteration_stats[iteration_id][phase_name].append(error_value)

    iteration_aggregated = {}
    for iteration_id, phases in iteration_stats.items():
        iteration_aggregated[iteration_id] = {}
        for phase_name, errors in phases.items():
            iteration_aggregated[iteration_id][phase_name] = {
                'mae': np.mean(np.abs(errors)),
                'bias': np.mean(errors),
                'rmse': np.sqrt(np.mean(np.array(errors) ** 2)),
                'count': len(errors)
            }

    results['iteration_level'] = iteration_aggregated

    # 4. OVERALL AGGREGATION
    print("\n=== OVERALL ERROR AGGREGATION ===")
    all_errors = []
    total_duration_errors = []

    for individual_id, iterations in indiv_errors_duration.items():
        for iteration_id, phase_data_list in iterations.items():
            for phase_data in phase_data_list:
                for phase_name, error_value in phase_data.items():
                    all_errors.append(error_value)
                    if phase_name == 'total_duration':
                        total_duration_errors.append(error_value)

    overall_stats = {
        'overall_mae': np.mean(np.abs(all_errors)),
        'overall_rmse': np.sqrt(np.mean(np.array(all_errors) ** 2)),
        'overall_bias': np.mean(all_errors),
        'overall_std': np.std(all_errors),
        'total_measurements': len(all_errors),
        'total_duration_mae': np.mean(np.abs(total_duration_errors)),
        'total_duration_bias': np.mean(total_duration_errors),
        'total_duration_rmse': np.sqrt(np.mean(np.array(total_duration_errors) ** 2))
    }

    results['overall'] = overall_stats

    print(f"Overall MAE: {overall_stats['overall_mae']:.3f}")
    print(f"Overall RMSE: {overall_stats['overall_rmse']:.3f}")
    print(f"Overall Bias: {overall_stats['overall_bias']:.3f}")
    print(f"Total Duration MAE: {overall_stats['total_duration_mae']:.3f}")
    print(f"Total Duration Bias: {overall_stats['total_duration_bias']:.3f}")

    return results


def create_error_dataframe(indiv_errors_duration):
    """
    Convert nested dictionary to pandas DataFrame for easier analysis.
    """
    rows = []

    for individual_id, iterations in indiv_errors_duration.items():
        dataset = individual_id.split('_')[1]
        for iteration_id, phase_data_list in iterations.items():
            for phase_data in phase_data_list:
                for phase_name, error_value in phase_data.items():
                    rows.append({
                        'individual_id': individual_id,
                        'iteration_id': iteration_id,
                        'phase': phase_name,
                        'error': error_value,
                        'abs_error': abs(error_value),
                        'dataset': dataset
                    })

    return pd.DataFrame(rows)


