import os
import time
from typing import DefaultDict

import matplotlib
from scipy.stats import pearsonr, linregress, spearmanr

import numpy as np
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
plt.ion()
from SaraFolder.settings import running_settings, utils_labelling, utils_evaluation, utils_MLnew
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from scipy import stats as scipy_stats
def loading_previous_comments(title, type):

    if title in os.listdir(running_settings.results_all):
        df_tests = pd.read_csv(running_settings.results_all + os.sep + title, index_col=0)
        print(f"Loaded existing comments from {running_settings.results_all + os.sep + title}")
        if 'gtManualS' in df_tests.columns:
            # Drop column
            df_tests.drop('gtManualS', axis=1, inplace=True)

    else:
        df_tests = pd.DataFrame(columns=['comment', 'whichstrange', 'error tot gwalk ', 'error tot manual',
                                         'gt gwalk', 'gt manual', 'secStart', 'secEnd'])

    return df_tests

def observesingletests(all_tests, method, title):
    """
    Args:
        all_tests: classes TUG test.

    I want to plot each row data and be able to write a comment for that test that goes into the pd dataframe at each iteration

    Returns:
        a pandas dataframe with index the user_id + str(session_id), columns ['comments', 'whichstrangesignal']

    """
    df_tests = loading_previous_comments(title, type='alltests')

    for test in all_tests:
        indexid = test.user_id + '_' + str(test.session_id)

        if indexid in df_tests.index:
            print(f"Skipping {indexid}, already in dataframe.")
            continue

        test.data_quality_investigation(plot=True)

        plt.draw()
        plt.pause(0.5)  # Pause for 0.5 seconds

        gtg = test.gt_total_gwalk
        if gtg > 5000:
            gtg = gtg / 1000
        gtm = test.gt_total_manual
        if gtm is not None:
            if gtm > 5000:
                gtm = gtm / 1000
        else:
            gtm = 0

        if not isinstance(test.results[method], str):
            error_tot_duration_gwalk = (test.results[method]['t_end'] - test.results[method]['t_start']) - gtg
            error_tot_duration_manual = (test.results[method]['t_end'] - test.results[method]['t_start']) - gtm
            print(f"Error with gwalk: {np.round(error_tot_duration_gwalk, 2)}, error with manual: {np.round(error_tot_duration_manual, 2)}")
        else:
            print(f"Result: {test.results[method]}")
            error_tot_duration_gwalk = test.results[method]
            error_tot_duration_manual = test.results[method]

        print(f"GT gwalk: {gtg}, GT manual: {gtm}")

        # Get user input for comments
        comment = input(f"Enter comment for test {indexid}: ")
        which_strange = input(f"Which strange signal for test {indexid}: ")

        # Select with cursor the start and end of the TUG test (x axis of the figure)
        print("Click on the plot to select START point (x-axis)...")
        plt.draw()
        start_point = plt.ginput(1, timeout=0)  # Wait for 1 click, no timeout
        if start_point:
            secStart = start_point[0][0]  # Extract x-coordinate
            # Draw vertical line at start
            ax = plt.gca()
            line_start = ax.axvline(x=secStart, color='green', linestyle='--', linewidth=2, label='Start')
            plt.draw()
        else:
            secStart = None

        print("Click on the plot to select END point (x-axis)...")
        plt.draw()
        end_point = plt.ginput(1, timeout=0)  # Wait for 1 click, no timeout
        if end_point:
            secEnd = end_point[0][0]  # Extract x-coordinate
            # Draw vertical line at end
            line_end = ax.axvline(x=secEnd, color='red', linestyle='--', linewidth=2, label='End')
            plt.legend()
            plt.draw()
            plt.pause(1)  # Show the selected points for 1 second
        else:
            secEnd = None

        # Add row to dataframe
        df_tests.loc[indexid] = [comment, which_strange, error_tot_duration_gwalk, error_tot_duration_manual, gtg, gtm,
                                 secStart, secEnd]

        print(f"Added comments for {indexid} - Start: {secStart}, End: {secEnd}\n")
        plt.close('all')

        # Save dataframe to CSV
    df_tests['gtManualS'] = df_tests['secEnd'] - df_tests['secStart']
    csvpath = running_settings.results_all + os.sep + title
    df_tests.to_csv(csvpath, index=True)
    print(f"Saved comments to {csvpath}")

    return df_tests

def observesingletests_skipped(all_tests, method, title):
    """
    Args:
        all_tests: classes TUG test.

    I want to plot each row data and be able to write a comment for that test that goes into the pd dataframe at each iteration

    Returns:
        a pandas dataframe with index the user_id + str(session_id), columns ['comments', 'whichstrangesignal']

    """
    df_tests = loading_previous_comments(title, type='skipped')

    for k, test in all_tests.items():
        context = k[-1]
        if context == 'u':
            continue
        indexid = test.user_id + '_' + str(test.session_id)
        test.dataset_id = 'parkapp_skipped'
        if indexid in df_tests.index:
            print(f"Skipping {indexid}, already in dataframe.")
            continue

        # Plot data
        test.plot_raw_data()
        plt.title(k)
        plt.draw()
        plt.pause(0.5)  # Pause for 0.5 seconds

        # Get user input for comments
        comment = input(f"Enter comment for test {k}: ")

        # Select with cursor the start and end of the TUG test (x axis of the figure)
        # Select with cursor the start and end of the TUG test (x axis of the figure)
        print("Click on the plot to select START point (x-axis)...")
        plt.draw()
        start_point = plt.ginput(1, timeout=0)  # Wait for 1 click, no timeout
        if start_point:
            secStart = start_point[0][0]  # Extract x-coordinate
            # Draw vertical line at start
            ax = plt.gca()
            line_start = ax.axvline(x=secStart, color='green', linestyle='--', linewidth=2, label='Start')
            plt.draw()
        else:
            secStart = None

        print("Click on the plot to select END point (x-axis)...")
        plt.draw()
        end_point = plt.ginput(1, timeout=0)  # Wait for 1 click, no timeout
        if end_point:
            secEnd = end_point[0][0]  # Extract x-coordinate
            # Draw vertical line at end
            line_end = ax.axvline(x=secEnd, color='red', linestyle='--', linewidth=2, label='End')
            plt.legend()
            plt.draw()
            plt.pause(1)  # Show the selected points for 1 second
        else:
            secEnd = None

        # Add row to dataframe
        if len(df_tests.columns) == 4:
            df_tests.loc[indexid] = [comment, context, secStart, secEnd]
        else:
            df_tests.loc[indexid] = [comment, context, secStart, secEnd, 0]


        print(f"Added comments for {indexid} - Start: {secStart}, End: {secEnd}\n")
        plt.close('all')

        # Save dataframe to CSV
    df_tests['gtManualS'] = df_tests['secEnd'] - df_tests['secStart']
    csvpath = running_settings.results_all + os.sep + title
    df_tests.to_csv(csvpath, index=True)
    print(f"Saved comments to {csvpath}")

    return df_tests


def define_start_end(all_tests, method, title):
    """
    Args:
        all_tests: classes TUG test.

    I want to plot each row data and be able to write a comment for that test that goes into the pd dataframe at each iteration

    Returns:
        a pandas dataframe with index the user_id + str(session_id), columns ['comments', 'whichstrangesignal']

    """
    if title in os.listdir(running_settings.results_all):
        df_tests = pd.read_csv(running_settings.results_all + os.sep + title, index_col=0)
        print(f"Loaded existing comments from {running_settings.results_all + os.sep + title}")
    else:
        df_tests = pd.DataFrame(columns=['error tot gwalk ', 'error tot manual', 'gt gwalk', 'gt manual', 'secStart', 'secEnd'])

    for test in all_tests:
        indexid = test.user_id + '_' + str(test.session_id)

        if indexid in df_tests.index:
            print(f"Skipping {indexid}, already in dataframe.")
            continue

        # Plot data
        test.plot_raw_data()
        test.plot_labelling(method=method, plot=False)

        plt.draw()
        plt.pause(0.5)  # Pause for 0.5 seconds

        gtg = test.gt_total_gwalk
        if gtg > 5000:
            gtg = gtg / 1000
        gtm = test.gt_total_manual
        if gtm > 5000:
            gtm = gtm / 1000

        if not isinstance(test.results[method], str):
            error_tot_duration_gwalk = (test.results[method]['t_end'] - test.results[method]['t_start']) - gtg
            error_tot_duration_manual = (test.results[method]['t_end'] - test.results[method]['t_start']) - gtm
            print(f"Error with gwalk: {np.round(error_tot_duration_gwalk, 2)}, error with manual: {np.round(error_tot_duration_manual, 2)}")

        else:
            print(f"Result: {test.results[method]}")
            error_tot_duration_gwalk = test.results[method]
            error_tot_duration_manual = test.results[method]

        print(f"GT gwalk: {gtg}, GT manual: {gtm}")

        # Select with cursor the start and end of the TUG test (x axis of the figure)
        print("Click on the plot to select START point (x-axis)...")
        plt.draw()
        start_point = plt.ginput(1, timeout=0)  # Wait for 1 click, no timeout
        if start_point:
            secStart = start_point[0][0]  # Extract x-coordinate
            # Draw vertical line at start
            ax = plt.gca()
            line_start = ax.axvline(x=secStart, color='green', linestyle='--', linewidth=2, label='Start')
            plt.draw()
        else:
            secStart = None

        print("Click on the plot to select END point (x-axis)...")
        plt.draw()
        end_point = plt.ginput(1, timeout=0)  # Wait for 1 click, no timeout
        if end_point:
            secEnd = end_point[0][0]  # Extract x-coordinate
            # Draw vertical line at end
            line_end = ax.axvline(x=secEnd, color='red', linestyle='--', linewidth=2, label='End')
            plt.legend()
            plt.draw()
            plt.pause(1)  # Show the selected points for 1 second
        else:
            secEnd = None

        # Add row to dataframe
        df_tests.loc[indexid] = [error_tot_duration_gwalk, error_tot_duration_manual, gtg, gtm, secStart, secEnd]

        print(f"Added comments for {indexid} - Start: {secStart}, End: {secEnd}\n")
        plt.close('all')

        # Save dataframe to CSV
    df_tests['gtManualS'] = df_tests['secEnd'] - df_tests['secStart']
    csvpath = running_settings.results_all + os.sep + title
    df_tests.to_csv(csvpath, index=True)
    print(f"Saved comments to {csvpath}")

    return df_tests



def dict_to_plot_stats(all_tests, stats_of_interest):
    dict_to_plot = {}
    dict_to_plot['dataset'] = []
    dict_to_plot['gtGwalk'] = []
    dict_to_plot['gtManual'] = []
    dict_to_plot['estimation'] = []
    dict_to_plot['estDuration'] = []

    for test in all_tests:
        dict_to_plot['dataset'].append(test.dataset_id)
        dict_to_plot['gtGwalk'].append(test.gt_total_gwalk*1000 if test.gt_total_gwalk < 5000 else test.gt_total_gwalk)
        dict_to_plot['gtManual'].append(test.gt_total_manual*1000 if test.gt_total_manual is not None and test.gt_total_manual < 5000 else test.gt_total_manual)
        dict_to_plot['estimation'].append(test.results['labelling'] if not isinstance(test.results['labelling'], str) else None)
        dict_to_plot['estDuration'].append((test.results['labelling']['t_end'] - test.results['labelling']['t_start']) if not isinstance(test.results['labelling'], str) else None)
        for stat in stats_of_interest:
            if stat not in dict_to_plot:
                dict_to_plot[stat] = []
            dict_to_plot[stat].append(test.quality['statssignal'][stat])

    df_plot = pd.DataFrame(dict_to_plot)
    return df_plot


def compute_stats_tests(all_tests, df_tests, plot=False):
    for test in all_tests:
        test.data_quality_investigation(plot=plot, df_tests=df_tests)
        if plot:
            # Add a break/pause in the code that I can unblock only by clicking 'Enter' in my keyboard
            input("Press Enter to continue to next plot...")
    pass

def compute_error_tests(all_tests, method):
    all_results, all_gts = utils_evaluation.define_res_gts(all_tests, gttype='gwalk', method=method)
    indiv_errors, indiv_errors_duration = utils_evaluation.phases_eval(all_results, all_gts)

    all_errors = {}
    for i, k in enumerate(all_results.keys()):
        print("Computing error for test: ", k)
        key = k[:-2]
        duration = all_results[k]['t_end'] - all_results[k]['t_start']
        if duration > 1000:
            duration = duration / 1000
        gt = all_gts[k]
        if isinstance(gt, dict): # TODO careful here NO MANUAL GT!
            gt = gt['t_end'] - gt['t_start']
        if gt>1000:
            gt=gt/1000
        all_errors[key] = duration - gt
        all_tests[i].error[method] = duration - gt


    return all_errors, indiv_errors_duration

def compute_tests_stats(all_tests, stats_of_interest):

    compute_stats_tests(all_tests)

    if True:
        # Plot statistics across all tests
        df_plot = dict_to_plot_stats(all_tests, stats_of_interest)

        fig = plt.figure(figsize=(15, 10))
        for i, stat in enumerate(stats_of_interest):
            plt.subplot(3, 3, i+1)
            for dataset in df_plot['dataset'].unique():
                subset = df_plot[df_plot['dataset'] == dataset]
                bins = 20
                plt.hist(subset[stat], alpha=0.5, bins=bins, label=dataset)
            plt.title(stat)
            plt.xlabel(stat)
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            plt.show()

        fig = plt.figure(figsize=(15, 10))
        for i, stat in enumerate(stats_of_interest):
            plt.subplot(3, 3, i + 1)

            # Plot each dataset separately
            for dataset in df_plot['dataset'].unique():
                data = df_plot[df_plot['dataset'] == dataset]
                plt.plot(data.index, data[stat], 'o', alpha=0.5, label=dataset)

            plt.title(stat)
            plt.xlabel('Index')
            plt.ylabel(stat)
            plt.legend()

        plt.tight_layout()
        plt.show()

    return all_tests


def calc_qualityscore(dict_stats, df):
    # 8. COMPOSITE QUALITY SCORE
    # Lower score = worse quality
    quality_score = 0

    # Penalize flat signals (low CV)
    if dict_stats['cv_acc'] < 0.01:
        quality_score -= 5

    # Penalize noisy signals (low SNR, high zero-crossing rate)
    if dict_stats['snr_acc'] < 10:
        quality_score -= 3
    if dict_stats['zcr_acc'] > 0.5:
        quality_score -= 3

    # Reward good signals
    if dict_stats['entropy_acc'] > 2.0 and dict_stats['std_acc'] > 0.1:
        quality_score += 5

    return float(quality_score)


def compute_test_stats(df):
    dict_stats = {}

    # Helper function to calculate entropy
    def calculate_entropy(series, bins=10):
        """Calculate Shannon entropy of a pandas Series"""
        counts, _ = np.histogram(series.dropna(), bins=bins)
        probabilities = counts[counts > 0] / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities))

    dict_stats['entropy_acc'] = float(calculate_entropy(df['sqrt(X²+Y²+Z²)']))
    dict_stats['std_acc'] = float(df['sqrt(X²+Y²+Z²)'].std())
    dict_stats['var_acc'] = float(df['sqrt(X²+Y²+Z²)'].var())
    dict_stats['median_acc'] = float(df['sqrt(X²+Y²+Z²)'].median())
    dict_stats['entropy_rotrate'] = float(calculate_entropy(df['rotRate_beta_gamma']))
    dict_stats['std_rotrate'] = float(df['rotRate_beta_gamma'].std())
    dict_stats['var_rotrate'] = float(df['rotRate_beta_gamma'].var())
    dict_stats['median_rotrate'] = float(df['rotRate_beta_gamma'].median())
    dict_stats['autocorr1sec_acc'] = float(df['sqrt(X²+Y²+Z²)'].autocorr(lag=60))
    dict_stats['autocorr1sec_alpha'] = float(df['alpha'].autocorr(lag=60))
    dict_stats['autocorr1sec_beta'] = float(df['beta'].autocorr(lag=60))
    dict_stats['autocorr2sec_beta'] = float(df['beta'].autocorr(lag=120))
    dict_stats['autocorr2sec_alpha'] = float(df['alpha'].autocorr(lag=120))
    dict_stats['autocorr2sec_acc'] = float(df['sqrt(X²+Y²+Z²)'].autocorr(lag=120))
    dict_stats['autocorr5sec_acc'] = float(df['sqrt(X²+Y²+Z²)'].autocorr(lag=300))
    dict_stats['autocorr5sec_alpha'] = float(df['alpha'].autocorr(lag=300))
    dict_stats['autocorr5sec_beta'] = float(df['beta'].autocorr(lag=300))
    dict_stats['entropy_alpha'] = float(calculate_entropy(df['alpha']))
    dict_stats['entropy_beta'] = float(calculate_entropy(df['beta']))

    # 2. FLATNESS INDICATORS (detect flat/constant signals)
    # Coefficient of variation (CV = std/mean) - low CV indicates flat signal
    dict_stats['cv_acc'] = float(df['sqrt(X²+Y²+Z²)'].std() / df['sqrt(X²+Y²+Z²)'].mean()) if df[
                                                                                                  'sqrt(X²+Y²+Z²)'].mean() != 0 else 0
    dict_stats['cv_rotrate'] = float(df['rotRate_beta_gamma'].std() / abs(df['rotRate_beta_gamma'].mean())) if df[
                                                                                                                   'rotRate_beta_gamma'].mean() != 0 else 0
    dict_stats['cv_alpha'] = float(df['alpha'].std() / abs(df['alpha'].mean())) if df['alpha'].mean() != 0 else 0
    dict_stats['cv_beta'] = float(df['beta'].std() / abs(df['beta'].mean())) if df['beta'].mean() != 0 else 0

    # Range as percentage of mean (normalized range)
    dict_stats['range_norm_acc'] = float(
        (df['sqrt(X²+Y²+Z²)'].max() - df['sqrt(X²+Y²+Z²)'].min()) / df['sqrt(X²+Y²+Z²)'].mean()) if df[
                                                                                                        'sqrt(X²+Y²+Z²)'].mean() != 0 else 0
    dict_stats['range_norm_rotrate'] = float(
        (df['rotRate_beta_gamma'].max() - df['rotRate_beta_gamma'].min()) / abs(df['rotRate_beta_gamma'].mean())) if df['rotRate_beta_gamma'].mean() != 0 else 0

    # 3. NOISE INDICATORS
    # Signal-to-Noise Ratio (SNR) approximation using signal power vs high-freq noise
    def estimate_snr(signal):
        """Estimate SNR by comparing signal variance to derivative variance."""
        signal_clean = signal.dropna()
        if len(signal_clean) < 2:
            return 0
        signal_power = signal_clean.var()
        noise_power = np.diff(signal_clean).var()  # High-frequency changes
        if noise_power == 0:
            return float('inf') if signal_power > 0 else 0
        return float(signal_power / noise_power)

    dict_stats['snr_acc'] = float(estimate_snr(df['sqrt(X²+Y²+Z²)']))
    dict_stats['snr_rotrate'] = float(estimate_snr(df['rotRate_beta_gamma']))
    dict_stats['snr_alpha'] = float(estimate_snr(df['alpha']))
    dict_stats['snr_beta'] = float(estimate_snr(df['beta']))

    # Zero-crossing rate (normalized) - high rate can indicate noise
    def zero_crossing_rate(signal):
        """Calculate zero-crossing rate."""
        signal_clean = signal.dropna()
        if len(signal_clean) < 2:
            return 0
        signal_centered = signal_clean - signal_clean.mean()
        crossings = np.sum(np.diff(np.sign(signal_centered)) != 0)
        return float(crossings / len(signal_clean))

    dict_stats['zcr_acc'] = zero_crossing_rate(df['sqrt(X²+Y²+Z²)'])
    dict_stats['zcr_rotrate'] = zero_crossing_rate(df['rotRate_beta_gamma'])
    dict_stats['zcr_alpha'] = zero_crossing_rate(df['alpha'])
    dict_stats['zcr_beta'] = zero_crossing_rate(df['beta'])

    # 4. SIGNAL COMPLEXITY
    # Interquartile range (IQR) - measure of spread
    dict_stats['iqr_acc'] = float(df['sqrt(X²+Y²+Z²)'].quantile(0.75) - df['sqrt(X²+Y²+Z²)'].quantile(0.25))
    dict_stats['iqr_rotrate'] = float(df['rotRate_beta_gamma'].quantile(0.75) - df['rotRate_beta_gamma'].quantile(0.25))
    dict_stats['iqr_alpha'] = float(df['alpha'].quantile(0.75) - df['alpha'].quantile(0.25))
    dict_stats['iqr_beta'] = float(df['beta'].quantile(0.75) - df['beta'].quantile(0.25))

    # 7. PEAK STATISTICS (can help identify structured vs noisy signals)
    def peak_statistics(signal):
        """Calculate number and properties of peaks."""
        signal_clean = signal.dropna().values
        if len(signal_clean) < 3:
            return 0, 0
        peaks, properties = find_peaks(signal_clean, prominence=signal_clean.std() * 0.5)
        num_peaks = len(peaks)
        peak_density = num_peaks / len(signal_clean)  # Peaks per sample
        return float(num_peaks), float(peak_density)

    dict_stats['num_peaks_acc'], dict_stats['peak_density_acc'] = peak_statistics(df['sqrt(X²+Y²+Z²)'])
    dict_stats['num_peaks_rotrate'], dict_stats['peak_density_rotrate'] = peak_statistics(df['rotRate_beta_gamma'])

    dict_stats['quality_score'] = calc_qualityscore(dict_stats, df)

    # Sample entropy
    def sample_entropy(signal, m=2, r=None):
        """Calculate sample entropy of a time series."""
        signal = np.array(signal.dropna())
        if len(signal) < m + 1:
            return np.nan
        if r is None:
            r = 0.2 * np.std(signal)
        N = len(signal)

        def _phi(m):
            x = np.array([signal[i:i + m] for i in range(N - m + 1)])
            C = np.sum([np.sum(np.max(np.abs(x - xi), axis=1) <= r) - 1 for xi in x])
            return C / ((N - m + 1) * (N - m))

        return -np.log(_phi(m + 1) / _phi(m)) if _phi(m) != 0 else np.nan

    dict_stats['sampen_acc'] = sample_entropy(df['sqrt(X²+Y²+Z²)'])
    dict_stats['sampen_alpha'] = sample_entropy(df['alpha'])
    dict_stats['sampen_alpha'] = sample_entropy(df['beta'])

    # 3. Lempel–Ziv Complexity (LZC)
    # LZC quantifies the diversity of patterns in a sequence — higher values mean a more complex and less repetitive signal.
    def lempel_ziv_complexity(signal):
        """Compute Lempel–Ziv complexity for a 1D signal."""
        signal = signal.dropna()
        if len(signal) < 10:
            return np.nan
        # Binarize around median
        median_val = np.median(signal)
        binary_seq = ''.join(['1' if x > median_val else '0' for x in signal])
        i, c, s = 0, 1, binary_seq[0]
        for j in range(1, len(binary_seq)):
            if binary_seq[j] not in s:
                c += 1
                s = binary_seq[:j + 1]
            i += 1
        return float(c / len(binary_seq))

    dict_stats['lzc_acc'] = lempel_ziv_complexity(df['sqrt(X²+Y²+Z²)'])
    dict_stats['lzc_beta'] = lempel_ziv_complexity(df['beta'])
    dict_stats['lzc_alpha'] = lempel_ziv_complexity(df['alpha'])

    return dict_stats


def normalize_dfplot(df_plot, stats_of_interest):
    # Normalizing features
    for stat in stats_of_interest:
        # 0-1 normalization
        min_val = df_plot[stat].min()
        max_val = df_plot[stat].max()
        df_plot[stat] = (df_plot[stat] - min_val) / (max_val - min_val)

    return df_plot


def error_vs_stats(all_tests, stats_of_interest):

    df_plot = dict_to_plot_stats(all_tests, stats_of_interest)

    # Normalize features
    df_plot = normalize_dfplot(df_plot, stats_of_interest)

    if True:
        correlation_results = {}
        # Analysis of correlation of error with statistics
        for col in df_plot.columns:
            if col in stats_of_interest:
                # Computing correlation between columns
                correlation_gwalk = (abs(df_plot['gtGwalk'] - df_plot['estDuration'])).corr(df_plot[col])
                correlation_manual = (abs(df_plot['gtManual'] - df_plot['estDuration'])).corr(df_plot[col])
                correlation_results[col] = {'gwalk': correlation_gwalk, 'manual': correlation_manual}

                if abs(correlation_gwalk) > 0.3 or abs(correlation_manual) > 0.3:
                    print(f"High correlation found for {col}: gwalk={correlation_gwalk}, manual={correlation_manual}")
                    fig = plt.figure(figsize=(10, 5))
                    plt.subplot(1, 1, 1)
                    plt.scatter(df_plot[col], abs(df_plot['gtGwalk'] - df_plot['estDuration']), alpha=0.5, label='gwalk error')

                    # plt.scatter(df_plot[col], abs(df_plot['gtManual'] - df_plot['estDuration']), alpha=0.5, label='manual error', color='orange')
                    plt.title(f'Error vs {col}, corr gwalk: {np.round(correlation_gwalk,2)}')
                    plt.xlabel(col)
                    plt.ylabel('Absolute Error')
                    plt.legend()
                    plt.show()


    return None


def check_signalvariability(df, cols, threshold):
    # Signal variability - check if there's meaningful variation in the data
    acc_std = df[cols].std().mean()
    if acc_std < threshold:  # Threshold for minimal variation
        quality = 'Low signal variability/'
        print(f"{quality} for {cols} with threshold {threshold}")
        return quality
    else:
        return ''


def quality_assessment(df, dataset_id):
    df, quality = utils_labelling.check_emptiness_3sec(df, dataset_id)
    if isinstance(df, str):
        return df, quality

    if False:
        quality += check_signalvariability(df, cols=['acc.x', 'acc.y', 'acc.z'], threshold=0.8)
        quality += check_signalvariability(df, cols=['sqrt(X²+Y²+Z²)'], threshold=0.8)
        quality += check_signalvariability(df, cols=['alpha', 'beta', 'gamma'], threshold=0.5)
        quality += check_signalvariability(df, cols=['rotRate.alpha', 'rotRate.beta', 'rotRate.gamma'], threshold=0.5)

    # if 'Low signal variability' in quality:
    if False:
        _= utils_labelling.plot_faulty_signal(df, 'quality check')

    # Sampling frequency - not less than 30 samples/second
    # Calculate time span in seconds
    time_span = (df['msFromStart'].max() - df['msFromStart'].min()) / 1000.0
    if time_span > 0:
        sampling_freq = len(df) / time_span
        if sampling_freq < 30:
            quality += 'Not valid sampling frequency/'

    # Test duration - not less than 3 seconds, not more than 50 seconds
    duration = (df['msFromStart'].max() - df['msFromStart'].min()) / 1000.0
    if duration < 3:
        quality += 'Too short duration/'
    elif duration > 25:
        quality += 'Too long duration/'

    return quality

def get_quality_all(all_tests, method):
    quality_all = {}
    for test in all_tests:
        testid = test.user_id + '_' + str(test.session_id)
        quality_all[testid] = test.quality['basic'] + test.quality[method] + test.quality['visual']
    return quality_all

def observe_qualityvariable(all_tests, method='labelling'):

    quality_all = get_quality_all(all_tests, method)

    summarize_quality_issues(quality_all)
    low_var_datasets = get_datasets_with_issue(quality_all, 'Low signal variability')
    fig = plot_quality_overview(quality_all)
    plt.show()
    comparison = compare_sources(quality_all)
    return None

def parse_quality_issues(quality_all):
    """
    Parse quality dictionary and extract issue types for each dataset.

    Parameters:
    -----------
    quality_all : dict
        Dictionary with dataset_id as keys and quality strings as values

    Returns:
    --------
    pd.DataFrame with columns: dataset_id, dataset_source, all_issues, issue_list
    """
    records = []

    for dataset_id, quality_str in quality_all.items():
        # Extract dataset source (synergy, pisa, parkapp)
        source = dataset_id.split('_')[1] if '_' in dataset_id else 'unknown'

        # Split quality string by '/' and filter out empty strings and 'okresults'
        issues = [issue.strip() for issue in quality_str.split('/')
                  if issue.strip() and issue.strip() != 'okresults']

        records.append({
            'dataset_id': dataset_id,
            'dataset_source': source,
            'all_issues': quality_str,
            'issue_list': issues,
            'num_issues': len(issues),
            'has_issues': len(issues) > 0
        })

    return pd.DataFrame(records)


def summarize_quality_issues(quality_all):
    """
    Print summary statistics of quality issues.
    """
    df = parse_quality_issues(quality_all)

    print("=" * 70)
    print("QUALITY ASSESSMENT SUMMARY")
    print("=" * 70)
    print(f"\nTotal datasets: {len(df)}")
    print(f"Datasets with issues: {df['has_issues'].sum()} ({df['has_issues'].sum() / len(df) * 100:.1f}%)")
    print(f"Datasets without issues: {(~df['has_issues']).sum()} ({(~df['has_issues']).sum() / len(df) * 100:.1f}%)")

    print("\n" + "-" * 70)
    print("BREAKDOWN BY SOURCE")
    print("-" * 70)
    source_summary = df.groupby('dataset_source').agg({
        'dataset_id': 'count',
        'has_issues': 'sum'
    }).rename(columns={'dataset_id': 'total', 'has_issues': 'with_issues'})
    source_summary['without_issues'] = source_summary['total'] - source_summary['with_issues']
    source_summary['issue_rate_%'] = (source_summary['with_issues'] / source_summary['total'] * 100).round(1)
    print(source_summary)

    print("\n" + "-" * 70)
    print("MOST COMMON ISSUES")
    print("-" * 70)
    all_issues = []
    for issues in df['issue_list']:
        all_issues.extend(issues)

    issue_counts = Counter(all_issues)
    for issue, count in issue_counts.most_common():
        print(f"{count:3d} ({count / len(df) * 100:5.1f}%) - {issue}")

    return df


def plot_quality_overview(quality_all, figsize=(14, 10)):
    """
    Create comprehensive visualization of quality issues.
    """
    df = parse_quality_issues(quality_all)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Quality Assessment Overview', fontsize=16, fontweight='bold')

    # 1. Overall pass/fail pie chart
    ax1 = axes[0, 0]
    pass_fail = df['has_issues'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    labels = ['No Issues', 'Has Issues']
    ax1.pie([pass_fail.get(False, 0), pass_fail.get(True, 0)],
            labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Overall Dataset Quality')

    # 2. Issues by source
    ax2 = axes[0, 1]
    source_issues = df.groupby(['dataset_source', 'has_issues']).size().unstack(fill_value=0)
    source_issues.plot(kind='bar', stacked=True, ax=ax2, color=['#2ecc71', '#e74c3c'])
    ax2.set_title('Quality by Data Source')
    ax2.set_xlabel('Data Source')
    ax2.set_ylabel('Number of Datasets')
    ax2.legend(['No Issues', 'Has Issues'], loc='upper right')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

    # 3. Issue type frequency
    ax3 = axes[1, 0]
    all_issues = []
    for issues in df['issue_list']:
        all_issues.extend(issues)

    issue_counts = Counter(all_issues)
    issue_df = pd.DataFrame(issue_counts.most_common(), columns=['Issue', 'Count'])

    if len(issue_df) > 0:
        ax3.barh(range(len(issue_df)), issue_df['Count'], color='#3498db')
        ax3.set_yticks(range(len(issue_df)))
        ax3.set_yticklabels(issue_df['Issue'], fontsize=9)
        ax3.set_xlabel('Frequency')
        ax3.set_title('Issue Type Distribution')
        ax3.invert_yaxis()

    # 4. Number of issues per dataset
    ax4 = axes[1, 1]
    issue_dist = df['num_issues'].value_counts().sort_index()
    ax4.bar(issue_dist.index, issue_dist.values, color='#9b59b6')
    ax4.set_xlabel('Number of Issues per Dataset')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Issue Count')
    ax4.set_xticks(range(int(df['num_issues'].max()) + 1))

    plt.tight_layout()
    return fig


def get_datasets_with_issue(quality_all, issue_keyword):
    """
    Find all datasets that have a specific issue.

    Parameters:
    -----------
    quality_all : dict
        Quality dictionary
    issue_keyword : str
        Keyword to search for (e.g., 'Low signal variability')

    Returns:
    --------
    list of dataset_ids that contain the issue
    """
    df = parse_quality_issues(quality_all)

    matching_datasets = []
    for _, row in df.iterrows():
        if any(issue_keyword.lower() in issue.lower() for issue in row['issue_list']):
            matching_datasets.append(row['dataset_id'])

    print(f"Found {len(matching_datasets)} datasets with '{issue_keyword}':")
    for ds in matching_datasets:
        print(f"  - {ds}")

    return matching_datasets


def compare_sources(quality_all):
    """
    Compare quality metrics across different data sources.
    """
    df = parse_quality_issues(quality_all)

    # Get all unique issues
    all_issues = set()
    for issues in df['issue_list']:
        all_issues.update(issues)

    # Create comparison matrix
    comparison = []
    for source in df['dataset_source'].unique():
        source_df = df[df['dataset_source'] == source]
        row = {'source': source, 'total': len(source_df)}

        for issue in all_issues:
            count = sum(any(issue in item for item in row['issue_list'])
                        for _, row in source_df.iterrows())
            row[issue] = count

        comparison.append(row)

    comparison_df = pd.DataFrame(comparison).set_index('source')

    print("\nIssue Frequency by Data Source:")
    print("=" * 70)
    print(comparison_df.to_string())

    return comparison_df


def get_stats_all(all_tests, method):
    quality_all = {}
    for test in all_tests:
        testid = test.user_id + '_' + str(test.session_id)
        quality_all[testid] = test.quality[method]
    return quality_all


def quality_stats(all_tests, method='labelling'):

    quality_all = get_quality_all(all_tests, method)
    stats_all = get_stats_all(all_tests, method)

    df = prepare_quality_stats_dataframe(quality_all, stats_all)

    key_stats = [s for s in df.columns if 'has' not in s and 'dataset' not in s and 'issue' not in s]

    fig1, df = plot_quality_stats_comparison(df, key_stats=key_stats)
    fig2, corr = plot_correlation_heatmap(df)
    results = statistical_comparison_by_issue(quality_all, stats_all)
    fig3 = plot_specific_issue_comparison(quality_all, stats_all, 'has_any_issue')

    return None


def plot_tests_witherror(all_tests, error_threshold, method):
    all_tests_left = []
    for test in all_tests:
        if abs(test.error[method]) > error_threshold:
            print(f"Error higher than {error_threshold}. Absolute error: {np.round(abs(test.error[method]),2)}, {test.user_id}_{test.session_id}")
            if method == 'labelling':
                test.plot_labelling(method=method, plot=True, show_info=False)
            if method == 'ml':
                utils_MLnew.plot_ml_prediction(test, test.error[method])
        else:
            all_tests_left.append(test)
            print(f"This test has not error higher than {error_threshold}, error: {test.error[method]}, {test.user_id}_{test.session_id}")
    return all_tests_left


def compare_error_all_indiv(error_all, indiv_error_duration):
    diffs = {}
    for k in error_all.keys():
        print("k", k)
        error_a = error_all[k]
        session = k.split('_')[2]
        k_i = k.split('_')[0] + '_' + k.split('_')[1]
        error_i = indiv_error_duration[k_i]

        if len(error_i)==1:
            key_one = list(error_i.keys())[0]
            # Comparing error_a and error_i
            diffs[k] = error_a - error_i[key_one][0]['total_duration']
        else:
            for e_i in error_i.keys():
                if session == e_i:
                    diffs[k] = error_a - error_i[e_i][0]['total_duration']
                    if diffs[k] != 0:
                        print("Problem here careful")


    pass


def investigate_stats_quality(all_tests, method='labelling'):
    """
    Investigate quality statistics across different datasets.

    Args:
        all_tests: List of test objects
        method: Method prefix for quality stats (default: 'labelling')
    """

    # Collect all quality data
    quality_data = []

    for test in all_tests:
        quality = test.quality[method + '_stats']

        # Extract user info
        user_parts = test.user_id.split('_')
        participant_id = user_parts[0]
        dataset = user_parts[1] if len(user_parts) > 1 else 'unknown'

        # Extract metrics for each timing variable
        for metric_name, metric_data in quality.items():
            quality_info = metric_data['quality']

            quality_data.append({
                'test_id': test.user_id + '_' + str(test.session_id),
                'participant_id': participant_id,
                'dataset': dataset,
                'session_id': test.session_id,
                'metric': metric_name,
                'value': metric_data['value'],
                'mean': quality_info['mean'],
                'std': quality_info['std'],
                'cv': quality_info['cv'],
                'max_deviation': quality_info['max_deviation'],
                'confidence': quality_info['confidence'],
                'agreement': quality_info['agreement'],
                'quality_score': quality_info['quality_score'],
                'n_valid': quality_info['n_valid'],
                'n_total': quality_info['n_total'],
                'inputs': metric_data['inputs']
            })

    df = pd.DataFrame(quality_data)

    # ===== OVERALL SUMMARY =====
    print("=" * 80)
    print("OVERALL QUALITY SUMMARY")
    print("=" * 80)
    print(f"Total tests analyzed: {len(all_tests)}")
    print(f"Total measurements: {len(df)}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Participants: {sorted(df['participant_id'].unique())}")
    print(f"Metrics analyzed: {sorted(df['metric'].unique())}")
    print()

    # ===== DATASET-SPECIFIC ANALYSIS =====
    print("=" * 80)
    print("ANALYSIS BY DATASET")
    print("=" * 80)

    for dataset in sorted(df['dataset'].unique()):
        dataset_df = df[df['dataset'] == dataset]

        print(f"\n{'─' * 80}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'─' * 80}")
        print(f"Number of tests: {len(dataset_df['test_id'].unique())}")
        print(f"Number of participants: {len(dataset_df['participant_id'].unique())}")
        print(f"Participants: {sorted(dataset_df['participant_id'].unique())}")
        print()

        # Quality metrics by timing variable
        print(f"Quality Metrics by Timing Variable:")
        print("-" * 80)

        summary_stats = dataset_df.groupby('metric').agg({
            'quality_score': ['mean', 'std', 'min', 'max'],
            'cv': ['mean', 'std', 'min', 'max'],
            'confidence': ['mean', 'min'],
            'agreement': ['mean', 'min'],
            'n_valid': ['mean', 'min', 'max'],
            'std': ['mean', 'max']
        }).round(4)

        print(summary_stats)
        print()

        # Identify problematic measurements
        print(f"Problematic Measurements (CV > 0.5 or Quality Score < 0.95):")
        print("-" * 80)
        problematic = dataset_df[
            (dataset_df['cv'] > 0.5) | (dataset_df['quality_score'] < 0.95)
            ]

        if len(problematic) > 0:
            for _, row in problematic.iterrows():
                print(f"  • Test: {row['test_id']}")
                print(f"    Metric: {row['metric']}")
                print(f"    Quality Score: {row['quality_score']:.4f}")
                print(f"    CV: {row['cv']:.4f}")
                print(f"    Std: {row['std']:.4f}")
                print(f"    Valid samples: {row['n_valid']}/{row['n_total']}")
                print(f"    Input values: {row['inputs']}")
                print()
        else:
            print("  ✓ No problematic measurements found!")
            print()

        # Sample variation analysis
        print(f"Sample Variation Analysis:")
        print("-" * 80)
        print(f"  Tests with 4/4 valid samples: {len(dataset_df[dataset_df['n_valid'] == 4])}")
        print(f"  Tests with 3/4 valid samples: {len(dataset_df[dataset_df['n_valid'] == 3])}")
        print(f"  Tests with 2/4 valid samples: {len(dataset_df[dataset_df['n_valid'] == 2])}")
        print(f"  Tests with <2/4 valid samples: {len(dataset_df[dataset_df['n_valid'] < 2])}")
        print()

    # ===== CROSS-DATASET COMPARISON =====
    print("\n" + "=" * 80)
    print("CROSS-DATASET COMPARISON")
    print("=" * 80)

    comparison = df.groupby(['dataset', 'metric']).agg({
        'quality_score': 'mean',
        'cv': 'mean',
        'confidence': 'mean',
        'n_valid': 'mean'
    }).round(4)

    print("\nAverage Quality Metrics by Dataset and Timing Variable:")
    print(comparison)
    print()

    # Statistical tests between datasets
    print("\nDataset Comparison Summary:")
    print("-" * 80)
    for metric in sorted(df['metric'].unique()):
        metric_df = df[df['metric'] == metric]
        print(f"\n{metric}:")
        for dataset in sorted(df['dataset'].unique()):
            dataset_metric = metric_df[metric_df['dataset'] == dataset]
            if len(dataset_metric) > 0:
                print(f"  {dataset}: Quality={dataset_metric['quality_score'].mean():.4f}, "
                      f"CV={dataset_metric['cv'].mean():.4f}, "
                      f"n={len(dataset_metric)}")

    # ===== VISUALIZATION =====
    create_quality_visualizations(df)

    return df


def create_quality_visualizations(df):
    """Create visualizations for quality analysis."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Quality Metrics Analysis by Dataset', fontsize=16, fontweight='bold')

    # 1. Quality Score by Dataset and Metric
    ax1 = axes[0, 0]
    pivot_quality = df.pivot_table(values='quality_score', index='metric', columns='dataset', aggfunc='mean')
    pivot_quality.plot(kind='bar', ax=ax1)
    ax1.set_title('Average Quality Score by Metric and Dataset')
    ax1.set_ylabel('Quality Score')
    ax1.set_xlabel('Timing Metric')
    ax1.legend(title='Dataset')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0.95, color='r', linestyle='--', label='Threshold (0.95)')

    # 2. Coefficient of Variation by Dataset
    ax2 = axes[0, 1]
    df_cv = df[df['cv'] < 2]  # Filter extreme outliers for visualization
    sns.boxplot(data=df_cv, x='dataset', y='cv', hue='metric', ax=ax2)
    ax2.set_title('Coefficient of Variation Distribution')
    ax2.set_ylabel('CV (lower is better)')
    ax2.set_xlabel('Dataset')
    ax2.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(axis='y', alpha=0.3)

    # 3. Valid Samples Distribution
    ax3 = axes[1, 0]
    valid_counts = df.groupby(['dataset', 'n_valid']).size().unstack(fill_value=0)
    valid_counts.plot(kind='bar', stacked=True, ax=ax3)
    ax3.set_title('Distribution of Valid Samples (out of 4)')
    ax3.set_ylabel('Count')
    ax3.set_xlabel('Dataset')
    ax3.legend(title='Valid Samples', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(axis='y', alpha=0.3)

    # 4. Agreement vs Confidence
    ax4 = axes[1, 1]
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        ax4.scatter(dataset_df['agreement'], dataset_df['confidence'],
                    label=dataset, alpha=0.6, s=50)
    ax4.set_title('Agreement vs Confidence by Dataset')
    ax4.set_xlabel('Agreement')
    ax4.set_ylabel('Confidence')
    ax4.legend(title='Dataset')
    ax4.grid(alpha=0.3)
    ax4.plot([0.95, 1], [0.95, 1], 'r--', alpha=0.5, label='Threshold')

    plt.tight_layout()
    plt.show()

    # Additional plot: CV distribution by metric
    fig2, ax = plt.subplots(figsize=(12, 6))
    df_cv_plot = df[df['cv'] < 2]  # Filter extreme outliers
    sns.violinplot(data=df_cv_plot, x='metric', y='cv', hue='dataset', ax=ax)
    ax.set_title('CV Distribution by Timing Metric and Dataset')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_xlabel('Timing Metric')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='High variation threshold')
    ax.legend(title='Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Usage example:
# df_quality = investigate_stats_quality(all_tests, method='labelling')

def quality_error(all_tests):
    method = 'labelling'  # darioalgo or labelling

    # Plot tests that have more than X seconds error
    utils_labelling.labelling_acrossall(all_tests, method=method, show_info=True)
    investigate_stats_quality(all_tests, method=method)
    error_all, indiv_error_duration = compute_error_tests(all_tests, method='labelling')
    compare_error_all_indiv(error_all, indiv_error_duration)
    plot_tests_witherror(all_tests, error_threshold=10, method='labelling')


    utils_labelling.labelling_acrossall(all_tests, method='darioalgo')
    error_all, indiv_error_duration = compute_error_tests(all_tests, method='darioalgo')
    plot_tests_witherror(all_tests, error_threshold=20, method='darioalgo')

    #

    if False:
        quality_all = get_quality_all(all_tests, method)

        df = prepare_quality_error_dataframe1(quality_all, error_all)

        # Select key statistics for visualization
        key_stats = ['entropy_acc',  'entropy_rotrate',
                     'autocorr2sec_acc', 'autocorr2sec_alpha',
                     'autocorr2sec_beta', 'entropy_alpha', 'entropy_beta',
                     'autocorr5sec_alpha', 'autocorr5sec_beta','autocorr5sec_acc','autocorr1sec_acc', 'autocorr1sec_alpha', 'autocorr1sec_beta']

        key_stats = [s for s in key_stats if s in df.columns]

        fig1, df = plot_error_by_quality(df)
        df_summary = error_summary_by_quality(df)
        fig2 = plot_error_distribution(df)
        corr_df = correlation_analysis(df)
        problematic = identify_problematic_datasets(df, error_threshold=1.5)

    return None


def prepare_quality_stats_dataframe(quality_all, stats_all):
    """
    Combine quality and stats dictionaries into a single DataFrame.

    Parameters:
    -----------
    quality_all : dict
        Dictionary with dataset_id as keys and quality strings as values
    stats_all : dict
        Dictionary with dataset_id as keys and statistics dictionaries as values

    Returns:
    --------
    pd.DataFrame with quality categories and statistics
    """
    records = []

    for dataset_id in quality_all.keys():
        if dataset_id not in stats_all:
            continue

        # Parse quality string
        quality_str = quality_all[dataset_id]
        issues = [issue.strip() for issue in quality_str.split('/')
                  if issue.strip() and issue.strip() != 'okresults' and issue.strip() != 'ok']

        # Create record
        record = {'dataset_id': dataset_id}

        # Add binary flags for each issue type
        record['has_low_signal_variability'] = any('Low signal variability' in issue for issue in issues)
        record['has_no_turns'] = any('No turns found' in issue for issue in issues)
        record['has_peaks_issue'] = any('peaks' in issue.lower() for issue in issues)
        record['has_any_issue'] = len(issues) > 0
        record['has_noisy'] = any('noisy' in issue.lower() or 'wavy' in issue.lower() for issue in issues)
        record['has_toolong'] = any('long' in issue.lower() for issue in issues)
        record['num_issues'] = len(issues)

        # Add all statistics
        record.update(stats_all[dataset_id])

        records.append(record)

    return pd.DataFrame(records)


def plot_quality_stats_comparison(df, key_stats, figsize=(16, 12)):
    """
    Create comprehensive visualization comparing quality categories with statistics.
    """
    fig, axes = plt.subplots(4, 4, figsize=figsize)
    fig.suptitle('Quality Categories vs Statistics Distribution #1', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    c=1
    idx_real = 0
    for idx, stat in enumerate(key_stats):
        print(f"Stat: {stat}, idx:{idx}, idx real {idx_real}")

        if idx >= 16*c:
            # Hide unused subplots
            for idx in range(len(key_stats), len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            plt.show()
            fig, axes = plt.subplots(4, 4, figsize=figsize)
            axes = axes.flatten()
            fig.suptitle(f'Quality Categories vs Statistics Distribution #{c}', fontsize=16, fontweight='bold')
            c+=1
            idx_real=0
            print(f"RESETTING Stat: {stat}, idx:{idx}, idx real {idx_real}")


        ax = axes[idx_real]
        idx_real+=1

        # Create boxplot comparing datasets with/without issues
        data_to_plot = [
            df[df['has_any_issue'] == False][stat].dropna(),
            df[df['has_any_issue'] == True][stat].dropna()
        ]

        bp = ax.boxplot(data_to_plot, labels=['No Issues', 'Has Issues'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')

        ax.set_ylabel(stat)
        ax.set_title(f'{stat}')
        ax.grid(True, alpha=0.3)

        # Add p-value from t-test
        if len(data_to_plot[0]) > 0 and len(data_to_plot[1]) > 0:
            t_stat, p_val = scipy_stats.ttest_ind(data_to_plot[0], data_to_plot[1])
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            ax.text(0.5, 0.95, f'p={p_val:.4f} {sig}',
                    transform=ax.transAxes, ha='center', va='top', fontsize=8, fontweight='bold')


    # Hide unused subplots
    for idx in range(len(key_stats), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig, df


def compute_correlation_matrix(df):
    """
    Compute correlation between quality flags and statistics.
    """

    # Get quality columns
    quality_cols = ['has_low_signal_variability', 'has_no_turns',
                    'has_peaks_issue', 'has_any_issue', 'num_issues']

    # Get statistic columns
    stat_cols = [col for col in df.columns if col not in
                 ['dataset_id'] + quality_cols]

    # Compute correlation matrix
    corr_matrix = df[quality_cols + stat_cols].corr()

    # Extract only quality vs stats correlations
    quality_stats_corr = corr_matrix.loc[stat_cols, quality_cols]

    return quality_stats_corr


def plot_correlation_heatmap(df, figsize=(12, 10)):
    """
    Plot heatmap of correlations between quality categories and statistics.
    """
    corr_matrix = compute_correlation_matrix(df)

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlation'})

    ax.set_title('Correlation between Quality Categories and Statistics',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Quality Categories', fontsize=12)
    ax.set_ylabel('Statistics', fontsize=12)

    plt.tight_layout()
    return fig, corr_matrix


def statistical_comparison_by_issue(quality_all, stats_all):
    """
    Perform statistical tests comparing statistics between datasets with/without each issue type.
    """
    df = prepare_quality_stats_dataframe(quality_all, stats_all)

    # Get statistic columns
    stat_cols = [col for col in df.columns if col not in
                 ['dataset_id', 'has_low_signal_variability', 'has_no_turns',
                  'has_peaks_issue', 'has_any_issue', 'num_issues']]

    # Issue types to test
    issue_types = {
        'Low Signal Variability': 'has_low_signal_variability',
        'No Turns Found': 'has_no_turns',
        'Peaks Issue': 'has_peaks_issue',
        'Any Issue': 'has_any_issue'
    }

    results = []

    for issue_name, issue_col in issue_types.items():
        n_with_issue = df[issue_col].sum()
        n_without_issue = (~df[issue_col]).sum()

        if n_with_issue == 0 or n_without_issue == 0:
            continue

        for stat in stat_cols:
            with_issue = df[df[issue_col] == True][stat].dropna()
            without_issue = df[df[issue_col] == False][stat].dropna()

            if len(with_issue) > 0 and len(without_issue) > 0:
                # T-test
                t_stat, p_val = scipy_stats.ttest_ind(with_issue, without_issue)

                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(with_issue) - 1) * with_issue.std() ** 2 +
                                      (len(without_issue) - 1) * without_issue.std() ** 2) /
                                     (len(with_issue) + len(without_issue) - 2))
                cohens_d = (with_issue.mean() - without_issue.mean()) / pooled_std if pooled_std > 0 else 0

                results.append({
                    'Issue Type': issue_name,
                    'Statistic': stat,
                    'Mean (With Issue)': with_issue.mean(),
                    'Mean (Without Issue)': without_issue.mean(),
                    'Difference': with_issue.mean() - without_issue.mean(),
                    'p-value': p_val,
                    'Cohen\'s d': cohens_d,
                    'Significant': 'Yes' if p_val < 0.05 else 'No'
                })

    results_df = pd.DataFrame(results).sort_values('p-value')

    return results_df


def plot_specific_issue_comparison(quality_all, stats_all, issue_type='has_low_signal_variability',
                                   top_n=8, figsize=(16, 10)):
    """
    Plot detailed comparison for a specific issue type.

    Parameters:
    -----------
    issue_type : str
        One of: 'has_low_signal_variability', 'has_no_turns', 'has_peaks_issue', 'has_any_issue'
    """
    df = prepare_quality_stats_dataframe(quality_all, stats_all)

    stat_cols = [col for col in df.columns if col not in
                 ['dataset_id', 'has_low_signal_variability', 'has_no_turns',
                  'has_peaks_issue', 'has_any_issue', 'num_issues']]

    # Calculate correlations for this issue type
    correlations = []
    for stat in stat_cols:
        corr = df[[issue_type, stat]].corr().iloc[0, 1]
        correlations.append((stat, abs(corr)))

    # Select top N statistics by absolute correlation
    top_stats = sorted(correlations, key=lambda x: x[1], reverse=True)[:top_n]
    top_stat_names = [s[0] for s in top_stats]

    # Create plots
    n_cols = 3
    n_rows = int(np.ceil(len(top_stat_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    issue_labels = {
        'has_no_turns': 'No Turns Found',
        'has_peaks_issue': 'Peaks Issue',
        'has_any_issue': 'Any Issue'
    }

    fig.suptitle(f'Top {top_n} Statistics Associated with: {issue_labels.get(issue_type, issue_type)}',
                 fontsize=16, fontweight='bold')

    for idx, stat in enumerate(top_stat_names):
        ax = axes[idx]

        data_to_plot = [
            df[df[issue_type] == False][stat].dropna(),
            df[df[issue_type] == True][stat].dropna()
        ]

        bp = ax.boxplot(data_to_plot, labels=['No', 'Yes'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')

        ax.set_ylabel(stat)
        ax.set_xlabel(f'{issue_labels.get(issue_type, issue_type)}')
        ax.set_title(f'{stat}')
        ax.grid(True, alpha=0.3)

        # Add statistics
        if len(data_to_plot[0]) > 0 and len(data_to_plot[1]) > 0:
            t_stat, p_val = scipy_stats.ttest_ind(data_to_plot[0], data_to_plot[1])
            corr = df[[issue_type, stat]].corr().iloc[0, 1]
            ax.text(0.5, 0.95, f'r={corr:.3f}, p={p_val:.4f}',
                    transform=ax.transAxes, ha='center', va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Hide unused subplots
    for idx in range(len(top_stat_names), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig

def prepare_quality_error_dataframe1(quality_all, error_all):
    """
    Combine quality and error dictionaries into a single DataFrame.

    Parameters:
    -----------
    quality_all : dict
        Dictionary with dataset_id as keys and quality strings as values
    error_all : dict
        Dictionary with dataset_id as keys and error values (float) as values

    Returns:
    --------
    pd.DataFrame with quality categories and errors
    """
    records = []

    for dataset_id in quality_all.keys():
        if dataset_id not in error_all:
            continue

        # Parse quality string
        quality_str = quality_all[dataset_id]
        source = dataset_id.split('_')[1] if '_' in dataset_id else 'unknown'

        if isinstance(quality_str, str):
            issues = [issue.strip() for issue in quality_str.split('/')
                      if issue.strip() and issue.strip() != 'okresults']

            # Create record
            record = {
                'dataset_id': dataset_id,
                'dataset_source': source,
                'error': error_all[dataset_id],
                'abs_error': abs(error_all[dataset_id])
            }

            # Add binary flags for each issue type
            record['has_low_signal_variability'] = any('Low signal variability' in issue for issue in issues)
            record['has_no_turns'] = any('No turns found' in issue for issue in issues)
            record['has_peaks_issue'] = any('peaks' in issue.lower() for issue in issues)
            record['has_any_issue'] = len(issues) > 0
            record['num_issues'] = len(issues)
            record['quality_string'] = quality_str

        if isinstance(quality_str, dict):
            # Extract dataset source
            record = {
                'dataset_id': dataset_id,
                'dataset_source': source,
                'error': error_all[dataset_id],
                'abs_error': abs(error_all[dataset_id]),
            }

            # Add to record all the values from quality_str
            for key, value in quality_str.items():
                record[key] = value

        records.append(record)

    return pd.DataFrame(records)


def plot_error_by_quality(df, figsize=(16, 10), proba=False):
    """
    Create comprehensive visualization comparing quality categories with errors.
    """
    if not proba:
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Error Analysis by Quality Categories', fontsize=16, fontweight='bold')

        # 1. Error distribution: with vs without issues
        ax = axes[0, 0]
        data_to_plot = [
            df[df['has_any_issue'] == False]['abs_error'].dropna(),
            df[df['has_any_issue'] == True]['abs_error'].dropna()
        ]
        bp = ax.boxplot(data_to_plot, labels=['No Issues', 'Has Issues'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Error by Overall Quality')
        ax.grid(True, alpha=0.3)

        if len(data_to_plot[0]) > 0 and len(data_to_plot[1]) > 0:
            t_stat, p_val = scipy_stats.ttest_ind(data_to_plot[0], data_to_plot[1])
            ax.text(0.5, 0.95, f'p={p_val:.4f}', transform=ax.transAxes,
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 2. Error by data source
        ax = axes[0, 1]
        sources = df['dataset_source'].unique()
        data_by_source = [df[df['dataset_source'] == src]['abs_error'].dropna() for src in sources]
        bp = ax.boxplot(data_by_source, labels=sources, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#3498db')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Error by Data Source')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 3. Scatter: Number of issues vs Error
        ax = axes[0, 2]
        ax.scatter(df['num_issues'], df['abs_error'], alpha=0.6, s=50)
        ax.set_xlabel('Number of Issues')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Error vs Number of Issues')
        ax.grid(True, alpha=0.3)

        # Add correlation
        if len(df) > 2:
            corr = df[['num_issues', 'abs_error']].corr().iloc[0, 1]
            ax.text(0.05, 0.95, f'r={corr:.3f}', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 4. Error by specific issue: Low signal variability
        ax = axes[1, 0]
        if df['has_low_signal_variability'].sum() > 0:
            data_to_plot = [
                df[df['has_low_signal_variability'] == False]['abs_error'].dropna(),
                df[df['has_low_signal_variability'] == True]['abs_error'].dropna()
            ]
            bp = ax.boxplot(data_to_plot, labels=['No', 'Yes'], patch_artist=True)
            bp['boxes'][0].set_facecolor('#2ecc71')
            bp['boxes'][1].set_facecolor('#e74c3c')
            ax.set_ylabel('Absolute Error')
            ax.set_title('Low Signal Variability')
            ax.grid(True, alpha=0.3)

            if len(data_to_plot[1]) > 0:
                t_stat, p_val = scipy_stats.ttest_ind(data_to_plot[0], data_to_plot[1])
                ax.text(0.5, 0.95, f'p={p_val:.4f}', transform=ax.transAxes,
                        ha='center', va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 5. Error by specific issue: No turns
        ax = axes[1, 1]
        if df['has_no_turns'].sum() > 0:
            data_to_plot = [
                df[df['has_no_turns'] == False]['abs_error'].dropna(),
                df[df['has_no_turns'] == True]['abs_error'].dropna()
            ]
            bp = ax.boxplot(data_to_plot, labels=['No', 'Yes'], patch_artist=True)
            bp['boxes'][0].set_facecolor('#2ecc71')
            bp['boxes'][1].set_facecolor('#e74c3c')
            ax.set_ylabel('Absolute Error')
            ax.set_title('No Turns Found')
            ax.grid(True, alpha=0.3)

            if len(data_to_plot[1]) > 0:
                t_stat, p_val = scipy_stats.ttest_ind(data_to_plot[0], data_to_plot[1])
                ax.text(0.5, 0.95, f'p={p_val:.4f}', transform=ax.transAxes,
                        ha='center', va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 6. Error by specific issue: Peaks issue
        ax = axes[1, 2]
        if df['has_peaks_issue'].sum() > 0:
            data_to_plot = [
                df[df['has_peaks_issue'] == False]['abs_error'].dropna(),
                df[df['has_peaks_issue'] == True]['abs_error'].dropna()
            ]
            bp = ax.boxplot(data_to_plot, labels=['No', 'Yes'], patch_artist=True)
            bp['boxes'][0].set_facecolor('#2ecc71')
            bp['boxes'][1].set_facecolor('#e74c3c')
            ax.set_ylabel('Absolute Error')
            ax.set_title('Peaks Issue')
            ax.grid(True, alpha=0.3)

            if len(data_to_plot[1]) > 0:
                t_stat, p_val = scipy_stats.ttest_ind(data_to_plot[0], data_to_plot[1])
                ax.text(0.5, 0.95, f'p={p_val:.4f}', transform=ax.transAxes,
                        ha='center', va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
    else:

        """
        Explore the relationship between error metrics (error, abs_error) 
        and quality statistics (mean, median, std, min, max, mean_entropy, derivative_std).

        Returns:
        --------
        fig : matplotlib figure object
        summary_df : dataframe with correlations to error/abs_error
        """
        quality_cols = [s for s in df.columns if 'error' not in s and 'dataset' not in s and 'issue' not in s]

        rows = []

        n = len(quality_cols)
        ncols = 3
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = axes.flatten()

        for idx, col in enumerate(quality_cols):

            if col not in df.columns:
                continue

            x = df[col].values
            err = df["error"].values
            abserr = df["abs_error"].values

            # Compute Pearson correlations
            pear_err, pear_err_p = pearsonr(x, err)
            pear_abs, pear_abs_p = pearsonr(x, abserr)

            # Store results for the summary dataframe
            rows.append({
                "metric": col,
                "pearson_error": pear_err,
                "pearson_error_pvalue": pear_err_p,
                "pearson_abs_error": pear_abs,
                "pearson_abs_error_pvalue": pear_abs_p,
            })

            # ---- Plotting ----
            ax = axes[idx]
            ax.scatter(x, err, alpha=0.6)

            # Regression line
            slope, intercept, _, _, _ = linregress(x, err)
            x_vals = np.linspace(np.min(x), np.max(x), 100)
            ax.plot(x_vals, slope * x_vals + intercept, linewidth=2)

            # Title with correlation
            ax.set_title(f"{col} | r = {pear_err:.3f}, p = {pear_err_p:.3f}")
            ax.set_xlabel(col)
            ax.set_ylabel("error")

        # Remove empty axes
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()

        summary_df = pd.DataFrame(rows)

    return fig, df


def error_summary_by_quality(df):
    """
    Generate summary statistics of errors grouped by quality categories.
    """

    print("=" * 80)
    print("ERROR ANALYSIS BY QUALITY CATEGORIES")
    print("=" * 80)

    # Overall statistics
    print("\n" + "-" * 80)
    print("OVERALL ERROR STATISTICS")
    print("-" * 80)
    print(f"Total datasets: {len(df)}")
    print(f"Mean error: {df['error'].mean():.4f}")
    print(f"Mean absolute error: {df['abs_error'].mean():.4f}")
    print(f"Std error: {df['error'].std():.4f}")
    print(f"Median absolute error: {df['abs_error'].median():.4f}")

    # By overall quality
    print("\n" + "-" * 80)
    print("ERROR BY OVERALL QUALITY STATUS")
    print("-" * 80)
    for has_issue in [False, True]:
        subset = df[df['has_any_issue'] == has_issue]
        label = "Has Issues" if has_issue else "No Issues"
        print(f"\n{label}:")
        print(f"  Count: {len(subset)}")
        print(f"  Mean error: {subset['error'].mean():.4f}")
        print(f"  Mean absolute error: {subset['abs_error'].mean():.4f}")
        print(f"  Std error: {subset['error'].std():.4f}")
        print(f"  Median absolute error: {subset['abs_error'].median():.4f}")

    # Statistical test
    if df['has_any_issue'].sum() > 0 and (~df['has_any_issue']).sum() > 0:
        no_issues = df[df['has_any_issue'] == False]['abs_error']
        has_issues = df[df['has_any_issue'] == True]['abs_error']
        t_stat, p_val = scipy_stats.ttest_ind(no_issues, has_issues)
        print(f"\n  T-test p-value: {p_val:.6f}")
        print(f"  Significant difference: {'Yes' if p_val < 0.05 else 'No'}")

    # By specific issues
    print("\n" + "-" * 80)
    print("ERROR BY SPECIFIC ISSUE TYPES")
    print("-" * 80)

    issue_types = [
        ('Low Signal Variability', 'has_low_signal_variability'),
        ('No Turns Found', 'has_no_turns'),
        ('Peaks Issue', 'has_peaks_issue')
    ]

    for issue_name, issue_col in issue_types:
        if df[issue_col].sum() > 0:
            print(f"\n{issue_name}:")
            subset = df[df[issue_col] == True]
            print(f"  Count: {len(subset)}")
            print(f"  Mean absolute error: {subset['abs_error'].mean():.4f}")
            print(f"  Median absolute error: {subset['abs_error'].median():.4f}")

            # T-test
            without = df[df[issue_col] == False]['abs_error']
            with_issue = df[df[issue_col] == True]['abs_error']
            if len(without) > 0 and len(with_issue) > 0:
                t_stat, p_val = scipy_stats.ttest_ind(without, with_issue)
                print(f"  T-test p-value: {p_val:.6f}")

    # By data source
    print("\n" + "-" * 80)
    print("ERROR BY DATA SOURCE")
    print("-" * 80)
    source_summary = df.groupby('dataset_source')['abs_error'].agg([
        'count', 'mean', 'std', 'median', 'min', 'max'
    ]).round(4)
    print(source_summary)

    return df


def plot_error_distribution(df, figsize=(14, 6)):
    """
    Plot error distributions with quality overlay.
    """

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Error Distribution Analysis', fontsize=14, fontweight='bold')

    # 1. Histogram with quality overlay
    ax = axes[0]
    no_issues = df[df['has_any_issue'] == False]['error']
    has_issues = df[df['has_any_issue'] == True]['error']

    ax.hist(no_issues, bins=20, alpha=0.6, label='No Issues', color='#2ecc71')
    ax.hist(has_issues, bins=20, alpha=0.6, label='Has Issues', color='#e74c3c')
    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution by Quality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # 2. Violin plot by number of issues
    ax = axes[1]
    issue_counts = sorted(df['num_issues'].unique())
    data_by_issues = [df[df['num_issues'] == n]['abs_error'].values for n in issue_counts]

    parts = ax.violinplot(data_by_issues, positions=issue_counts, widths=0.7,
                          showmeans=True, showmedians=True)
    ax.set_xlabel('Number of Issues')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error Distribution by Issue Count')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(issue_counts)

    plt.tight_layout()
    return fig


def correlation_analysis(df):
    """
    Compute correlations between quality flags and errors.
    """
    quality_cols = ['has_low_signal_variability', 'has_no_turns',
                    'has_peaks_issue', 'has_any_issue', 'num_issues']

    correlations = []
    for col in quality_cols:
        # Point-biserial correlation for binary variables
        if df[col].dtype == bool:
            corr, p_val = scipy_stats.pointbiserialr(df[col], df['abs_error'])
        else:
            corr, p_val = scipy_stats.pearsonr(df[col], df['abs_error'])

        correlations.append({
            'Quality Category': col.replace('has_', '').replace('_', ' ').title(),
            'Correlation with Abs Error': corr,
            'P-value': p_val,
            'Significant': 'Yes' if p_val < 0.05 else 'No'
        })

    corr_df = pd.DataFrame(correlations).sort_values('Correlation with Abs Error',
                                                     key=abs, ascending=False)

    print("\n" + "=" * 80)
    print("CORRELATION BETWEEN QUALITY CATEGORIES AND ERROR")
    print("=" * 80)
    print(corr_df.to_string(index=False))

    return corr_df


def identify_problematic_datasets(df, error_threshold=1.5):
    """
    Identify datasets with high errors and their quality issues.
    """

    high_error = df[df['abs_error'] > error_threshold].sort_values('abs_error', ascending=False)

    print("\n" + "=" * 80)
    print(f"DATASETS WITH ABSOLUTE ERROR > {error_threshold}")
    print("=" * 80)
    print(f"Found {len(high_error)} datasets\n")

    for _, row in high_error.iterrows():
        print(f"Dataset: {row['dataset_id']}")
        print(f"  Error: {row['error']:.4f} (abs: {row['abs_error']:.4f})")
        print(f"  Source: {row['dataset_source']}")
        print(f"  Quality issues: {row['quality_string']}")
        print()

    return high_error

def pass_filter(data, cutoff, fs, order=4, btype='low'):
    from scipy import signal

    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data

def apply_filter(alpha, time, cutoff, order, btype):
    # Calculate sampling frequency
    dt = np.mean(np.diff(time))
    fs = 1 / dt  # Sampling frequency in Hz
    return pass_filter(alpha, cutoff=cutoff, fs=fs, order=order, btype=btype), fs

    # Apply low-pass filter (10 Hz cutoff)


def explore_data_smoothing(processed_data, plot=False, cutoff=1.5, order=8, btype='low'):
    data = processed_data[['msFromStart', 'alpha', 'beta', 'sqrt(X²+Y²+Z²)']]

    time = data['msFromStart'].values / 1000  # Convert to seconds
    alpha = data['alpha'].values
    beta = data['beta'].values
    magnitude = data['sqrt(X²+Y²+Z²)'].values

    alpha_filtered, fs = apply_filter(alpha, time, cutoff, order, btype)
    beta_filtered, fs = apply_filter(beta, time, cutoff, order, btype)

    if plot:
        plot_smoothing_frequency(alpha, alpha_filtered, beta, beta_filtered,
                                 magnitude, cutoff, fs, time)

    return alpha_filtered, beta_filtered


def smoothing_investigation(all_tests_pisa):
    for test in all_tests_pisa:
        _, _ = explore_data_smoothing(test.processed_data, plot=True)
    return all_tests_pisa


def plot_smoothing_frequency(alpha, alpha_filtered, beta, beta_filtered, magnitude, cutoff, fs, time):
    def compute_spectrum(signal_data, fs):
        n = len(signal_data)
        freq = np.fft.rfftfreq(n, 1 / fs)
        spectrum = np.abs(np.fft.rfft(signal_data))
        # Normalize
        spectrum = spectrum * 2 / n
        return freq, spectrum

    freq_alpha, spec_alpha = compute_spectrum(alpha, fs)
    freq_beta, spec_beta = compute_spectrum(beta, fs)
    freq_mag, spec_mag = compute_spectrum(magnitude, fs)
    freq_alpha_filt, spec_alpha_filt = compute_spectrum(alpha_filtered, fs)
    freq_beta_filt, spec_beta_filt = compute_spectrum(beta_filtered, fs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Subplot 1: Original time-series data with filtered signals overlaid
    ax1.plot(time, alpha, label='Alpha', alpha=0.4, linewidth=1, color='C0')
    ax1.plot(time, beta, label='Beta', alpha=0.4, linewidth=1, color='C1')
    ax1.plot(time, magnitude, label='√(X²+Y²+Z²)', alpha=0.7, linewidth=1, color='C2')
    ax1.plot(time, alpha_filtered, label=f'Alpha (filtered, {cutoff} Hz)', linewidth=1.5, color='C0',
             linestyle='--')
    ax1.plot(time, beta_filtered, label=f'Beta (filtered, {cutoff} Hz)', linewidth=1.5, color='C1', linestyle='--')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title(f'Original Time-Series Data with Low-Pass Filtered Signals ({cutoff} Hz)', fontsize=13,
                  fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Frequency spectrum
    ax2.plot(freq_alpha, spec_alpha, label='Alpha', alpha=0.4, linewidth=1, color='C0')
    ax2.plot(freq_beta, spec_beta, label='Beta', alpha=0.4, linewidth=1, color='C1')
    ax2.plot(freq_mag, spec_mag, label='√(X²+Y²+Z²)', alpha=0.7, linewidth=1, color='C2')
    ax2.plot(freq_alpha_filt, spec_alpha_filt, label=f'Alpha (filtered, {cutoff} Hz)', linewidth=1.5, color='C0',
             linestyle='--')
    ax2.plot(freq_beta_filt, spec_beta_filt, label=f'Beta (filtered, {cutoff} Hz)', linewidth=1.5, color='C1',
             linestyle='--')
    ax2.axvline(x=cutoff, color='red', linestyle=':', linewidth=2, label=f'{cutoff} Hz cutoff', alpha=0.7)
    ax2.set_xlabel('Frequency (Hz)', fontsize=11)
    ax2.set_ylabel('Magnitude', fontsize=11)
    ax2.set_title('Frequency Spectrum', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, min(50, fs / 2))  # Show up to 50 Hz or Nyquist frequency

    plt.tight_layout()
    plt.show()

    return None


def compute_diffs(gt_gwalk, gt_manual):
    diffs = {}
    for key in gt_gwalk.keys():
        if gt_gwalk[key] is not None and gt_manual[key] is not None and gt_manual[key] is not np.nan:
            gtg = gt_gwalk[key]
            gtm = gt_manual[key]
            if gtg > 500:
                gtg = gtg/500
            if gtm > 500:
                gtm = gtm/500
            diffs[key] = gtg - gtm
            if abs(diffs[key]) > 5:
                print("Large diff in " + key + ": " + str(diffs[key]))
    return diffs


def plot_gts(gt_gwalk, gt_manual):
    diffs = compute_diffs(gt_gwalk, gt_manual)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(list(diffs.keys()), list(diffs.values()), marker='o', linestyle='', color='red')
    ax.set_xlabel('Test Index')
    ax.set_ylabel('GWalk GT - Manual GT')
    ax.set_title('Difference between GWalk and Manual Ground Truths')
    plt.xticks(rotation=90)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(running_settings.figures_all + os.sep + 'ground_truth_differences.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Prepare data: use gt_gwalk if available, otherwise gt_manual
    data_by_dataset = {'synergy': [], 'pisa': [], 'parkapp': []}
    colors_by_dataset = {'synergy': [], 'pisa': [], 'parkapp': []}
    all_data = []
    all_colors = []

    for key in gt_gwalk.keys():
        # Determine dataset
        if 'synergy' in key:
            dataset = 'synergy'
        elif 'pisa' in key:
            dataset = 'pisa'
        elif 'parkapp' in key:
            dataset = 'parkapp'
        else:
            continue

        # Use gt_gwalk if available, otherwise gt_manual
        if gt_gwalk[key] is not None and not np.isnan(gt_gwalk[key]):
            print("ok")
            value = gt_gwalk[key]
            if value > 500: value=value/1000
            color = 'primary'  # Blue for gt_gwalk
        else:
            print("not ok - manual" + key)
            value = gt_manual[key]
            if value > 500: value=value/1000
            color = 'manual'  # Orange for gt_manual

        data_by_dataset[dataset].append(value)
        colors_by_dataset[dataset].append(color)
        all_data.append(value)
        all_colors.append(color)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))

    datasets = ['synergy', 'pisa', 'parkapp', 'all']
    titles = ['Synergy', 'PISA', 'ParkApp', 'All Datasets']

    for idx, (ax, dataset, title) in enumerate(zip(axes, datasets, titles)):
        if dataset == 'all':
            data = all_data
            colors = all_colors
        else:
            data = data_by_dataset[dataset]
            colors = colors_by_dataset[dataset]

        if not data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        # Separate data by color
        primary_data = [d for d, c in zip(data, colors) if c == 'primary']
        manual_data = [d for d, c in zip(data, colors) if c == 'manual']

        # Create boxplot
        positions = []
        box_data = []
        box_colors = []

        if primary_data:
            positions.append(1)
            box_data.append(primary_data)
            box_colors.append('#1f77b4')  # Blue

        if manual_data:
            positions.append(1.3)
            box_data.append(manual_data)
            box_colors.append('#ff7f0e')  # Orange

        bp = ax.boxplot(box_data, positions=positions, widths=0.25, patch_artist=True,
                        showfliers=True, notch=False)

        # Color the boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Styling
        if len(box_data) == 1:
            title = title + ' (' + str(len(np.concatenate(box_data))) + ' tests)'
        else:
            title = title + ' (' + str(len(box_data[0])) + ' gwalk, '+ str(len(box_data[1])) + ' manual)'
        ax.set_title(title, fontsize=14)
        ax.set_ylabel('Ground Truth Value', fontsize=11)
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Unify y-axis across all subplots
    all_values = all_data
    if all_values:
        y_min = min(all_values) * 0.95
        y_max = max(all_values) * 1.05
        for ax in axes:
            ax.set_ylim(y_min, y_max)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', alpha=0.7, label='gt_gwalk'),
        Patch(facecolor='#ff7f0e', alpha=0.7, label='gt_manual (fallback)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.99),
               ncol=2, frameon=False, fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(running_settings.figures_all + os.sep + 'ground_truth_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Total samples - Synergy: {len(data_by_dataset['synergy'])}, "
          f"PISA: {len(data_by_dataset['pisa'])}, "
          f"ParkApp: {len(data_by_dataset['parkapp'])}, "
          f"All: {len(all_data)}")

    pass


def observe_groundtruth(all_tests):
    gt_gwalk = {}
    gt_manual = {}
    tests_gwalk = []

    for test in all_tests:
        index = test.user_id + '_' + str(test.session_id)
        gt_gwalk[index] = test.gt_total_gwalk
        gt_manual[index] = test.gt_total_manual

        if np.isnan(gt_gwalk[index]) or gt_gwalk[index] is None:
            print(f"Missing gwalk ground truth for test: {index}")
        else:
            tests_gwalk.append(test)

    if False:
        df_gts = pd.DataFrame(
            {'GWalk_GT': gt_gwalk,
             'Manual_GT': gt_manual}
        )

        df_gts.to_csv(running_settings.results_all + os.sep + 'groundtruths_comparison.csv')

    plot_gts(gt_gwalk, gt_manual)

    return tests_gwalk
