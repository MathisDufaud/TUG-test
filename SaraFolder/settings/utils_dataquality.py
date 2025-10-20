import os
import time

import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
from SaraFolder.settings import running_settings


def observesingletests(all_tests, method):
    """
    Args:
        all_tests: classes TUG test.

    I want to plot each row data and be able to write a comment for that test that goes into the pd dataframe at each iteration

    Returns:
        a pandas dataframe with index the user_id + str(session_id), columns ['comments', 'whichstrangesignal']

    """
    df_tests = pd.DataFrame(columns=['comments', 'whichstrangesignal', 'error tot gwalk ', 'error tot manual', 'gt gwalk', 'gt manual'])

    for test in all_tests:
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

        indexid = test.user_id + '_' + str(test.session_id)

        # Get user input for comments
        comment = input(f"Enter comment for test {indexid}: ")
        which_strange = input(f"Which strange signal for test {indexid}: ")

        # Add row to dataframe
        df_tests.loc[indexid] = [comment, which_strange, error_tot_duration_gwalk, error_tot_duration_manual, gtg, gtm]

        print(f"Added comments for {indexid}\n")
        plt.close('all')

    # Save dataframe to CSV
    csvpath = running_settings.results_all + os.sep + 'testssupervised_comments.csv'
    df_tests.to_csv(csvpath, index=True)
    print(f"Saved comments to {csvpath}")

    return None


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
        dict_to_plot['gtManual'].append(test.gt_total_manual*1000 if test.gt_total_manual < 5000 else test.gt_total_manual)
        dict_to_plot['estimation'].append(test.results['labelling'] if not isinstance(test.results['labelling'], str) else None)
        dict_to_plot['estDuration'].append((test.results['labelling']['t_end'] - test.results['labelling']['t_start']) if not isinstance(test.results['labelling'], str) else None)
        for stat in stats_of_interest:
            if stat not in dict_to_plot:
                dict_to_plot[stat] = []
            dict_to_plot[stat].append(test.data_quality_stats[stat])

    df_plot = pd.DataFrame(dict_to_plot)
    return df_plot


def compute_tests_stats(all_tests, stats_of_interest):
    for test in all_tests:
        test.plot_labelling(method='labelling', plot=False)
        test.data_quality_investigation(plot=False)

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
    dict_stats['autocorr2sec_beta'] = float(df['beta'].autocorr(lag=120))
    dict_stats['autocorr2sec_alpha'] = float(df['alpha'].autocorr(lag=120))
    dict_stats['autocorr2sec_acc'] = float(df['sqrt(X²+Y²+Z²)'].autocorr(lag=120))
    dict_stats['entropy_alpha'] = float(calculate_entropy(df['alpha']))
    dict_stats['entropy_beta'] = float(calculate_entropy(df['beta']))
    dict_stats['autocorr5sec_acc'] = float(df['sqrt(X²+Y²+Z²)'].autocorr(lag=300))
    dict_stats['autocorr5sec_alpha'] = float(df['alpha'].autocorr(lag=300))
    dict_stats['autocorr5sec_beta'] = float(df['beta'].autocorr(lag=300))

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