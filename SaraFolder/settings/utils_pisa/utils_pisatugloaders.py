import sys

import pandas as pd
import numpy as np
import os

import matplotlib
from pingouin import intraclass_corr

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.ion()
from SaraFolder.settings import running_settings, classes, utils_plots, utils_dataquality
from SaraFolder.settings.utils_parkaapp import utils_parkapp
from SaraFolder.settings.utils_synergy import utils_synloaders


def load_pisatugtests(dataset_id):
    return None


# def load_ref_data():
#     tug_refs = pd.read_csv(data_path + os.sep + "tugs.csv")
#
#     df_tugs = tug_refs[["userKey", "tugTimeMs", "GWALKReferenceMs", "manualRefStartMs", "manualRefEndtMs", "homeClinic"]]
#     # df_tug_home = df_tugs[df_tugs["homeClinic"] == "home"].drop(columns=["homeClinic", "manualRefStartMs", "manualRefEndtMs", "GWALKReferenceMs"]).sort_values(by=["userKey"]) ######### ONLY HOME TESTS
#
#     # df_tug considering all userKey with value > 1000
#     # ALL THE SWEDISH
#     df_tug_home = df_tugs[df_tugs["userKey"] > 1000].drop(columns=["homeClinic", "manualRefStartMs", "manualRefEndtMs", "GWALKReferenceMs"]).sort_values(by=["userKey"])
#
#     # New columns test_iteration: add 0,1,2,3 to each row according to the userKey
#     df_tug_home["test_iteration"] = df_tug_home.groupby("userKey").cumcount()
#
#     return df_tug_home


def plot_reliability_results(results_dict=None, save_plots=False, title='',
                             figures_path=running_settings.figures_parkapp, figsize=(15, 10)):
    """
    Create comprehensive visualizations for test-retest reliability results
    """

    # Extract data for plotting
    results_data = {
        'n_tests': [],
        'n_userKeys': [],
        'n_observations': [],
        'icc': [],
        'ci_lower': [],
        'ci_upper': []
    }

    # Sort by number of tests
    sorted_keys = sorted(results_dict.keys(), key=lambda x: results_dict[x]['n_tests'])

    for key in sorted_keys:
        data = results_dict[key]
        results_data['n_tests'].append(data['n_tests'])
        results_data['n_userKeys'].append(data['n_userKeys'])
        results_data['n_observations'].append(data['n_observations'])
        results_data['icc'].append(float(data['value']))
        results_data['ci_lower'].append(float(data['ci_lower']))
        results_data['ci_upper'].append(float(data['ci_upper']))

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)

    # Define reliability interpretation colors
    def get_reliability_color(icc_value):
        if icc_value < 0.5:
            return '#ff6b6b'  # Poor (red)
        elif icc_value < 0.75:
            return '#4ecdc4'  # Moderate (teal)
        else:
            return '#45b7d1'  # Good (blue)

    # Plot 1: ICC with Confidence Intervals
    ax1 = plt.subplot(2, 3, 1)
    x_pos = np.arange(len(results_data['n_tests']))
    colors = [get_reliability_color(icc) for icc in results_data['icc']]

    # Plot ICC points
    bars = ax1.bar(x_pos, results_data['icc'], color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1)

    # Add confidence intervals
    ci_heights = np.array(results_data['ci_upper']) - np.array(results_data['ci_lower'])
    ax1.errorbar(x_pos, results_data['icc'],
                 yerr=[np.array(results_data['icc']) - np.array(results_data['ci_lower']),
                       np.array(results_data['ci_upper']) - np.array(results_data['icc'])],
                 fmt='none', color='black', capsize=5, capthick=2)

    ax1.set_xlabel('Number of Tests per Participant')
    ax1.set_ylabel('ICC(2,1)')
    ax1.set_title('Test-Retest Reliability by Number of Tests')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(results_data['n_tests'])
    ax1.grid(True, alpha=0.3)

    # Add reliability interpretation zones
    ax1.axhspan(0, 0.5, alpha=0.1, color='red', label='Poor (<0.5)')
    ax1.axhspan(0.5, 0.75, alpha=0.1, color='orange', label='Moderate (0.5-0.75)')
    ax1.axhspan(0.75, 1.0, alpha=0.1, color='green', label='Good (>0.75)')

    # Plot 2: Sample Sizes
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.bar(x_pos, results_data['n_userKeys'], color='skyblue',
                    alpha=0.7, edgecolor='black', linewidth=1)

    ax2.set_xlabel('Number of Tests per Participant')
    ax2.set_ylabel('Number of userKeys')
    ax2.set_title('Sample Size by Test Count')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(results_data['n_tests'])
    ax2.grid(True, alpha=0.3)

    # Plot 3: Confidence Interval Widths
    ax3 = plt.subplot(2, 3, 3)
    ci_widths = np.array(results_data['ci_upper']) - np.array(results_data['ci_lower'])
    bars3 = ax3.bar(x_pos, ci_widths, color='lightcoral', alpha=0.7,
                    edgecolor='black', linewidth=1)

    # Add value labels
    for bar, width in zip(bars3, ci_widths):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{width:.3f}', ha='center', va='bottom')

    ax3.set_xlabel('Number of Tests per Participant')
    ax3.set_ylabel('95% CI Width')
    ax3.set_title('Precision of ICC Estimates')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(results_data['n_tests'])
    ax3.grid(True, alpha=0.3)

    # Plot 4: ICC vs Sample Size Relationship
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(results_data['n_userKeys'], results_data['icc'],
                          c=results_data['n_tests'], s=100, cmap='viridis',
                          alpha=0.7, edgecolors='black', linewidth=2)

    # Add labels for each point
    for i, n_test in enumerate(results_data['n_tests']):
        ax4.annotate(f'{n_test} tests',
                     (results_data['n_userKeys'][i], results_data['icc'][i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax4.set_xlabel('Number of userKeys')
    ax4.set_ylabel('ICC(2,1)')
    ax4.set_title('Reliability vs Sample Size')
    ax4.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Number of Tests')

    # Plot 5: Comprehensive Summary Table
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')

    # Create summary table
    table_data = []
    for i in range(len(results_data['n_tests'])):
        reliability_level = "Poor" if results_data['icc'][i] < 0.5 else \
            "Moderate" if results_data['icc'][i] < 0.75 else "Good"

        table_data.append([
            f"{results_data['n_tests'][i]} tests",
            f"{results_data['n_userKeys'][i]}",
            f"{results_data['icc'][i]:.3f}",
            f"[{results_data['ci_lower'][i]:.3f}, {results_data['ci_upper'][i]:.3f}]",
            reliability_level
        ])

    table = ax5.table(cellText=table_data,
                      colLabels=['Test Count', 'N', 'ICC(2,1)', '95% CI', 'Level'],
                      cellLoc='center',
                      loc='center',
                      colColours=['lightgray'] * 5)

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax5.set_title('Summary Table', pad=20)

    # Plot 6: Error Bar Plot with Trends
    ax6 = plt.subplot(2, 3, 6)

    # Plot ICC with error bars
    ax6.errorbar(results_data['n_tests'], results_data['icc'],
                 yerr=[np.array(results_data['icc']) - np.array(results_data['ci_lower']),
                       np.array(results_data['ci_upper']) - np.array(results_data['icc'])],
                 fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8,
                 color='darkblue', ecolor='darkblue', alpha=0.8)

    # Add reliability zones
    ax6.axhspan(0, 0.5, alpha=0.1, color='red')
    ax6.axhspan(0.5, 0.75, alpha=0.1, color='orange')
    ax6.axhspan(0.75, 1.0, alpha=0.1, color='green')

    ax6.set_xlabel('Number of Tests per Participant')
    ax6.set_ylabel('ICC(2,1)')
    ax6.set_title('Reliability Trend Across Test Counts')
    ax6.grid(True, alpha=0.3)

    # Add horizontal lines for reference
    ax6.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Poor/Moderate threshold')
    ax6.axhline(y=0.75, color='green', linestyle='--', alpha=0.5, label='Moderate/Good threshold')
    ax6.legend(fontsize=8)

    plt.tight_layout()

    if save_plots:
        plt.savefig(running_settings.figures_path + os.sep + title, dpi=400, bbox_inches='tight')
        print("Plot saved")

    plt.show()

    return fig

def compute_test_retest_reliability(df, min_tests=2, method='icc', verbose=True):
    """
    Compute test-retest reliability for userKeys with varying number of tests

    Returns:
    --------
    dict : Results dictionary containing ICC
    """

    if verbose:
        print(f"{len(df)} total observations from {df['userKey'].nunique()} userKeys \n")

    results = {}
    for n_tests in [2, 3, 4, 5, 6]:
        userKey_counts = df.groupby('userKey').size()
        valid_userKeys = userKey_counts[userKey_counts >= n_tests].index

        # Step 3: Limit to max_tests test_iterations per userKey (keep first max_tests test_iterations)
        max_tests = n_tests
        if max_tests is not None:
            filtered_rows = []
            for pid in valid_userKeys:
                userKey_data = df[df['userKey'] == pid].sort_values('test_iteration')
                # Keep only the first max_tests test_iterations
                limited_data = userKey_data.head(max_tests)
                filtered_rows.append(limited_data)
            df_filtered = pd.concat(filtered_rows, ignore_index=True)

        if len(valid_userKeys) < 2:
            raise ValueError("Need at least 2 userKeys for reliability analysis")

        # Method 1: ICC Analysis (uses all available data points)
        if method in ['icc', 'both']:
            try:
                # For ICC, we can use all available test_iterations
                df_icc = df_filtered.copy()

                icc_result = intraclass_corr(
                    data=df_icc,
                    targets='userKey',
                    raters='test_iteration',
                    ratings='tugTimeMs'
                )

                # Extract ICC(2,1) - Two-way random effects, single measurement, absolute agreement
                icc = icc_result[icc_result['Type'] == 'ICC3'] # or ICC3

                if not icc.empty:
                    results['ICC_'+str(n_tests)] = {
                        'type': icc_result['Type'],
                        'value': icc['ICC'].iloc[0],
                        'ci_lower': icc['CI95%'].iloc[0][0],
                        'ci_upper': icc['CI95%'].iloc[0][1],
                        'n_userKeys': len(valid_userKeys),
                        'n_observations': len(df_icc),
                        'n_tests': n_tests
                    }

                    if verbose:
                        print(f"\n Results for {n_tests} tests and {len(valid_userKeys)} participants:")
                        print(f"ICC = {results['ICC_'+str(n_tests)]['value']:.3f}")
                        print(f"95% CI: [{results['ICC_'+str(n_tests)]['ci_lower']:.3f}, {results['ICC_'+str(n_tests)]['ci_upper']:.3f}]")

            except Exception as e:
                if verbose:
                    print(f"ICC calculation failed: {str(e)}")
    return results


def icc_analysis(tug_ref_data, title, figures_path):

    results = compute_test_retest_reliability(
        tug_ref_data,
        min_tests=2,
        method='icc',
        verbose=True
    )

    plot_reliability_results(results, save_plots=True, title=title, figures_path=figures_path)  # Using your actual results dictionary

    return None


def load_json_data():
    # JSON: % FOR ICC, USE THE FOLLOWING: each number is a subject, numbers in the array are measurements in ms

    json_data = {"1": [5950], "2": [27599, 19432, 22749, 22433],
                 "3": [8516, 8667, 6899, 7800, 8149, 6582, 7966],
                 "4": [4167, 9167, 8300, 8033, 7450], "5": [8583, 7850],
                 "6": [5283, 5052, 4741, 5483, 5900, 5233],
                 "7": [9634, 9583, 9967, 10666, 9783, 14066],
                 "8": [8716, 8716, 7984, 7649, 7817, 7150],
                 "9": [8750, 8833, 9866, 10199, 11049, 9650]}

    tug_df = pd.DataFrame()
    c=0
    for user_key, measurements in json_data.items():
        for i, measurement in enumerate(measurements):
            tug_df = pd.concat([tug_df, pd.DataFrame(data={
                "userKey": int(user_key),
                "tugTimeMs": measurement,
                "test_iteration": i}, index=[c])])
            c+=1

    return tug_df


def twitching_orientationalpha(df_orientation):
    if df_orientation.shape[0]>0:
        # Make a copy to avoid modifying the original dataframe
        df = df_orientation.copy()

        # Ensure correct data types
        df['msFromStart'] = df['msFromStart'].astype(int)
        df['orA'] = df['orA'].astype(float)
        df['orB'] = df['orB'].astype(float)
        df['orG'] = df['orG'].astype(float)

        # Store original values before modification
        df['orA_original'] = df['orA']
        df['orB_original'] = df['orB']
        df['orG_original'] = df['orG']

        # Initialize previous values
        try:
            previous_orA = df.iloc[0]['orA']
            previous_orB = df.iloc[0]['orB']
            previous_orG = df.iloc[0]['orG']
        except:
            print('issue here')

        # Process each row starting from index 1
        for i in range(1, len(df)):
            # Get current original values
            current_orA_original = df.iloc[i]['orA_original']
            current_orB_original = df.iloc[i]['orB_original']
            current_orG_original = df.iloc[i]['orG_original']

            # Get previous original values
            previous_orA_original = df.iloc[i - 1]['orA_original']
            previous_orB_original = df.iloc[i - 1]['orB_original']
            previous_orG_original = df.iloc[i - 1]['orG_original']

            # Calculate and apply deltaA
            deltaA = current_orA_original - previous_orA_original
            if abs(deltaA) > 90:
                deltaA = 0
            df.iloc[i, df.columns.get_loc('orA')] = previous_orA + deltaA

            # Calculate and apply deltaB
            deltaB = current_orB_original - previous_orB_original
            if abs(deltaB) > 90:
                deltaB = 0
            df.iloc[i, df.columns.get_loc('orB')] = previous_orB + deltaB

            # Calculate and apply deltaG
            deltaG = current_orG_original - previous_orG_original
            if abs(deltaG) > 90:
                deltaG = 0
            df.iloc[i, df.columns.get_loc('orG')] = previous_orG + deltaG

            # Update previous values with modified values
            previous_orA = df.iloc[i]['orA']
            previous_orB = df.iloc[i]['orB']
            previous_orG = df.iloc[i]['orG']

        # Remove the temporary original columns
        df = df.drop(columns=['orA_original', 'orB_original', 'orG_original'])

    return df


def load_test(path, test_id, df_tug_ref):
    tests = []
    participant = path.split(os.sep)[-1].split("_")[1]
    if int(participant) < 10:
        unique_tests = np.unique([f.split("tug")[1].split("_")[0] for f in os.listdir(path) if f.endswith(".csv")])
        dataset = 'synergy'
    else:
        tests_csvs = [f for f in os.listdir(path) if f.startswith("tug")]
        unique_tests = np.unique([f.split("tug")[1].split("_")[0] for f in tests_csvs])
        dataset = 'pisa'

    for i, t in enumerate(unique_tests):
        test_id += 1
        motion = path + os.sep + 'tug' + t + '_motion.csv'
        orientation = path + os.sep + 'tug' + t + '_orientation.csv'

        if os.path.exists(motion) and os.path.exists(orientation):
            test = classes.TUGTest(test_id = test_id,
                                   session_id = int(t),
                                   user_id = str(int(participant)) + '_' + dataset,
                                   dataset_id = dataset)
            context = df_tug_ref[(df_tug_ref['tugId'] == int(t))]['homeClinic'].values[0]
            if context == 'home':
                context = 'unsupervised'
            else:
                context = 'supervised'
            test.context = context

            test.gt_total_gwalk = df_tug_ref[(df_tug_ref['tugId'] == int(t))]['GWALKReferenceMs'].values[0]
            if df_tug_ref[(df_tug_ref['tugId'] == int(t))]['manualRefEndtMs'].values[0] is not None and df_tug_ref[(df_tug_ref['tugId'] == int(t))]['manualRefStartMs'].values[0] is not None:
                test.gt_total_manual = df_tug_ref[(df_tug_ref['tugId'] == int(t))]['manualRefEndtMs'].values[0] - df_tug_ref[(df_tug_ref['tugId'] == int(t))]['manualRefStartMs'].values[0]
            else:
                test.gt_total_manual = None

            df_motion = pd.read_csv(motion)
            df_orientation = pd.read_csv(orientation)
            if df_motion.shape[0] != 0 and df_orientation.shape[0] != 0:
                df_orientation = twitching_orientationalpha(df_orientation)

                # Merge on timestamp column
                df_motion = df_motion.sort_values('msFromStart')
                df_orientation = df_orientation.sort_values('msFromStart')

                df_merged = pd.merge(df_motion, df_orientation, on='msFromStart', how='outer').sort_values('msFromStart').reset_index(drop=True)

                test.raw_data = df_merged
                # test.processed_data = process_data(df_merged)
                # # Remove nan rows
                # test.processed_data = test.processed_data.dropna().reset_index(drop=True)
                tests.append(test)
            else:
                print("Motion or orientation dataframes are empty")

    return tests, test_id


def upload_gt_newpisa(path):
    import re
    for element in os.listdir(path):
        if element.endswith("Summary_TUG.txt"):
            gt_file = path + os.sep + element
            with open(gt_file, 'r', encoding='utf-16') as f:
                content = f.read()

            # Search for the pattern
            match = re.search(r'Analysis Duration \(s\)\s+([0-9.]+)', content)

            if match:
                duration = float(match.group(1))
                return duration
            else:
                print("bug")
                return None
    return None




def load_test_newpisa(path, test_id, participant):
    tests = []

    dataset = 'pisa_new'
    test_id += 1
    motion = path + os.sep + 'tug_motion.csv'
    orientation = path + os.sep + 'tug_orientation.csv'

    if os.path.exists(motion) and os.path.exists(orientation):
        test = classes.TUGTest(test_id = test_id,
                               session_id = 1, # TODO careful here
                               user_id = str(participant) + '_' + dataset,
                               dataset_id = dataset)

        context = 'supervised'
        test.context = context

        test.gt_total_manual = np.nan
        test.gt_total_gwalk = upload_gt_newpisa(path=path.strip(r'\\TUG_raw'))

        df_motion = pd.read_csv(motion)
        df_orientation = pd.read_csv(orientation)
        if df_motion.shape[0] != 0 and df_orientation.shape[0] != 0:
            df_orientation = twitching_orientationalpha(df_orientation)

            # Merge on timestamp column
            df_motion = df_motion.sort_values('msFromStart')
            df_orientation = df_orientation.sort_values('msFromStart')

            df_merged = pd.merge(df_motion, df_orientation, on='msFromStart', how='outer').sort_values('msFromStart').reset_index(drop=True)

            test.raw_data = df_merged
            tests.append(test)
        else:
            print("Motion or orientation dataframes are empty")
            test.raw_data = pd.DataFrame()
            test.processed_data = pd.DataFrame()

    return tests, test_id

def load_synpisatests():
    all_tests = []
    test_id = running_settings.test_id_start_synergy
    data_path = running_settings.data_synpisa
    tugscsv = data_path + os.sep + "tugs.csv"
    df_tug_ref = pd.read_csv(tugscsv)
    for t in os.listdir(data_path):
        if os.path.isdir(data_path + os.sep + t) and t.startswith("p"):
            print("################################ Participant folder: ", t)
            tests, test_id = load_test(data_path + os.sep + t, test_id, df_tug_ref)
            all_tests.extend(tests)
        print("\n")

    print("Keeping tests only if GT is available")
    returntests = [test for test in all_tests if not np.isnan(test.gt_total_manual) or not np.isnan(test.gt_total_gwalk)]

    return returntests

def tugt_overview_pisa(all_tests, logging):
    df_general = utils_parkapp.build_general_df(all_tests)
    resultspath = running_settings.results_pisatug + os.sep + running_settings.tugt_overview_pisa
    utils_synloaders.overview_general(df_general, all_tests, resultspath=resultspath, logging=logging, plot=True)


def resample60(df_raw):
    df_raw['datetime_index'] = pd.to_timedelta(df_raw['relative_timestamp'], unit='s')
    df = df_raw.set_index('datetime_index')

    # Sort by index to ensure proper interpolation
    df = df.sort_index()
    df_resampled = df.resample(running_settings.parameters['resamplingdelta']).mean()

    # Interpolate all numeric columns
    numeric_cols = df_resampled.select_dtypes(include=[np.number]).columns
    df_resampled[numeric_cols] = df_resampled[numeric_cols].interpolate(method='linear')

    # Recalculate msFromStart and relative_timestamp based on new sampling rate
    time_seconds = df_resampled.index.total_seconds()
    df_resampled['relative_timestamp'] = time_seconds
    df_resampled['msFromStart'] = (time_seconds * 1000).astype(int)

    # Reset index to get datetime_index as a column, then drop it
    df_resampled = df_resampled.reset_index()
    df_final = df_resampled.drop('datetime_index', axis=1)

    return df_final


def smoothalphabeta(df_final, cutoff=1, order=8, btype='low', plot=False):
    alpha_filtered, beta_filtered = utils_dataquality.explore_data_smoothing(df_final,
                                                                             cutoff=cutoff,
                                                                             order=order,
                                                                             btype=btype,
                                                                             plot=plot)
    df_final['alpha'] = alpha_filtered
    df_final['beta'] = beta_filtered
    return df_final


def process_data(df_raw):

    if len(df_raw) > 0:
        if not 'relative_timestamp' in df_raw.columns:
            df_raw['relative_timestamp'] = pd.to_timedelta(df_raw['msFromStart'], unit='milliseconds').dt.total_seconds()

        if not 'sqrt(X²+Y²+Z²)' in df_raw.columns and 'accX' in df_raw.columns:
            df_raw['sqrt(X²+Y²+Z²)'] = np.sqrt((np.abs(df_raw['accX']))**2 +
                                           (np.abs(df_raw['accY']))**2 +
                                           (np.abs(df_raw['accZ']))**2)

        elif not 'sqrt(X²+Y²+Z²)' in df_raw.columns and 'acc.x' in df_raw.columns:
            df_raw['sqrt(X²+Y²+Z²)'] = np.sqrt((np.abs(df_raw['acc.x'])) ** 2 +
                                               (np.abs(df_raw['acc.y'])) ** 2 +
                                               (np.abs(df_raw['acc.z'])) ** 2)

        df_final = resample60(df_raw)

        if 'orA' in df_final.columns:
            df_final = df_final.rename(columns={'orA':'alpha', 'orB':'beta', 'orG':'gamma'})
            df_final = df_final.rename(columns={'rotA':'rotRate.alpha', 'rotB':'rotRate.beta', 'rotG':'rotRate.gamma'})
            df_final = df_final.rename(columns={'accX':'acc.x', 'accY':'acc.y', 'accZ':'acc.z'})

        if 'alpha' in df_final.columns and 'beta' in df_final.columns:
            df_final = smoothalphabeta(df_final, plot=False)

        if 'all' not in df_final:
            df_final = utils_parkapp.new_columns(df_final)

        if 'label' in df_raw.columns:
            df_raw['testBoool'] = False
            df_final['testBoool'] = False

            df_raw.loc[df_raw['label'] != 'SEATED', 'testBoool'] = True
            msStart = df_raw[df_raw['testBoool'] == True]['msFromStart'].values[0]
            msEnd = df_raw[df_raw['testBoool'] == True]['msFromStart'].values[-1]
            df_final.loc[(df_final['msFromStart'] >= msStart) &
                                    (df_final['msFromStart'] <= msEnd), 'testBool'] = True

        return df_final
    else:
        return 'empty df raw'


def load_newpisa():
    all_tests = []
    test_id = running_settings.test_id_start_pisatug
    data_path = running_settings.data_synpisa

    for t in os.listdir(data_path):
        if os.path.isdir(data_path + os.sep + t) and t.startswith("a"):
            print("################################ Participant folder: ", t)
            tug_path = data_path + os.sep + t + os.sep + "PreIntervention" + os.sep + 'TUG_raw'
            tests, test_id = load_test_newpisa(tug_path, test_id, participant=t.split("_")[1])
            all_tests.extend(tests)
        print("\n")

    print("Keeping tests only if GT is available")
    returntests = [test for test in all_tests if not np.isnan(test.gt_total_manual) or not np.isnan(test.gt_total_gwalk)]

    return returntests