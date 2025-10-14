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