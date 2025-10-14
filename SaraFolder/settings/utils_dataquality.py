import os
import time

import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
from SaraFolder.settings import running_settings


def observesingletests(all_tests):
    """
    Args:
        all_tests: classes TUG test.

    I want to plot each row data and be able to write a comment for that test that goes into the pd dataframe at each iteration

    Returns:
        a pandas dataframe with index the user_id + str(session_id), columns ['comments', 'whichstrangesignal']

    """
    df_tests = pd.DataFrame(columns=['comments', 'whichstrangesignal'])
    for test in all_tests:
        # Plot data
        test.plot_raw_data()

        plt.draw()
        plt.pause(0.5)  # Pause for 0.5 seconds

        indexid = test.user_id + '_' + str(test.session_id)
        # Get user input for comments
        comment = input(f"Enter comment for test {indexid}: ")
        which_strange = input(f"Which strange signal for test {indexid}: ")

        # Add row to dataframe
        df_tests.loc[indexid] = [comment, which_strange]

        print(f"Added comments for {indexid}\n")
        plt.close('all')
    # Save dataframe to CSV
    csvpath = running_settings.results_all + os.sep + 'alltests_comments.csv'
    df_tests.to_csv(csvpath, index=True)
    print(f"Saved comments to {csvpath}")

    return None