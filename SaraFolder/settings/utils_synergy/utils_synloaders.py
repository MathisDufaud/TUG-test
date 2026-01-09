import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from SaraFolder.settings import classes, running_settings, utils_plots
from SaraFolder.settings.utils_parkaapp import utils_parkapp


def load_syntests(dataset_id):
    return None


def overview_general(df_general, all_tests, resultspath, logging, plot):
    if plot:
        utils_plots.plot_tugtoverview(df_general, image_path=resultspath.replace('.md', '.jpg'))

    if logging:
        lg = classes.Logger(resultspath)
        sys.stdout = lg

    print("Original amount of tests: ", len(all_tests)) 
    print("\n")
    print("Remaining tests for analysis: ", len(all_tests))
    print("\n")
    print("Participants: ", df_general['participant'].nunique())
    print("\n")
    print("Average frequency (Hz) samples/seconds: ",
          round((df_general['samples'] / df_general['duration']).mean(), 2))
    print("\n")
    print("Average # test per participant: ", df_general.groupby('participant').size().mean())
    print("\n")
    print("Min amount of test per participant: ", df_general.groupby('participant').size().min())
    print("\n")
    print("Max amount of test per participant: ", df_general.groupby('participant').size().max())
    print("\n")
    print("Std amount of test per participant: ", round(df_general.groupby('participant').size().std(), 2))
    print("\n")
    print("Average test duration (s) (raw): ", round(df_general['duration'].mean(), 2))
    print("\n")
    print("Min test duration (s) (raw): ", round(df_general['duration'].min(), 2))
    print("\n")
    print("Max test duration (s) (raw): ", round(df_general['duration'].max(), 2))
    print("\n")
    print("Std test duration (s) (raw): ", round(df_general['duration'].std(), 2))
    print("\n")
    print("Average test duration (s) (GT manual): ", round(df_general['durationGTm'].mean(), 2))
    print("\n")
    print("Min test duration (s) (GT manual): ", round(df_general['durationGTm'].min(), 2))
    print("\n")
    print("Max test duration (s) (GT manual): ", round(df_general['durationGTm'].max(), 2))
    print("\n")
    print("Std test duration (s) (GT manual): ", round(df_general['durationGTm'].std(), 2))
    print("\n")
    print("Average test duration (s) (GT gwalk): ", round(df_general['durationGTg'].mean(), 2))
    print("\n")
    print("Min test duration (s) (GT gwalk): ", round(df_general['durationGTg'].min(), 2))
    print("\n")
    print("Max test duration (s) (GT gwalk): ", round(df_general['durationGTg'].max(), 2))
    print("\n")
    print("Std test duration (s) (GT gwalk): ", round(df_general['durationGTg'].std(), 2))
    print("\n")

    if logging:
        lg.stop_logging()
        sys.stdout = sys.__stdout__

def overview_total(all_tests, resultspath, title):

    lg = classes.Logger(resultspath + os.sep + 'overview'+title+'.md')
    sys.stdout = lg
    print("# ALL TESTS (parkapp, pisa + pisa new1, synergy old0 + synergy new1)")
    df_general = utils_parkapp.build_general_df(all_tests)
    overview_general(df_general, df_general, resultspath=resultspath, logging=False, plot=True)
    print("-------")

    print("\n # PARK APP TESTS #########################################")
    df_parkapp= df_general[df_general['dataset']=='parkapp']
    overview_general(df_parkapp, df_parkapp, resultspath=resultspath, logging=False, plot=False)
    
    print("-------")

    print("\n # PISA TESTS #########################################")
    df_pisa = df_general[df_general['dataset']=='pisa']
    overview_general(df_pisa, df_pisa, resultspath=resultspath, logging=False, plot=False)

    print("-------")
    print("\n # SYNERGY TESTS")
    df_syn = df_general[df_general['dataset']=='synergy']
    overview_general(df_syn, df_syn, resultspath=resultspath, logging=False, plot=False)
    lg.stop_logging()
    sys.stdout = sys.__stdout__

    checkgts(all_tests)
    return None


def checkgts(all_tests):
    # Print distribution of gwalk gt and manual

    gwalk_gt = []
    manual_gt = []
    for test in all_tests:
        if not np.isnan(test.gt_total_gwalk):
            gt = test.gt_total_gwalk
            if gt<1000:
                gt=gt*1000
            gwalk_gt.append(gt)

        if test.gt_total_manual != None:
            gt = test.gt_total_manual
            if gt<1000:
                gt=gt*1000
            manual_gt.append(gt)


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))

    bins = range(int(np.min(np.concatenate([gwalk_gt, manual_gt]))), int(np.max(np.concatenate([gwalk_gt, manual_gt]))), 400)
    # Plot histograms for both gwalk_gt and manual_gt
    ax.hist(gwalk_gt, bins=bins, alpha=0.7, label='GWalk GT', color='blue')
    ax.hist(manual_gt, bins=bins, alpha=0.7, label='Manual GT', color='orange')

    # Add labels, title, and legend
    ax.set_xlabel('Duration (s)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of GWalk and Manual Ground Truth Durations')
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig(running_settings.figures_all + os.sep + 'gtcomparison.jpg', dpi=400)
    plt.show()
    return None