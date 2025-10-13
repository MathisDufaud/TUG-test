import os
import sys

from SaraFolder.settings import classes, running_settings, utils_plots
from SaraFolder.settings.utils_parkaapp import utils_parkapp


def load_syntests(dataset_id):
    return None


def overview_general(df_general, all_tests, resultspath):
    utils_plots.plot_tugtoverview(df_general, image_path=resultspath.replace('.txt', '.jpg'))

    lg = classes.Logger(resultspath)
    sys.stdout = lg

    print("Original amount of tests: ", len(all_tests))
    print("Remaining tests for analysis: ", len(all_tests))
    print("# Participants: ", df_general['Participant'].nunique())
    print("Average frequency (Hz) samples/seconds: ",
          round((df_general['samples'] / df_general['duration']).mean(), 2))
    print("Average # test per participant: ", df_general.groupby('Participant').size().mean())
    print("Min # test per participant: ", df_general.groupby('Participant').size().min())
    print("Max # test per participant: ", df_general.groupby('Participant').size().max())
    print("Std # test per participant: ", round(df_general.groupby('Participant').size().std(), 2))
    print("Average test duration (s) (raw): ", round(df_general['duration'].mean(), 2))
    print("Min test duration (s) (raw): ", round(df_general['duration'].min(), 2))
    print("Max test duration (s) (raw): ", round(df_general['duration'].max(), 2))
    print("Std test duration (s) (raw): ", round(df_general['duration'].std(), 2))
    print("Average test duration (s) (GT manual): ", round(df_general['durationGTm'].mean(), 2))
    print("Min test duration (s) (GT manual): ", round(df_general['durationGTm'].min(), 2))
    print("Max test duration (s) (GT manual): ", round(df_general['durationGTm'].max(), 2))
    print("Std test duration (s) (GT manual): ", round(df_general['durationGTm'].std(), 2))
    print("Average test duration (s) (GT gwalk): ", round(df_general['durationGTg'].mean(), 2))
    print("Min test duration (s) (GT gwalk): ", round(df_general['durationGTg'].min(), 2))
    print("Max test duration (s) (GT gwalk): ", round(df_general['durationGTg'].max(), 2))
    print("Std test duration (s) (GT gwalk): ", round(df_general['durationGTg'].std(), 2))
    lg.stop_logging()
    sys.stdout = sys.__stdout__
def tugt_overview_synergy(all_tests):
    df_general = utils_parkapp.build_general_df(all_tests)

    resultspath = running_settings.results_synergy + os.sep + running_settings.tugt_overview_synergy
    overview_general(df_general, all_tests, resultspath=resultspath)


    return None