from SaraFolder.settings import utils_plots
from SaraFolder.settings.utils_parkaapp import utils_parkapp
import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    # 406 motion and orientation files
    # In skipped: 132 names
    # Left to analyse: 274
    # df_gt_dict 274
    # TODO: all results df is 265 (9 skipped). Labelling algorithm doesn't work with 9 of the 274.

    all_tests = utils_parkapp.load_all_tests(dataset_id ='parkapp', context='supervised')

    icc_s = utils_parkapp.tugt_icc(all_tests)
    utils_plots.plot_icc(icc_s, icctype = 'ICC2')

    if True:
        utils_parkapp.tugt_overview_parkapp(all_tests, logging=False)

    print(1)
