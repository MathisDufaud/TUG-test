from SaraFolder.settings import utils_plots, utils_external
from SaraFolder.settings.utils_parkaapp import utils_parkapp
import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    all_test_extra = utils_external.load_data()
    all_tests = utils_parkapp.load_everything()

    icc_s = utils_parkapp.tugt_icc(all_test_extra)
    utils_plots.plot_icc(icc_s, icctype = 'ICC2')

    if True:
        utils_parkapp.tugt_overview_parkapp(all_test_extra, logging=False)

    print(1)
