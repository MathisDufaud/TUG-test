import numpy as np

from SaraFolder.settings import utils_labelling, utils_plots, utils_evaluation, running_settings, utils_dataquality
from SaraFolder.settings.utils_parkaapp import utils_parkapp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')

    if False:
        all_parkapp_skipped = utils_parkapp.load_all_tests_skipped(dataset_id ='parkapp', context='supervised')

    all_tests_parkapp = utils_parkapp.load_all_tests(dataset_id ='parkapp', context='supervised')
    all_tests_pisa = utils_parkapp.load_all_tests(dataset_id = 'pisa', context='supervised')
    all_tests_synergy = utils_parkapp.load_all_tests(dataset_id = 'synergy', context='supervised')
    all_tests = list(np.concatenate([all_tests_synergy, all_tests_pisa, all_tests_parkapp]))

    if False:
        utils_parkapp.tugt_overview_parkapp(all_tests, logging=False)

    if False:
        method = 'labelling' # darioalgo or labelling
        utils_labelling.labelling_acrossall(all_tests, method=method)
        utils_dataquality.observe_qualityvariable(all_tests)

    if False:
        # Function to track comments for each test
        utils_dataquality.observesingletests(all_tests, method='labelling', title='testssupervised_manual.csv')

    if True:
        # Observe quality and stats
        utils_dataquality.quality_error(all_tests)

        utils_dataquality.quality_stats(all_tests)

    if True:
        utils_dataquality.compute_tests_stats(all_tests, stats_of_interest = ['entropy_acc', 'std_acc', 'entropy_rotrate', 'std_rotrate',
                  'autocorr2sec_beta', 'autocorr2sec_alpha', 'autocorr2sec_acc', 'entropy_alpha', 'entropy_beta',
                  'median_acc', 'median_rotrate', 'var_acc', 'var_rotrate', 'autocorr5sec_acc', 'autocorr5sec_alpha', 'autocorr5sec_beta'
                                                                         ])

        utils_dataquality.error_vs_stats(all_tests, stats_of_interest = ['entropy_acc', 'std_acc', 'entropy_rotrate', 'std_rotrate',
                  'autocorr2sec_beta', 'autocorr2sec_alpha', 'autocorr2sec_acc', 'entropy_alpha', 'entropy_beta',
                  'median_acc', 'median_rotrate', 'var_acc', 'var_rotrate', 'autocorr5sec_acc', 'autocorr5sec_alpha', 'autocorr5sec_beta'
                                                                         ])

    print('End of the document')

