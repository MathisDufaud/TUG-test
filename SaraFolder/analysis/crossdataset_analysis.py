import numpy as np

from SaraFolder.settings import utils_labelling, utils_plots, utils_evaluation, running_settings
from SaraFolder.settings.utils_parkaapp import utils_parkapp
from SaraFolder.settings.utils_synergy import utils_synloaders

import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    all_tests_parkapp = utils_parkapp.load_all_tests(dataset_id ='parkapp', context='supervised')
    all_tests_pisa = utils_parkapp.load_all_tests(dataset_id = 'pisa', context='supervised')
    all_tests_synergy = utils_parkapp.load_all_tests(dataset_id = 'synergy', context='supervised')
    all_tests = list(np.concatenate([all_tests_synergy, all_tests_pisa, all_tests_parkapp]))

    # All tests overview:
    if True:
        utils_synloaders.overview_total(all_tests_parkapp, all_tests_pisa, all_tests_synergy,
                                        resultspath=running_settings.results_all, title='alltestssupervised')

    # Run Simon and Mathis algo
    if True:
        utils_labelling.labelling_acrossall(all_tests)
        utils_evaluation.evaluate_results(all_tests, eval_type='duration',
                                          method='labelling',
                                          gttype='gwalk',
                                          dataset='all', title='all')


    print('End of the document')

