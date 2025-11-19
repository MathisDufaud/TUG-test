import numpy as np

from SaraFolder.settings import utils_labelling, utils_plots, utils_evaluation, running_settings, utils_dataquality
from SaraFolder.settings.utils_parkaapp import utils_parkapp
from SaraFolder.settings.utils_synergy import utils_synloaders

import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    all_tests = utils_parkapp.load_everything()

    if False:
        all_tests = utils_dataquality.observe_groundtruth(all_tests)

    # All tests overview:
    if False:
        utils_synloaders.overview_total(all_tests_parkapp, all_tests_pisa, all_tests_synergy,
                                        resultspath=running_settings.results_all, title='alltestssupervised')

    # Run algorithms
    if True:
        method = 'labelling'    # darioalgo or labelling
        utils_labelling.labelling_acrossall(all_tests, method=method)
        blacklist = utils_labelling.observe_noresult_tests([test for test in all_tests if isinstance(test.results[method], str)], method='labelling')

        utils_evaluation.evaluate_results(all_tests, eval_type='duration',
                                          method=method, gttype='gwalk', dataset='all', title='all'+method+'_help', logging=True)

    # Parameter optimization
    if True:
        method='labelling'
        utils_labelling.parameter_optimization_labelling(all_tests, method=method,
                                                         eval_type = 'duration',
                                                         gttype='gwalk',
                                                         dataset = 'all',
                                                         title = 'all_paramopt_thresshold_amplitude')
    print('End of the document')

