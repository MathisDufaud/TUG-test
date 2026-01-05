# %%
import numpy as np

from SaraFolder.settings import utils_labelling, utils_plots, utils_evaluation, running_settings, utils_dataquality, \
    utils_external
from SaraFolder.settings.utils_parkaapp import utils_parkapp
from SaraFolder.settings.utils_synergy import utils_synloaders

import matplotlib


if __name__ == "__main__":

    # all_test_extra = utils_external.load_data()
    all_tests = utils_parkapp.load_everything()

    # Run algorithms
    if True:
        method = 'labelling'    # darioalgo or labelling
        utils_labelling.labelling_acrossall(all_tests, method=method)
        blacklist = utils_labelling.observe_noresult_tests([test for test in all_tests if isinstance(test.results[method], str)], method='labelling')
        utils_evaluation.evaluate_results(all_tests, eval_type='duration',
                                          method=method, gttype='gwalk', dataset='all', title='all'+method, logging=True)

        error_all, indiv_error_duration = utils_dataquality.compute_error_tests(all_tests, method='labelling')
        utils_dataquality.compare_error_all_indiv(error_all, indiv_error_duration)
        utils_dataquality.plot_tests_witherror(all_tests, error_threshold=15, method='labelling')

    # Parameter optimization
    if False:
        method='labelling'
        utils_labelling.parameter_optimization_labelling(all_tests, method=method,
                                                         eval_type = 'duration',
                                                         gttype='gwalk',
                                                         dataset = 'all',
                                                         title = 'all_paramopt_thresshold_amplitude')
        
    print('End of the document')

