import numpy as np

from SaraFolder.settings import utils_labelling, utils_plots, utils_evaluation, running_settings
from SaraFolder.settings.utils_parkaapp import utils_parkapp
from SaraFolder.settings.utils_synergy import utils_synloaders

import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    all_tests_pisa_new = utils_parkapp.load_all_tests(dataset_id = 'pisa_new', context='supervised')

    all_tests_parkapp = utils_parkapp.load_all_tests(dataset_id ='parkapp', context='supervised')
    all_tests_pisa = utils_parkapp.load_all_tests(dataset_id = 'pisa', context='supervised')

    all_tests_synergy = utils_parkapp.load_all_tests(dataset_id = 'synergy', context='supervised')
    all_tests = list(np.concatenate([all_tests_synergy, all_tests_pisa, all_tests_parkapp]))

    # All tests overview:
    if False:
        utils_synloaders.overview_total(all_tests_parkapp, all_tests_pisa, all_tests_synergy,
                                        resultspath=running_settings.results_all, title='alltestssupervised')

    # Run algorithms
    if True:
        method = 'labelling' # darioalgo or labelling
        utils_labelling.labelling_acrossall(all_tests, method=method)
        blacklist = utils_labelling.observe_noresult_tests([test for test in all_tests if isinstance(test.results[method],str)], method='labelling')

        utils_evaluation.evaluate_results(all_tests, eval_type='duration',
                                          method=method, gttype='gwalk',
                                          dataset='all', title='all'+method+'_help', logging=True)


    # Parameter optimization
    if True:
        method='labelling'
        utils_labelling.parameter_optimization_labelling(all_tests, method=method,
                                                         eval_type = 'duration',
                                                         gttype='gwalk',
                                                         dataset = 'all',
                                                         title = 'all_paramopt_thresshold_amplitude')
    print('End of the document')

