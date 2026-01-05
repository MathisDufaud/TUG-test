from SaraFolder.settings import utils_plots, utils_evaluation
from SaraFolder.settings.utils_parkaapp import utils_parkapp
import matplotlib

from SaraFolder.settings.utils_pisa import utils_pisatugloaders
from SaraFolder.settings.utils_synergy import utils_synloaders

if __name__ == "__main__":

    all_tests = utils_pisatugloaders.load_synpisatests()

    if False:
        all_tests[0].plot_labelling(method='labelling', plot=True)

    utils_pisatugloaders.tugt_overview_pisa([test for test in all_tests if test.dataset_id == 'pisa'], logging=False)

    utils_synloaders.tugt_overview_synergy([test for test in all_tests if test.dataset_id == 'synergy'], logging=False)

    for a in all_tests:
        a.plot_labelling(method='labelling', plot=False)

    utils_evaluation.evaluate_results(all_tests, eval_type='duration', gttype='gwalk', method='labelling', dataset='pisa')


    print(1)

