from SaraFolder.settings import utils_plots, utils_evaluation
from SaraFolder.settings.utils_parkaapp import utils_parkapp
import matplotlib

from SaraFolder.settings.utils_pisa import utils_pisatugloaders

matplotlib.use('TkAgg')

if __name__ == "__main__":

    # tug_data = utils_pisatugloaders.load_json_data()

    all_tests = utils_pisatugloaders.load_synpisatests()

    if False:
        all_tests[0].plot_labelling(method='labelling', plot=True)

    for a in all_tests:
        a.plot_labelling(method='labelling', plot=False)

    utils_evaluation.evaluate_results(all_tests, eval_type='duration', method='labelling', dataset='pisa')

    print(1)

