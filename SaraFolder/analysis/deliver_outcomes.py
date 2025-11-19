import numpy as np

from SaraFolder.settings import utils_labelling, utils_plots, utils_evaluation, running_settings, utils_dataquality
from SaraFolder.settings.utils_parkaapp import utils_parkapp
from SaraFolder.settings.utils_synergy import utils_synloaders

import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    all_tests = utils_parkapp.load_everything()
    tests_dict = {test.user_id + '_' + str(test.session_id): test for test in all_tests}

    tests_dict['1_synergy_1'].show_tuttecose(method='labelling')

    print(1)
