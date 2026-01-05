    
# %%
import numpy as np

from SaraFolder.settings import utils_labelling, utils_plots, utils_evaluation, running_settings, utils_dataquality
from SaraFolder.settings.utils_parkaapp import *
from SaraFolder.settings.utils_parkaapp.utils_parkapp import load_all_tests, merge_pisaoldnew
from SaraFolder.settings.utils_synergy import utils_synloaders

import matplotlib


if __name__ == "__main__":


    all_tests_synergy = load_all_tests(dataset_id='synergy', context='supervised')
    
    all_tests_parkapp = load_all_tests(dataset_id='parkapp', context='supervised')

    all_tests_pisa_new = load_all_tests(dataset_id='pisa_new', context='supervised')
    all_tests_pisa = load_all_tests(dataset_id='pisa', context='supervised')
    all_tests_pisa_new = merge_pisaoldnew(all_tests_pisa, all_tests_pisa_new)



