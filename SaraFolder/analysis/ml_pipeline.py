import numpy as np

from SaraFolder.settings import utils_labelling, utils_plots, utils_evaluation, running_settings, utils_MLnew
from SaraFolder.settings.utils_parkaapp import utils_parkapp
from SaraFolder.settings.utils_synergy import utils_synloaders

import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    all_tests_parkapp = utils_parkapp.load_all_tests(dataset_id ='parkapp', context='supervised')
    all_tests_pisa = utils_parkapp.load_all_tests(dataset_id = 'pisa', context='supervised')
    all_tests_synergy = utils_parkapp.load_all_tests(dataset_id = 'synergy', context='supervised')
    all_tests = list(np.concatenate([all_tests_synergy, all_tests_pisa, all_tests_parkapp]))

    # Set up csv with new manual start and end times
    if True:
        utils_MLnew.setup_manual_labelling_csv(all_tests, filename='testssupervised_manualmsstartend.csv')

    model = utils_MLnew.ML_pipeline(all_tests,
                                    model_name="mdl_15str3_strongbs.h5",
                                    architecture='strongbs', # strongbs, '',
                                    use_cv=True,
                                    n_splits=5,  # lopo
                                    training_epochs=30,
                                    save_model=False,
                                    input_type='triaxial')  # or 'magnitude_acc' # triaxial
    if True:
        pass

    print(1)
