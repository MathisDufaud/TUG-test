import numpy as np

from SaraFolder.settings import utils_labelling, utils_plots, utils_evaluation, running_settings, utils_MLnew
from SaraFolder.settings.utils_parkaapp import utils_parkapp
from SaraFolder.settings.utils_synergy import utils_synloaders

import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    all_tests_pisa_new = utils_parkapp.load_all_tests(dataset_id = 'pisa_new', context='supervised')
    all_tests_pisa = utils_parkapp.load_all_tests(dataset_id = 'pisa', context='supervised')
    all_tests_pisa_new = utils_parkapp.merge_pisaoldnew(all_tests_pisa, all_tests_pisa_new)
    all_tests_parkapp = utils_parkapp.load_all_tests(dataset_id ='parkapp', context='supervised')
    all_tests_synergy = utils_parkapp.load_all_tests(dataset_id = 'synergy', context='supervised')
    all_tests = list(np.concatenate([all_tests_synergy, all_tests_pisa, all_tests_parkapp]))



    # Set up csv with new manual start and end times
    if True:
        utils_MLnew.setup_manual_labelling_csv(all_tests, filename='testssupervised_manualmsstartend_new.csv')

    model = utils_MLnew.ML_pipeline(all_tests_parkapp,
                                    model_name="mdl_strongbsparkapp.h5",
                                    architecture='strongbs', # strongbs, '', # strongbs # bs_predictbatch
                                    use_cv=True,
                                    n_splits=5,  # lopo # equalcvsplit # int number
                                    training_epochs=5,
                                    save_model=False,
                                    input_type='triaxial',
                                    output_steps=0)  # or 'magnitude_acc' # triaxial
    if True:
        pass

    print(1)
