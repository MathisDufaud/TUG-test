import numpy as np

from SaraFolder.settings import utils_labelling, utils_plots, utils_evaluation, running_settings, utils_MLnew, \
    utils_external
from SaraFolder.settings.utils_parkaapp import utils_parkapp
from SaraFolder.settings.utils_synergy import utils_synloaders

import matplotlib


if __name__ == "__main__":
    all_tests = utils_parkapp.load_everything()
    all_tests_matey = utils_external.load_data()

    # Set up csv with new manual start and end times
    if True:
        all_tests = utils_MLnew.setup_manual_labelling_csv(all_tests, filename='testssupervised_manualmsstartend_new.csv')
        
        model, scaler, best_fold, all_fold_tests = utils_MLnew.ML_pipeline(all_tests,
                                    model_name="mdl_strongbs_sixax.h5", # let's keep the bigger kernels!
                                    architecture='strongbs', # strongbs, 'cnn_bilstm', # strongbs # bs_predictbatch, 'tcn'
                                    use_cv=True,
                                    n_splits='equalcvsplit',  # lopo # equalcvsplit # int number # type: ignore
                                    training_epochs=2,
                                    save_model=True,
                                    input_type='sixaxial', # or 'magnitude_acc' # triaxial
                                    output_steps=0,
                                    load_existing=False, 
                                    evaluation = False
                                    ) # type: ignore

    if True:
        utils_MLnew.holdout_external_testing(model, scaler, best_fold,
                                             input_type='sixaxial',
                                             output_steps=0,
                                             all_tests_matey=all_tests_matey)

    if False:
        utils_MLnew.investigate_predictedproba(all_fold_tests)

    if True:
        if isinstance(all_fold_tests, list):
            all_fold_tests[1].show_tuttecose()

    print(1)
