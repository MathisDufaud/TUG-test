import numpy as np

from SaraFolder.settings import utils_labelling, utils_plots, utils_evaluation, running_settings, utils_MLnew
from SaraFolder.settings.utils_parkaapp import utils_parkapp
from SaraFolder.settings.utils_synergy import utils_synloaders

import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    all_tests = utils_parkapp.load_everything()

    # Set up csv with new manual start and end times
    if True:
        utils_MLnew.setup_manual_labelling_csv(all_tests, filename='testssupervised_manualmsstartend_new.csv')

    model = utils_MLnew.ML_pipeline(all_tests,
                                    model_name="mdl_tcn.h5",
                                    architecture='tcn', # strongbs, 'cnn_bilstm', # strongbs # bs_predictbatch, 'tcn'
                                    use_cv=True,
                                    n_splits='equalcvsplit',  # lopo # equalcvsplit # int number
                                    training_epochs=30,
                                    save_model=True,
                                    input_type='triaxial', # or 'magnitude_acc' # triaxial
                                    output_steps=0,
                                    load_existing=False
                                    )

    print(1)
