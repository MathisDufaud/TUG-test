import os
import pickle

import numpy as np
import pandas as pd

from SaraFolder.settings import running_settings, classes
from SaraFolder.settings.utils_parkaapp import utils_parkapp
from data_external.matey_sanz.utils import data_loading, exploration, visualization


def fix_groundtruth(all_tests):
    for test in all_tests:
        print(1)
    pass


def load_mateysanz(tests, subject_info):
    all_tests = []

    for k, t in tests.items():
        session_id = k.split('_')[1]
        user_id = k.split('_')[0].strip('s')
        dev_pos = k.split('_')[-1]
        test = classes.TUGTest(test_id=None,
                               session_id = session_id,
                               user_id = user_id + '_' + 'matey_' + dev_pos,
                               dataset_id = 'matey_' + dev_pos)

        context = 'supervised'
        test.context = context

        test.gt_total_manual = np.nan
        test.gt_total_gwalk = np.nan
        t.timestamp = t.timestamp - t.timestamp.values[0]
        t_start = t[t.label!='SEATED'].timestamp.values[0]
        t_end = t[t.label!='SEATED'].timestamp.values[-1]
        test.gt_total_gwalk = t_end - t_start

        # Rename columns
        t = t.rename(columns={'x_gyro': 'rotRate.alpha', 'y_gyro': 'rotRate.beta', 'z_gyro': 'rotRate.gamma'})
        t = t.rename(columns={'x_acc': 'acc.x', 'y_acc': 'acc.y', 'z_acc': 'acc.z'})
        t = t.rename(columns={'timestamp': 'msFromStart'})
        t['relative_timestamp'] = pd.to_timedelta(t['msFromStart'], unit='milliseconds').dt.total_seconds()

        test.raw_data = t[['msFromStart', 'relative_timestamp', 'acc.x', 'acc.y', 'acc.z',
                            'rotRate.alpha', 'rotRate.beta', 'rotRate.gamma', 'label']].copy()

        all_tests.append(test)

    all_tests = utils_parkapp.process_tests_data(all_tests)
    return all_tests


def load_data(dataset_id='matey_sanz', context='supervised'):
    # Load Matey-Sanz
    if dataset_id == 'matey_sanz':
        if 'matey_sanz_tests.pickle' in os.listdir(running_settings.data_synpisa):
            with open(running_settings.data_synpisa + os.sep + 'matey_sanz_tests.pickle', 'rb') as handle:
                all_tests = pickle.load(handle)
        else:
            tests = data_loading.load_data(path=running_settings.data_mateysanz)
            subject_info = data_loading.load_subjects_info(path=os.path.join(running_settings.data_mateysanz, 'subjects_info.csv'))
            all_tests = load_mateysanz(tests, subject_info)

            with open(running_settings.data_synpisa + os.sep + 'matey_sanz_tests.pickle', 'wb') as handle:
                pickle.dump(all_tests, handle)



    return all_tests