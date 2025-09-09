import sys

import pandas as pd
import numpy as np
import os
import glob

import pickle

from SaraFolder.settings import running_settings, classes, utils_plots

base_path = running_settings.data_path


def ready_df():
    if not os.path.exists(base_path + os.sep + running_settings.name_df_processed):
        motion_files, orientation_files = loader()
        df = load_fusiondf(motion_files, orientation_files)
        # Save dictionary to pickle
        pickle.dump(df, open(base_path + os.sep + running_settings.name_df_processed, "wb"))
    else:
        df = pickle.load(open(base_path + os.sep + running_settings.name_df_processed, "rb"))

    if 'labelling' in running_settings.name_df_processed:
        return df
    else:
        return labelling_phases(df)


def clinic_sessions():
    # Looping through patients to upload motion and orientation files
    return [
        '35254651', '38554653', '35580718',
        '39301023', '35954567', '42647424', '37383188', '41473916',
        '37437161', '40790328', '37806365', '41530610', '41526585', '39247525', '42385176', '39254756', '42217210',
        '39294883', '42057118', '41905208', '45309974', '41909772',
        '45308241', '42654273', '43019325', '48230514', '48228763',
        '43765084', '44142351', '48234516', '44143711', '44451585', '48597759', '48974829', '46371264',
        '46690193', '51363556', '48341540', '51577424', '48610252', '50189491'
    ]


def loader():
    motion_files = {}
    orientation_files = {}

    clinic_sess = clinic_sessions()

    # Looping through patients to upload motion and orientation files
    for patient_folder in os.listdir(base_path):
        patient_path = os.path.join(base_path, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        for session_folder in os.listdir(patient_path):
            session_path = os.path.join(patient_path, session_folder)
            if not os.path.isdir(session_path):
                continue

            motion_pattern = os.path.join(session_path, "*_motion.csv")
            orientation_pattern = os.path.join(session_path, "*_orientation.csv")

            motion_file_list = glob.glob(motion_pattern)
            orientation_file_list = glob.glob(orientation_pattern)

            session = session_folder.split('_')[-1]
            if session in clinic_sess:
                session_folder = session_folder + "_s"  # Supervised test
            else:
                session_folder = session_folder + "_u"  # Unsupervised test

            if motion_file_list:
                df_motion = pd.read_csv(motion_file_list[0], index_col=0)
                df_motion['relative_timestamp'] = pd.to_timedelta(df_motion['relative_timestamp']).dt.total_seconds()
                motion_files[session_folder] = df_motion

            if orientation_file_list:
                df_orientation = pd.read_csv(orientation_file_list[0], index_col=0)
                df_orientation['relative_timestamp'] = pd.to_timedelta(
                    df_orientation['relative_timestamp']).dt.total_seconds()
                orientation_files[session_folder] = df_orientation

    return motion_files, orientation_files


def moving_average(data, window=5):
    serie = pd.Series(data)
    rolling_mean = serie.rolling(window, min_periods=1, center=True).mean()
    return rolling_mean


def setup_df(df_m, df_o):
    if 'lstm' in running_settings.name_df_processed:
        """
        This version is from tugt_LSTM
        """
        df_corrected = df_o.copy()
        for elem in ['alpha', 'beta', 'gamma']:
            base = df_corrected[elem].iloc[0]
            for i in range(len(df_corrected)):
                val = df_corrected[elem].iloc[i]
                # Correction: TODO - empirically tested - check
                if val > base + 50:
                    df_corrected.loc[i:, elem] -= abs(val - base)
                elif val < base - 50:
                    df_corrected.loc[i:, elem] += abs(val - base)
                base = df_corrected[elem].iloc[i]

        # interpolate - why is it necessary? Isnt' the 'relative_timestamp' column in df_m and in df_corrected, the same?
        df_final = df_m.copy()
        df_final['alpha'] = np.interp(df_m['relative_timestamp'], df_corrected['relative_timestamp'],
                                      df_corrected['alpha'])
        df_final['beta'] = np.interp(df_m['relative_timestamp'], df_corrected['relative_timestamp'],
                                     df_corrected['beta'])
        df_final['gamma'] = np.interp(df_m['relative_timestamp'], df_corrected['relative_timestamp'],
                                      df_corrected['gamma'])
        df_final['phase'] = np.zeros(len(df_final))

        df_final['alpha'] = moving_average(df_final['alpha'], 20)
        for elem in ['beta', 'gamma', 'acc.x', 'acc.y', 'acc.z', 'rotRate.alpha', 'rotRate.beta', 'rotRate.gamma']:
            df_final[elem] = moving_average(df_final[elem])
        return df_final

    if 'labelling' in running_settings.name_df_processed:
        """
        This version is from tugt_labelling
        """
        df_corrected = df_o.copy()

        for elem in ['alpha', 'beta', 'gamma']:
            base = df_corrected[elem].iloc[0]
            for i in range(len(df_corrected)):
                val = df_corrected[elem].iloc[i]
                if val > base + 50:
                    df_corrected.loc[i:, elem] -= abs(val - base)
                elif val < base - 50:
                    df_corrected.loc[i:, elem] += abs(val - base)
                base = df_corrected[elem].iloc[i]

        # interpolate
        df_final = df_m.copy()
        df_final['alpha'] = np.interp(df_m['relative_timestamp'], df_corrected['relative_timestamp'],
                                      df_corrected['alpha'])
        df_final['beta'] = np.interp(df_m['relative_timestamp'], df_corrected['relative_timestamp'],
                                     df_corrected['beta'])
        df_final['gamma'] = np.interp(df_m['relative_timestamp'], df_corrected['relative_timestamp'],
                                      df_corrected['gamma'])

        # create new columns
        df_final['all'] = np.sqrt(((np.abs(df_final['acc.x']) - np.min(np.abs(df_final['acc.x']))) / (
                np.max(np.abs(df_final['acc.x'])) - np.min(np.abs(df_final['acc.x'])))) ** 2
                                  + ((np.abs(df_final['acc.y']) - np.min(np.abs(df_final['acc.y']))) / (
                np.max(np.abs(df_final['acc.y'])) - np.min(np.abs(df_final['acc.y'])))) ** 2
                                  + ((np.abs(df_final['acc.z']) - np.min(np.abs(df_final['acc.z']))) / (
                np.max(np.abs(df_final['acc.z'])) - np.min(np.abs(df_final['acc.z'])))) ** 2
                                  + ((np.abs(df_final['rotRate.alpha']) - np.min(np.abs(df_final['rotRate.alpha']))) / (
                np.max(np.abs(df_final['rotRate.alpha'])) - np.min(np.abs(df_final['rotRate.alpha'])))) ** 2
                                  + ((np.abs(df_final['rotRate.beta']) - np.min(np.abs(df_final['rotRate.beta']))) / (
                np.max(np.abs(df_final['rotRate.beta'])) - np.min(np.abs(df_final['rotRate.beta'])))) ** 2
                                  + ((np.abs(df_final['rotRate.gamma']) - np.min(np.abs(df_final['rotRate.gamma']))) / (
                np.max(np.abs(df_final['rotRate.gamma'])) - np.min(np.abs(df_final['rotRate.gamma'])))) ** 2)

        df_final['derivative'] = np.abs(np.gradient(df_final['alpha'])) + np.abs(
            np.gradient(df_final['beta'])) + np.abs(np.gradient(df_final['gamma']))

        df_final['der_beta_gamma'] = np.abs(np.gradient(df_final['beta'])) + np.abs(np.gradient(df_final['gamma']))

        df_final['rotRate_beta_gamma'] = np.sqrt((df_final['rotRate.beta']) ** 2 + (df_final['rotRate.gamma']) ** 2)

        return df_final


def labelling_phases(df_fusion):
    times = pd.read_csv(base_path + os.sep + "manual_times.csv", index_col=0)
    if running_settings.phases_to_consider == 1:
        for key, df in df_fusion.items():
            key = key[:-2]  # remove _s or _u
            t0 = times['t_start'][key]
            t6 = times['t_end'][key]
            df.loc[(df['relative_timestamp'] >= t0) & (df['relative_timestamp'] <= t6), 'phase'] = 1

        df = pd.concat(df_fusion, ignore_index=True)
        return df
    else:
        print(1)
    return df_fusion


def load_fusiondf(motion_files, orientation_files):
    skipped = list(pd.read_csv(base_path + os.sep + "skipped.csv", index_col=0).index)

    df_fusion = {}
    for key in list(motion_files.keys()):
        if key[:-2] in skipped:
            del orientation_files[key]
            del motion_files[key]
        else:
            df_motion = motion_files[key]
            df_orientation = orientation_files[key]
            try:
                df_final = setup_df(df_motion, df_orientation)
                df_fusion[key] = df_final
            except Exception as e:
                print(f"Error on the setup for {key} : {e}")
    return df_fusion


def load_groundtruth_samplepersample(df_fusion):
    times = pd.read_csv(base_path + os.sep + "manual_times.csv", index_col=0)
    skipped = list(pd.read_csv(base_path + os.sep + "skipped.csv", index_col=0).index)

    for key, df in df_fusion.items():
        key = key[:-2]  # remove _s or _u
        if key in skipped:
            print("Skipping key ", key)
            # Remove df from the dictionary
            continue
        try:
            # ['t_start', 't_end_stand', 't_start_turn', 't_end_turn', 't_start_turn2',
            #        't_start_sit', 't_end']
            t0 = times['t_start'][key]
            t1 = times['t_end_stand'][key]
            t2 = times['t_start_turn'][key]
            t3 = times['t_end_turn'][key]
            t4 = times['t_start_turn2'][key]
            t5 = times['t_start_sit'][key]
            t6 = times['t_end'][key]
            df.loc[(df['relative_timestamp'] >= t0) & (df['relative_timestamp'] <= t6), 'test'] = True
            df.loc[(df['relative_timestamp'] <= t0) | (df['relative_timestamp'] > t6), 'test'] = False
            df.loc[(df['relative_timestamp'] <= t0) | (df['relative_timestamp'] > t6), 'phase'] = 0
            df.loc[(df['relative_timestamp'] >= t0) & (df['relative_timestamp'] < t1), 'phase'] = 1  # Stand up
            df.loc[(df['relative_timestamp'] >= t1) & (df['relative_timestamp'] < t2), 'phase'] = 2  # Walk 1
            df.loc[(df['relative_timestamp'] >= t2) & (df['relative_timestamp'] < t3), 'phase'] = 3  # Turn 1
            df.loc[(df['relative_timestamp'] >= t3) & (df['relative_timestamp'] < t4), 'phase'] = 2  # Walk 2
            df.loc[(df['relative_timestamp'] >= t4) & (df['relative_timestamp'] < t5), 'phase'] = 3  # Turn 2
            df.loc[(df['relative_timestamp'] >= t5) & (df['relative_timestamp'] <= t6), 'phase'] = 4  # Sit down
            print("Ok GT for key ", key)
        except Exception as e:
            print(f"Error on the GT for {key} : {e}")

    return {k: v[['test', 'phase']] for k, v in df_fusion.items() if k[:-2] not in skipped}


def load_groundtruth_dict():
    times = pd.read_csv(base_path + os.sep + "manual_times.csv", index_col=0)
    skipped = list(pd.read_csv(base_path + os.sep + "skipped.csv", index_col=0).index)
    dict_times = times.transpose().to_dict()
    for key in skipped:
        if key in dict_times:
            del dict_times[key]
    return dict_times




def tugt_overview(df_fusion, df_gt_dict):
    skipped = list(pd.read_csv(base_path + os.sep + "skipped.csv", index_col=0).index)


    df_general = pd.DataFrame(columns=['Participant', 'Session', 'samples', 'duration', 'duration GT'])
    for key, df in df_fusion.items():
        participant = key.split('_')[0]
        session = key.split('_')[1]
        n_samples = len(df)
        duration = (df['msFromStart'].iloc[-1] - df['msFromStart'].iloc[0])/1000
        durationGT = df_gt_dict[key[:-2]]['t_end'] - df_gt_dict[key[:-2]]['t_start']
        df_general = pd.concat([df_general, pd.DataFrame({'Participant': [participant],
                                                          'Session': [session],
                                                          'samples': [n_samples],
                                                          'duration': [duration],
                                                          'durationGT':[durationGT]})], ignore_index=True)
    utils_plots.plot_tugtoverview(df_general)

    lg = classes.Logger(running_settings.results_path + os.sep + running_settings.tugt_overview)
    sys.stdout = lg

    print("Original amount of tests: ", len(df_fusion) + len(skipped))
    print("\nSkipped tests for various reasons: ", len(skipped))
    print("\nRemaining tests for analysis: ", len(df_fusion))
    print("\n# Participants: ", df_general['Participant'].nunique())
    print("\nAverage frequency (Hz) samples/seconds: ", round((df_general['samples']/df_general['duration']).mean(), 2))
    print("\nAverage # test per participant: ", df_general.groupby('Participant').size().mean())
    print("\nMin # test per participant: ", df_general.groupby('Participant').size().min())
    print("\nMax # test per participant: ", df_general.groupby('Participant').size().max())
    print("\nStd # test per participant: ", round(df_general.groupby('Participant').size().std(), 2))
    print("\nAverage test duration (s) (raw): ", round(df_general['duration'].mean(), 2))
    print("\nMin test duration (s) (raw): ", round(df_general['duration'].min(), 2))
    print("\nMax test duration (s) (raw): ", round(df_general['duration'].max(), 2))
    print("\nStd test duration (s) (raw): ", round(df_general['duration'].std(), 2))
    print("\nAverage test duration (s) (GT): ", round(df_general['durationGT'].mean(), 2))
    print("\nMin test duration (s) (GT): ", round(df_general['durationGT'].min(), 2))
    print("\nMax test duration (s) (GT): ", round(df_general['durationGT'].max(), 2))
    print("\nStd test duration (s) (GT): ", round(df_general['durationGT'].std(), 2))

    lg.stop_logging()
    sys.stdout = sys.__stdout__

    return None
