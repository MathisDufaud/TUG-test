import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import datetime as dt

import pickle

from SaraFolder.settings import running_settings

base_path = running_settings.data_path

def ready_df():
    if not os.path.exists(base_path + os.sep + "df_processed.pickle"):
        motion_files, orientation_files = loader()
        df = load_fusiondf(motion_files, orientation_files)
        # Save dictionary to pickle
        pickle.dump(df, open(base_path + os.sep + "df_processed.pickle", "wb"))
    else:
        df = pickle.load(open(base_path + os.sep + "df_processed.pickle", "rb"))

    return labelling_phases(df)


def clinic_sessions():
    # Looping through patients to upload motion and orientation files
    return [
    '35254651','38554653','35580718',
    '39301023','35954567','42647424','37383188','41473916',
    '37437161','40790328','37806365','41530610','41526585','39247525', '42385176','39254756','42217210',
    '39294883', '42057118', '41905208',  '45309974', '41909772',
    '45308241', '42654273', '43019325', '48230514', '48228763',
    '43765084','44142351', '48234516', '44143711', '44451585', '48597759','48974829','46371264',
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
                session_folder = session_folder + "_u" # Unsupervised test

            if motion_file_list:
                df_motion = pd.read_csv(motion_file_list[0], index_col=0)
                df_motion['relative_timestamp'] = pd.to_timedelta(df_motion['relative_timestamp']).dt.total_seconds()
                motion_files[session_folder] = df_motion

            if orientation_file_list:
                df_orientation = pd.read_csv(orientation_file_list[0], index_col=0)
                df_orientation['relative_timestamp'] = pd.to_timedelta(df_orientation['relative_timestamp']).dt.total_seconds()
                orientation_files[session_folder] = df_orientation

    return motion_files, orientation_files

def moving_average(data, window=5):
    serie = pd.Series(data)
    rolling_mean = serie.rolling(window, min_periods=1, center=True).mean()
    return rolling_mean

def setup_df(df_m,df_o):
    df_corrected = df_o.copy()
    for elem in ['alpha','beta','gamma']:
        base = df_corrected[elem].iloc[0]
        for i in range(len(df_corrected)):
            val = df_corrected[elem].iloc[i]
            # Correction: TODO - empirically tested - check
            if val > base + 50:
                df_corrected.loc[i:,elem] -= abs(val-base)
            elif val < base - 50:
                df_corrected.loc[i:,elem] += abs(val-base)
            base = df_corrected[elem].iloc[i]

    #interpolate - why is it necessary? Isnt' the 'relative_timestamp' column in df_m and in df_corrected, the same?
    df_final = df_m.copy()
    df_final['alpha'] = np.interp(df_m['relative_timestamp'],df_corrected['relative_timestamp'],df_corrected['alpha'])
    df_final['beta'] = np.interp(df_m['relative_timestamp'],df_corrected['relative_timestamp'],df_corrected['beta'])
    df_final['gamma'] = np.interp(df_m['relative_timestamp'],df_corrected['relative_timestamp'],df_corrected['gamma'])
    df_final['phase'] = np.zeros(len(df_final))

    df_final['alpha'] = moving_average(df_final['alpha'],20)
    for elem in ['beta','gamma','acc.x','acc.y','acc.z','rotRate.alpha','rotRate.beta','rotRate.gamma']:
        df_final[elem] = moving_average(df_final[elem])

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
        if key in skipped:
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