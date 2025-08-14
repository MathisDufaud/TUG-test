import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import datetime as dt
import running_settings

base_path = running_settings.base_path

def ready_df():
    motion_files, orientation_files = loader()
    skipped = {} # ???
    df = load_fusiondf(motion_files, orientation_files, skipped)
    return df

def loader():
    motion_files = {}
    orientation_files = {}

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
            if val > base + 50:
                df_corrected.loc[i:,elem] -= abs(val-base)
            elif val < base - 50:
                df_corrected.loc[i:,elem] += abs(val-base)
            base = df_corrected[elem].iloc[i]

    #interpolate
    df_final = df_m.copy()
    df_final['alpha'] = np.interp(df_m['relative_timestamp'],df_corrected['relative_timestamp'],df_corrected['alpha'])
    df_final['beta'] = np.interp(df_m['relative_timestamp'],df_corrected['relative_timestamp'],df_corrected['beta'])
    df_final['gamma'] = np.interp(df_m['relative_timestamp'],df_corrected['relative_timestamp'],df_corrected['gamma'])
    df_final['phase'] = np.zeros(len(df_final))

    df_final['alpha'] = moving_average(df_final['alpha'],20)
    for elem in ['beta','gamma','acc.x','acc.y','acc.z','rotRate.alpha','rotRate.beta','rotRate.gamma']:
        df_final[elem] = moving_average(df_final[elem])

    return df_final

def load_fusiondf(motion_files, orientation_files, skipped):
    times = pd.read_csv(base_path + os.sep + "manual_times.csv", index_col=0)

    df_fusion = {}

    for key in list(motion_files.keys()):
        if key in list(skipped['skipped_keys']):
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

    # labels
    for key, df in df_fusion.items():
        t0 = times['t_start'][key]
        t6 = times['t_end'][key]
        df.loc[(df['relative_timestamp'] >= t0) & (df['relative_timestamp'] <= t6), 'phase'] = 1

    df = pd.concat(df_fusion, ignore_index=True)

    return df