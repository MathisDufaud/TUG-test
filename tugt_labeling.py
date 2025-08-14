import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import os
import glob
import datetime as dt

# all test (folder 1 in attachements)
base_path = r"C:\Users\mathi\Documents\stage\tugt_patients\tugt_for_students\all"

motion_files = {}
orientation_files = {}

# patients P1 to P29
for patient_id in range(1, 30):
    patient_folder = f"P{patient_id}"
    patient_path = os.path.join(base_path, patient_folder)

    # sub folder from 1 to max 60
    for session_id in range(1, 61):
        session_folder = f"{patient_id}_{session_id}"
        session_path = os.path.join(patient_path, session_folder)

        # paths
        motion_file = os.path.join(session_path, f"P{patient_id}_{session_id}_motion.csv")
        orientation_file = os.path.join(session_path, f"P{patient_id}_{session_id}_orientation.csv")

        # load csv to df
        if os.path.exists(motion_file):
            df_motion = pd.read_csv(motion_file, index_col = 0)
            df_motion['relative_timestamp'] = pd.to_timedelta(df_motion['relative_timestamp']).dt.total_seconds()
            motion_files[session_folder] = df_motion

        if os.path.exists(orientation_file):
            df_orientation = pd.read_csv(orientation_file, index_col = 0)
            df_orientation['relative_timestamp'] = pd.to_timedelta(df_orientation['relative_timestamp']).dt.total_seconds()
            orientation_files[session_folder] = df_orientation


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

    #create new columns
    df_final['all'] = np.sqrt(((np.abs(df_final['acc.x'])-np.min(np.abs(df_final['acc.x'])))/(np.max(np.abs(df_final['acc.x']))-np.min(np.abs(df_final['acc.x']))))**2
                                + ((np.abs(df_final['acc.y'])-np.min(np.abs(df_final['acc.y'])))/(np.max(np.abs(df_final['acc.y']))-np.min(np.abs(df_final['acc.y']))))**2
                                + ((np.abs(df_final['acc.z'])-np.min(np.abs(df_final['acc.z'])))/(np.max(np.abs(df_final['acc.z']))-np.min(np.abs(df_final['acc.z']))))**2
                                + ((np.abs(df_final['rotRate.alpha'])-np.min(np.abs(df_final['rotRate.alpha'])))/(np.max(np.abs(df_final['rotRate.alpha']))-np.min(np.abs(df_final['rotRate.alpha']))))**2
                                + ((np.abs(df_final['rotRate.beta'])-np.min(np.abs(df_final['rotRate.beta'])))/(np.max(np.abs(df_final['rotRate.beta']))-np.min(np.abs(df_final['rotRate.beta']))))**2
                                + ((np.abs(df_final['rotRate.gamma'])-np.min(np.abs(df_final['rotRate.gamma'])))/(np.max(np.abs(df_final['rotRate.gamma']))-np.min(np.abs(df_final['rotRate.gamma']))))**2)

    df_final['derivative'] = np.abs(np.gradient(df_final['alpha'])) + np.abs(np.gradient(df_final['beta'])) + np.abs(np.gradient(df_final['gamma']))

    df_final['der_beta_gamma'] = np.abs(np.gradient(df_final['beta'])) + np.abs(np.gradient(df_final['gamma']))

    df_final['rotRate_beta_gamma'] = np.sqrt((df_final['rotRate.beta'])**2 + (df_final['rotRate.gamma'])**2)

    return df_final

df_fusion = {}

for key in motion_files:
    if key in orientation_files:
        df_motion = motion_files[key]
        df_orientation = orientation_files[key]
        try:
            df_final = setup_df(df_motion, df_orientation)
            df_fusion[key] = df_final
        except Exception as e:
            print(f"Erreur lors de la fusion pour {key} : {e}")
    else:
        print(f"Données manquantes pour {key} dans orientation_data")

def moving_average(data, window=5):
    serie = pd.Series(data)
    rolling_mean = serie.rolling(window, min_periods=1, center=True).mean()
    return rolling_mean

# find end of close to zero phase
def find_zero_phase_end(data, min_duration=30):
    end_index = 0

    data = np.array(data)
    threshold = (np.max(data)-np.min(data)) * 0.04
    near_zero = np.abs(data) < threshold


    count = 0

    for i, is_near in enumerate(near_zero):
        if is_near:
            count += 1
        else:
            if count >= min_duration:
                end_index = i
                break
            count = 0

    return end_index

# find end of close to zero phase starting from the end
def find_zero_phase_end_reverse(data, min_duration=30):
    end_index = len(data)-1

    data = np.array(data)
    threshold = (np.max(data)-np.min(data)) * 0.05
    near_zero = np.abs(data) < threshold

    enum_data = tuple(enumerate(near_zero))
    reverse_data = reversed(enum_data)

    count = 0

    for i, is_near in reverse_data:
        if is_near:
            count += 1
        else:
            if count >= min_duration:
                end_index = i
                break
            count = 0

    return end_index

# find the 2 turns
def start_change(base_data, window_size=100, threshold=120):
    data = np.array(base_data)
    detected_segments = []

    i = 0
    while i <= len(data) - window_size:
        window = data[i:i + window_size]
        amplitude = np.max(window) - np.min(window)

        if amplitude >= threshold:
            center = i + int(window_size * 0.5)

            # going left from center
            start = center
            while start-1 > 0 and np.abs(data[start] - data[start - 1]) > 0.3:
                start -= 1

            # going right from center
            end = center
            while end + 1 < len(data) - 1 and np.abs(data[end] - data[end + 1]) > 0.3:
                end += 1

            if end - start >= 50:
                detected_segments.append((start, end))

            i = end
        i += 1

    if len(detected_segments) >= 2:
        # sorting by amplitude+length
        def sort_key(seg):
            start, end = seg
            seg_data = data[start:end+1]
            amplitude = np.max(seg_data) - np.min(seg_data)
            length = end - start
            return amplitude + length

        # we keep the 2 bests
        top_segments = sorted(detected_segments, key=sort_key, reverse=True)[:2]

        # sort them by index
        return sorted(top_segments, key=lambda seg: seg[0])

    if threshold == 80:
        return None

    return start_change(base_data, window_size=window_size+10, threshold=threshold-10)

def find_zero_phase_end2(data, min_duration=30,k=0.05):
    end_index = 0

    data = np.array(data)
    threshold = (np.max(data)-np.min(data)) * k
    near_zero = np.abs(data) < threshold

    enum_data = tuple(enumerate(near_zero))
    reverse_data = reversed(enum_data)

    count = 0

    for i, is_near in reverse_data:
        if is_near and i!=0:
            count += 1
        else:
            if count >= min_duration:
                end_index = i+count
                return end_index
            count = 0

    return end_index

def find_zero_phase_end_reverse2(data, min_duration=30,k=0.04):

    data = np.array(data)
    end_index = len(data)-1
    threshold = (np.max(data)-np.min(data)) * k
    near_zero = np.abs(data) < threshold

    count = 0

    for i, is_near in enumerate(near_zero):
        if is_near and i!=len(data)-1:
            count += 1
        else:
            if count >= min_duration:
                end_index = i-count
                return end_index
            count = 0

    return end_index


def first_peak(base_data, threshold = 1.5):

    data = np.array(base_data)

    # Finding peaks
    if threshold > 10:
        while threshold > 10:
            peaks,_ = signal.find_peaks(data, height=threshold, distance=70)
            if len(peaks)>= 1:
                return peaks[-1]
            threshold -= 5
    else:
        while threshold > 0.5:
            peaks,_ = signal.find_peaks(data, height=threshold, distance=70)
            if len(peaks)>= 1:
                return peaks[-1]
            threshold -= 0.1
    return len(data)-1


def last_peak(base_data, threshold = 1.2):

    data = np.array(base_data)

    # Finding peaks
    if threshold > 10:
        while threshold > 10:
            peaks,_ = signal.find_peaks(data, height=threshold, distance=70)
            if len(peaks)>= 1:
                return peaks[0]
            threshold -= 5
    else:
        while threshold > 0.5:
            peaks,_ = signal.find_peaks(data, height=threshold, distance=70)
            if len(peaks)>= 1:
                return peaks[0]
            threshold -= 0.1
    return 0

def full_algo(df_start):

    if df_start.empty:
        return "empty df"

    # remove first 3 sec
    df = df_start.loc[(df_start['relative_timestamp'] >= 3)].copy()
    if df.empty:
        return "empty df after removing 3 sec"

    df.reset_index(drop=True, inplace=True)

    # find the 2 turns
    alpha_ma = moving_average(df['alpha'], 20)

    result = start_change(alpha_ma)

    if result is None:
        return "no turn found"
    else:
        (start_turn,end_turn),(start_turn2,end_turn2) = result
        t_start_turn = df.at[start_turn, 'relative_timestamp']
        t_end_turn = df.at[end_turn, 'relative_timestamp']
        t_start_turn2 = df.at[start_turn2, 'relative_timestamp']
        t_end_turn2 = df.at[end_turn2, 'relative_timestamp']

        start_limit = max(df.at[0, 'relative_timestamp'], t_start_turn - (t_end_turn2 - t_start_turn) * 1.5)
        end_limit = min(df.at[df.index[-1], 'relative_timestamp'], t_end_turn2 + (t_end_turn2 - t_start_turn))

        df_red = df.loc[(df['relative_timestamp'] >= start_limit) & (df['relative_timestamp'] <= end_limit)].copy()

        if df_red.empty:
            return "df_red empty"

        df_red.reset_index(drop=True, inplace=True)

        df_test = df_red.loc[(df_red['relative_timestamp'] >= t_end_turn2 - 1)].copy()
        if df_test.empty:
            return "df_test empty"

        df_test.reset_index(drop=True, inplace=True)

        new_start_der = find_zero_phase_end2(df_red.loc[df_red['relative_timestamp'] <= t_start_turn, 'derivative'])
        new_end_der = find_zero_phase_end_reverse2(df_test['derivative'], 20)
        t_new_start_der = df_red.at[new_start_der, 'relative_timestamp']
        t_new_end_der = df_test.at[new_end_der, 'relative_timestamp']

        new_start_all = find_zero_phase_end2(moving_average(df_red.loc[df_red['relative_timestamp'] <= t_start_turn, 'all']), 20, k=0.15)
        new_end_all = find_zero_phase_end_reverse2(moving_average(df_test['all']), 20, k=0.15)
        t_new_start_all = df_red.at[new_start_all, 'relative_timestamp']
        t_new_end_all = df_test.at[new_end_all, 'relative_timestamp']

        start_beta_gamma = first_peak(moving_average(df_red.loc[df_red['relative_timestamp'] <= t_start_turn, 'der_beta_gamma']))
        end_beta_gamma = last_peak(moving_average(df_test['der_beta_gamma']))
        t_start_beta_gamma = df_red.at[start_beta_gamma, 'relative_timestamp']
        t_end_beta_gamma = df_test.at[end_beta_gamma, 'relative_timestamp']

        start_rot = first_peak(moving_average(df_red.loc[df_red['relative_timestamp'] <= t_start_turn, 'rotRate_beta_gamma']), 75)
        end_rot = last_peak(moving_average(df_test['rotRate_beta_gamma']), 75)
        t_start_rot = df_red.at[start_rot, 'relative_timestamp']
        t_end_rot = df_test.at[end_rot, 'relative_timestamp']

        # end of standing and start of sitting
        end_stand_beta_gamma = last_peak(moving_average(df_red.loc[df_red['relative_timestamp'] >= t_start_beta_gamma, 'der_beta_gamma']),1)
        start_sit_beta_gamma = first_peak(moving_average(df_red.loc[(df_red['relative_timestamp'] >= t_start_turn2) & (df_red['relative_timestamp'] <= t_end_beta_gamma), 'der_beta_gamma']),1)
        t_end_stand_beta_gamma = df_red.at[start_beta_gamma + end_stand_beta_gamma, 'relative_timestamp']
        t_start_sit_beta_gamma = df.at[start_turn2 + start_sit_beta_gamma, 'relative_timestamp']

        end_stand_rot = last_peak(moving_average(df_red.loc[df_red['relative_timestamp'] >= t_start_rot, 'rotRate_beta_gamma']), 70)
        start_sit_rot = first_peak(moving_average(df_red.loc[(df_red['relative_timestamp'] >= t_start_turn2) & (df_red['relative_timestamp'] <= t_end_rot), 'rotRate_beta_gamma']), 70)
        t_end_stand_rot = df_red.at[start_rot + end_stand_rot, 'relative_timestamp']
        t_start_sit_rot = df.at[start_turn2 + start_sit_rot, 'relative_timestamp']

        # final times (adjust +-0.2 because the peak in the derivative/rotRate is around the center of the increase/decrease)
        t_start = np.mean([t_start_beta_gamma - 0.2, t_new_start_der, t_new_start_all, t_start_rot - 0.2])
        t_end = np.mean([t_end_beta_gamma + 0.2, t_new_end_der, t_new_end_all, t_end_rot + 0.2])
        t_end_stand = np.mean([t_end_stand_beta_gamma, t_end_stand_rot]) + 0.2
        t_start_sit = np.mean([t_start_sit_beta_gamma, t_start_sit_rot]) - 0.2

        return t_start, t_end_stand, t_start_turn, t_end_turn, t_start_turn2, t_end_turn2, t_start_sit, t_end

choice = '1_1'

df_plot = df_fusion[choice]
res = full_algo(df_plot)
if res is None:
    print(res)
else:
    (t_start, t_end_stand, t_start_turn, t_end_turn, t_start_turn2, t_end_turn2, t_start_sit, t_end) = res

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(df_plot["relative_timestamp"], df_plot["sqrt(X²+Y²+Z²)"],
         label="Motion (m/s²)", color="blue", linestyle="-")

ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Acceleration (m/s²)", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")

ax1.axvspan(t_start, t_end, color="orange", alpha=0.2, label="Total duration")
ax1.axvspan(t_start_turn, t_end_turn, color="limegreen", alpha=0.5, label="First turn")
ax1.axvspan(t_start_turn2, t_end_turn2, color="darkgreen", alpha=0.5, label="Second turn")
ax1.axvspan(t_start, t_end_stand, color="red", alpha=0.5, label="First turn")
ax1.axvspan(t_start_sit, t_end, color="pink", alpha=0.5, label="Second turn")


ax3 = ax1.twinx()
ax3.plot(df_plot["relative_timestamp"], df_plot["alpha"], label="Alpha (°)", color="red", linestyle="--")
ax3.plot(df_plot["relative_timestamp"], df_plot["beta"], label="Beta (°)", color="green", linestyle="-.")
ax3.plot(df_plot["relative_timestamp"], df_plot["gamma"], label="Gamma (°)", color="purple", linestyle=":")

plt.xticks(rotation=45)

ax1.grid()

ax1.legend(loc="upper left")
ax3.legend(loc="lower right")

plt.title("Results on motion and orientation")

plt.show()

# loop for all tests
algo_results = {}

for key, df_final in df_fusion.items():
        result = full_algo(df_final)
        if isinstance(result, str):
            print(f"erreur avec {key} :",result)
        else:
            t_start, t_end_stand, t_start_turn, t_end_turn, t_start_turn2, t_end_turn2, t_start_sit, t_end = result
            algo_results[key] = {
            "t_start": t_start,
            "t_end_stand": t_end_stand,
            "t_start_turn": t_start_turn,
            "t_end_turn": t_end_turn,
            "t_start_turn2": t_start_turn2,
            "t_end_turn2": t_end_turn2,
            "t_start_sit": t_start_sit,
            "t_end": t_end
            }
            print(f"Time for {key} :",t_end-t_start)
