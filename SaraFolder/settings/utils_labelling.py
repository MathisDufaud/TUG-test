import numpy as np
import scipy.signal as signal

from SaraFolder.settings.utils_parkaapp import utils_parkapp


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
    # Window size starts at 100 (approx 1.7 seconds at 60Hz)
    i = 0
    while i <= len(data) - window_size:
        window = data[i:i + window_size]
        amplitude = np.max(window) - np.min(window)

        if amplitude >= threshold:
            center = i + window_size // 2

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
    """
    Find the end index of a phase where the signal (DERIVATIVE!) is close to zero for at least `min_duration` samples.
    Args:
        data: Derivative signal.
        min_duration: Minimum duration (in samples) of the near-zero phase to be considered valid.
        k: Threshold factor to determine "close to zero" based on the data's amplitude.
    Returns:
        The end index of the near-zero phase. If no such phase is found, returns 0.
    """
    end_index = 0

    data = np.array(data)
    # Compute a dynamic threshold = fraction k of the signal’s amplitude range.
    threshold = (np.max(data)-np.min(data)) * k

    # Boolean mask: True where the signal is close to zero.
    near_zero = np.abs(data) < threshold

    enum_data = tuple(enumerate(near_zero))
    reverse_data = reversed(enum_data)

    count = 0  # keeps track of consecutive near-zero samples

    # Finding near-zero regions end index
    # If streak length ≥ min_duration, return the end index of that near-zero region
    for i, is_near in reverse_data:
        if is_near and i!=0:
            count += 1
        else:
            if count >= min_duration:
                end_index = i+count
                return end_index
            count = 0

    return end_index

def find_zero_phase_end_reverse2(data, min_duration=30, k=0.04):

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
    print("FS: ", np.round(df.shape[0] /((df.iloc[-1, 0] - df.iloc[0,0])/1000), 2))

    # find the 2 turns. 20 samples correspond approx to 1/3 seconds
    alpha_ma = utils_parkapp.moving_average(df['alpha'], 20)

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

        # Subset after second turn end
        df_test = df_red.loc[(df_red['relative_timestamp'] >= t_end_turn2 - 1)].copy()
        if df_test.empty:
            return "df_test empty"

        df_test.reset_index(drop=True, inplace=True)

        # --- Detect zero-phase regions (derivative & all) ---
        # Finding index before first turn and after the last turn
        new_start_der = find_zero_phase_end2(df_red.loc[df_red['relative_timestamp'] <= t_start_turn, 'derivative'])
        new_end_der = find_zero_phase_end_reverse2(df_test['derivative'], 20) # Similar to previous, but scans forward instead of backward.
        t_new_start_der = df_red.at[new_start_der, 'relative_timestamp']
        t_new_end_der = df_test.at[new_end_der, 'relative_timestamp']

        # Do the same for the “all” signal (with stronger threshold).
        new_start_all = find_zero_phase_end2(
            utils_parkapp.moving_average(df_red.loc[df_red['relative_timestamp'] <= t_start_turn, 'all']), 20, k=0.15)
        new_end_all = find_zero_phase_end_reverse2(utils_parkapp.moving_average(df_test['all']), 20, k=0.15)
        t_new_start_all = df_red.at[new_start_all, 'relative_timestamp']
        t_new_end_all = df_test.at[new_end_all, 'relative_timestamp']

        # Find first/last peaks in der_beta_gamma and rotRate_beta_gamma
        start_beta_gamma = first_peak(
            utils_parkapp.moving_average(df_red.loc[df_red['relative_timestamp'] <= t_start_turn, 'der_beta_gamma']))
        end_beta_gamma = last_peak(utils_parkapp.moving_average(df_test['der_beta_gamma']))
        t_start_beta_gamma = df_red.at[start_beta_gamma, 'relative_timestamp']
        t_end_beta_gamma = df_test.at[end_beta_gamma, 'relative_timestamp']
        start_rot = first_peak(utils_parkapp.moving_average(df_red.loc[df_red['relative_timestamp'] <= t_start_turn, 'rotRate_beta_gamma']), 75)
        end_rot = last_peak(utils_parkapp.moving_average(df_test['rotRate_beta_gamma']), 75)
        t_start_rot = df_red.at[start_rot, 'relative_timestamp']
        t_end_rot = df_test.at[end_rot, 'relative_timestamp']

        # end of standing and start of sitting
        end_stand_beta_gamma = last_peak(utils_parkapp.moving_average(df_red.loc[df_red['relative_timestamp'] >= t_start_beta_gamma, 'der_beta_gamma']), 1)
        start_sit_beta_gamma = first_peak(utils_parkapp.moving_average(df_red.loc[(df_red['relative_timestamp'] >= t_start_turn2) & (df_red['relative_timestamp'] <= t_end_beta_gamma), 'der_beta_gamma']), 1)
        t_end_stand_beta_gamma = df_red.at[start_beta_gamma + end_stand_beta_gamma, 'relative_timestamp']
        t_start_sit_beta_gamma = df.at[start_turn2 + start_sit_beta_gamma, 'relative_timestamp']

        end_stand_rot = last_peak(utils_parkapp.moving_average(df_red.loc[df_red['relative_timestamp'] >= t_start_rot, 'rotRate_beta_gamma']), 70)
        start_sit_rot = first_peak(utils_parkapp.moving_average(df_red.loc[(df_red['relative_timestamp'] >= t_start_turn2) & (df_red['relative_timestamp'] <= t_end_rot), 'rotRate_beta_gamma']), 70)
        t_end_stand_rot = df_red.at[start_rot + end_stand_rot, 'relative_timestamp']
        t_start_sit_rot = df.at[start_turn2 + start_sit_rot, 'relative_timestamp']

        # Final times (adjust +-0.2 because the peak in the derivative/rotRate is around the center of the increase/decrease)
        t_start = np.mean([t_start_beta_gamma - 0.2, t_new_start_der, t_new_start_all, t_start_rot - 0.2])
        t_end = np.mean([t_end_beta_gamma + 0.2, t_new_end_der, t_new_end_all, t_end_rot + 0.2])
        t_end_stand = np.mean([t_end_stand_beta_gamma, t_end_stand_rot]) + 0.2
        t_start_sit = np.mean([t_start_sit_beta_gamma, t_start_sit_rot]) - 0.2

        return t_start, t_end_stand, t_start_turn, t_end_turn, t_start_turn2, t_end_turn2, t_start_sit, t_end


def looping_tests(all_tests):
    # TODO: keep track of what is not computed (missed turns, no data...)
    for test in all_tests:
        result = full_algo(test.raw_data)
        test.results = {}
        if isinstance(result, str):
            print(f"erreur avec {test.test_id} :", result)
        else:
            t_start, t_end_stand, t_start_turn, t_end_turn, t_start_turn2, t_end_turn2, t_start_sit, t_end = result
            algo_results = {
                "t_start": t_start,
                "t_end_stand": t_end_stand,
                "t_start_turn": t_start_turn,
                "t_end_turn": t_end_turn,
                "t_start_turn2": t_start_turn2,
                "t_end_turn2": t_end_turn2,
                "t_start_sit": t_start_sit,
                "t_end": t_end
            }
            test.results['labelling'] = algo_results

    return all_tests