import numpy as np
import scipy.signal as signal

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
from SaraFolder.settings import classes, utils_darioalgo, running_settings, utils_dataquality
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
def start_change(base_data, window_size=100, threshold=120, show_info=False):
    data = np.array(base_data)
    detected_segments = []
    # Window size starts at 100 (approx 1.7 seconds at 60Hz)
    i = 0
    while i <= len(data) - window_size:
        window = data[i:i + window_size]
        amplitude = np.max(window) - np.min(window)

        if amplitude >= threshold:
            if show_info:
                print(f"Detected window at index {i} with amplitude {amplitude}, higher than threshold {threshold}")

            center = i + window_size // 2

            # going left from center
            start = center
            while start-1 > 0 and np.abs(data[start] - data[start - 1]) > 0.3:
                start -= 1

            # going right from center
            end = center
            while end + 1 < len(data) - 1 and np.abs(data[end] - data[end + 1]) > 0.3:
                end += 1

            if end - start >= 45:
                detected_segments.append((start, end))
                if show_info:
                    print(f"Segment with turn potential lasts long: {end - start} samples, higher than 45 samples")

            i = end
        i += 1

    if len(detected_segments) >= 2:
        # sorting by amplitude+length
        if show_info:
            print(f"Found more than two samples: {len(detected_segments)}, selecting the best two based on amplitude and length.")

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

    if show_info:
        print(f"Turns not found yet, "
              f"lowering threshold to {threshold-10} and "
              f"increasing window size to {window_size+10}")

    return start_change(base_data, window_size=window_size+10, threshold=threshold-10)

def start_change_optimization(base_data, window_size=100, threshold=120):
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

def find_zero_phase_end2(data, min_duration=30, k=0.05):
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


def plot_faulty_signal(df_plot, name, quality=None, stats=None, method='labelling'):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df_plot["relative_timestamp"], df_plot["sqrt(X²+Y²+Z²)"],
             label="Motion (m/s²)", color="blue", linestyle="-")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Acceleration (m/s²)", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax3 = ax1.twinx()
    ax3.plot(df_plot["relative_timestamp"], df_plot["alpha"], label="Alpha (°)", color="red", linestyle="--",
             linewidth=3)
    ax3.plot(df_plot["relative_timestamp"], df_plot["beta"], label="Beta (°)", color="green", linestyle="-.",
             linewidth=3)
    ax3.plot(df_plot["relative_timestamp"], df_plot["gamma"], label="Gamma (°)", color="purple", linestyle=":",
             linewidth=3)

    ax3.plot(df_plot["relative_timestamp"], df_plot["rotRate.alpha"], label="RotRate Alpha (°)", color='darkred',
             linestyle="--")
    ax3.plot(df_plot["relative_timestamp"], df_plot["rotRate.beta"], label="RotRate Beta (°)", color='darkgreen',
             linestyle="-.")
    ax3.plot(df_plot["relative_timestamp"], df_plot["rotRate.gamma"], label="RotRate Gamma (°)", color='darkred',
             linestyle=":")

    ax1.grid()
    ax1.legend(loc="upper left")
    ax3.legend(loc="lower right")

    #plt.title(f"TUG raw data, "
              #f"{user_id}_{session_id}, "
              #f"GTmanual = {np.round(gtm, 2)},"
              #f"GTgwalk = {np.round(gtg, 2)}")

    # Add quality information box
    if quality is not None and 'Quality' in name:
        qualitytot = (quality['basic'] + quality[method]).split('/')
        # Clean up the list
        qualitytot = [q.strip() for q in qualitytot if q.strip() and q.strip() != 'okresults']

        # Determine quality text and color
        if not qualitytot or qualitytot == ['']:
            quality_text = 'OK Quality'
            box_color = '#90EE90'  # Light green
        else:
            quality_text = 'Quality Issues:\n' + '\n'.join([f'• {q}' for q in qualitytot])
            box_color = '#FFB6C6'  # Light red

        # Add text box in upper right corner
        props = dict(boxstyle='round', facecolor=box_color, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.text(0.98, 0.98, quality_text, transform=ax1.transAxes,
                 fontsize=9, verticalalignment='top', horizontalalignment='right',
                 bbox=props)

    plt.title(name)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return fig


def check_emptiness_3sec(df, dataset_id):

    if df.empty:
        quality = 'Empty df (1)/'
        return "empty df", quality
    # remove first 3 sec
    if dataset_id == 'parkapp': #or dataset_id == 'pisa_new':
        df = df.loc[(df['relative_timestamp'] >= 3)]
        if df.empty:
            quality = 'Empty df (2)/'
            return "empty df after removing 3 sec", quality

    return df, ''


def compute_timestamp_quality(timestamps, weights=None):
    """
    Compute quality metrics for timestamp estimates.

    Parameters:
    -----------
    timestamps : array-like
        List of timestamp estimates to average
    weights : array-like, optional
        Weights for each timestamp (e.g., based on signal quality)

    Returns:
    --------
    dict with quality metrics
    """
    timestamps = np.array(timestamps)

    # Remove NaN values
    valid_timestamps = timestamps[~np.isnan(timestamps)]

    if len(valid_timestamps) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'confidence': 0.0,
            'agreement': 0.0,
            'quality_score': 0.0,
            'n_valid': 0
        }

    if len(valid_timestamps) == 1:
        return {
            'mean': valid_timestamps[0],
            'std': 0.0,
            'confidence': 0.5,  # Medium confidence with single estimate
            'agreement': 1.0,
            'quality_score': 0.5,
            'n_valid': 1
        }

    # Basic statistics
    mean_time = np.mean(valid_timestamps)
    std_time = np.std(valid_timestamps)

    # 1. Coefficient of Variation (CV) - normalized dispersion
    cv = std_time / mean_time if mean_time != 0 else np.inf

    # 2. Agreement score - how close timestamps are to each other
    # Lower std relative to time scale = better agreement
    max_deviation = np.max(np.abs(valid_timestamps - mean_time))
    agreement_score = 1.0 / (1.0 + max_deviation / 100)  # Normalize by 100ms

    # 3. Confidence based on standard deviation
    # High confidence if std < 50ms, low if std > 200ms
    confidence = np.exp(-std_time / 100)  # Exponential decay

    # 4. Overall quality score (0-1 scale)
    quality_score = (agreement_score * 0.4 +
                     confidence * 0.4 +
                     (len(valid_timestamps) / len(timestamps)) * 0.2)

    return {
        'mean': mean_time,
        'std': std_time,
        'cv': cv,
        'max_deviation': max_deviation,
        'confidence': confidence,
        'agreement': agreement_score,
        'quality_score': quality_score,
        'n_valid': len(valid_timestamps),
        'n_total': len(timestamps)
    }

def compute_all_timestamp_qualities(t_start_beta_gamma, t_new_start_der, t_new_start_all, t_start_rot,
                                    t_end_beta_gamma, t_new_end_der, t_new_end_all, t_end_rot,
                                    t_end_stand_beta_gamma, t_end_stand_rot,
                                    t_start_sit_beta_gamma, t_start_sit_rot):
    """
    Compute quality metrics for all timestamp estimates.
    """

    # Start time
    t_start_inputs = [t_start_beta_gamma - 0.2, t_new_start_der, t_new_start_all, t_start_rot - 0.2]
    t_start_quality = compute_timestamp_quality(t_start_inputs)
    t_start = t_start_quality['mean']

    # End time
    t_end_inputs = [t_end_beta_gamma + 0.2, t_new_end_der, t_new_end_all, t_end_rot + 0.2]
    t_end_quality = compute_timestamp_quality(t_end_inputs)
    t_end = t_end_quality['mean']

    # End stand time
    t_end_stand_inputs = [t_end_stand_beta_gamma, t_end_stand_rot]
    t_end_stand_quality = compute_timestamp_quality([x + 0.2 for x in t_end_stand_inputs])
    t_end_stand = t_end_stand_quality['mean']

    # Start sit time
    t_start_sit_inputs = [t_start_sit_beta_gamma, t_start_sit_rot]
    t_start_sit_quality = compute_timestamp_quality([x - 0.2 for x in t_start_sit_inputs])
    t_start_sit = t_start_sit_quality['mean']

    results = {
        't_start': {
            'value': t_start,
            'quality': t_start_quality,
            'inputs': t_start_inputs
        },
        't_end': {
            'value': t_end,
            'quality': t_end_quality,
            'inputs': t_end_inputs
        },
        't_end_stand': {
            'value': t_end_stand,
            'quality': t_end_stand_quality,
            'inputs': [x + 0.2 for x in t_end_stand_inputs]
        },
        't_start_sit': {
            'value': t_start_sit,
            'quality': t_start_sit_quality,
            'inputs': [x - 0.2 for x in t_start_sit_inputs]
        }
    }

    return results

def print_quality_report(results):
    """
    Print a formatted quality report for timestamp estimates.
    """
    print("=" * 70)
    print("TIMESTAMP QUALITY REPORT")
    print("=" * 70)

    for name, data in results.items():
        q = data['quality']
        print(f"\n{name.upper()}:")
        print(f"  Estimated value: {data['value']:.2f} ms")
        print(f"  Standard deviation: {q['std']:.2f} ms")
        print(f"  Max deviation: {q['max_deviation']:.2f} ms")
        print(f"  Confidence: {q['confidence']:.3f} (0-1 scale)")
        print(f"  Agreement: {q['agreement']:.3f} (0-1 scale)")
        print(f"  Quality score: {q['quality_score']:.3f} (0-1 scale)")
        print(f"  Valid estimates: {q['n_valid']}/{q['n_total']}")

        # Quality assessment
        if q['quality_score'] > 0.8:
            assessment = "EXCELLENT"
        elif q['quality_score'] > 0.6:
            assessment = "GOOD"
        elif q['quality_score'] > 0.4:
            assessment = "FAIR"
        else:
            assessment = "POOR"
        print(f"  Assessment: {assessment}")

        # Show individual inputs
        print(f"  Input values: {[f'{x:.2f}' for x in data['inputs']]}")

    print("\n" + "=" * 70)

def full_algo(df, dataset_id, name, show_info=False):

    df, quality = check_emptiness_3sec(df, dataset_id)

    if isinstance(df, str):
        return df, quality

    df.reset_index(drop=True, inplace=True)

    # find the 2 turns. 20 samples correspond approx to 1/3 seconds
    alpha_ma = utils_parkapp.moving_average(df['alpha'], 20)
    try:
        result = start_change(alpha_ma, show_info=show_info)

        if result is None:
            quality = 'No turns found with classic approach/'
            if False:
                _ = plot_faulty_signal(df, 'Cant find turns: ' + name)
            print("No found turns with first approach")
            df = utils_darioalgo.add_tug_features(df)
            search_start_ms, search_end_ms, peak1, peak2, tug_data, quality_dario = utils_darioalgo.find_peaks_algo(df)

            if peak1 is not None and peak2 is not None:
                start_turn = peak1['ms']
                start_turn2 = peak2['ms']
                start_turn = df['relative_timestamp'][df['msFromStart'] == start_turn].index[0]
                start_turn2 = df['relative_timestamp'][df['msFromStart'] == start_turn2].index[0]
                end_turn = start_turn+90
                end_turn2 = start_turn2+90
                if end_turn2 > df.shape[0]:
                    end_turn2 = df.shape[0]-1

                quality = quality + quality_dario

            else:
                print("Still no turns found")
                quality_dario = quality + quality_dario + 'No found turns with second approach/'
                return "no turn found", quality_dario
        else:
            (start_turn,end_turn), (start_turn2,end_turn2) = result

        t_start_turn = df.at[start_turn, 'relative_timestamp']
        t_end_turn = df.at[end_turn, 'relative_timestamp']
        t_start_turn2 = df.at[start_turn2, 'relative_timestamp']
        t_end_turn2 = df.at[end_turn2, 'relative_timestamp']

        start_limit = max(df.at[0, 'relative_timestamp'], t_start_turn - (t_end_turn2 - t_start_turn) * 1.5)
        end_limit = min(df.at[df.index[-1], 'relative_timestamp'], t_end_turn2 + (t_end_turn2 - t_start_turn))

        df_red = df.loc[(df['relative_timestamp'] >= start_limit) & (df['relative_timestamp'] <= end_limit)].copy()

        if df_red.empty:
            quality = quality + 'Empty df (3)/'
            return "df_red empty", quality

        df_red.reset_index(drop=True, inplace=True)

        # Subset after second turn end
        df_testend = df_red.loc[(df_red['relative_timestamp'] >= t_end_turn2 - 1)].copy()
        df_teststart = df_red.loc[(df_red['relative_timestamp'] <= t_start_turn)].copy()
        if df_testend.empty:
            quality = quality + 'Empty df (4)'
            return "df_test empty", quality

        df_testend.reset_index(drop=True, inplace=True)

        # --- Detect zero-phase regions (derivative & all) ---
        # Finding index before first turn and after the last turn
        new_start_der = find_zero_phase_end2(df_red.loc[df_red['relative_timestamp'] <= t_start_turn, 'derivative'])
        new_end_der = find_zero_phase_end_reverse2(df_testend['derivative'], 20) # Similar to previous, but scans forward instead of backward.
        t_new_start_der = df_red.at[new_start_der, 'relative_timestamp']
        t_new_end_der = df_testend.at[new_end_der, 'relative_timestamp']

        # Do the same for the “all” signal (with stronger threshold).
        new_start_all = find_zero_phase_end2(
            utils_parkapp.moving_average(df_red.loc[df_red['relative_timestamp'] <= t_start_turn, 'all']), 20, k=0.15)
        new_end_all = find_zero_phase_end_reverse2(utils_parkapp.moving_average(df_testend['all']), 20, k=0.15)
        t_new_start_all = df_red.at[new_start_all, 'relative_timestamp']
        t_new_end_all = df_testend.at[new_end_all, 'relative_timestamp']

        # Find first/last peaks in der_beta_gamma and rotRate_beta_gamma
        start_beta_gamma = first_peak(
            utils_parkapp.moving_average(df_red.loc[df_red['relative_timestamp'] <= t_start_turn, 'der_beta_gamma']))
        end_beta_gamma = last_peak(utils_parkapp.moving_average(df_testend['der_beta_gamma']))
        t_start_beta_gamma = df_red.at[start_beta_gamma, 'relative_timestamp']
        t_end_beta_gamma = df_testend.at[end_beta_gamma, 'relative_timestamp']
        start_rot = first_peak(utils_parkapp.moving_average(df_red.loc[df_red['relative_timestamp'] <= t_start_turn, 'rotRate_beta_gamma']), 75)
        end_rot = last_peak(utils_parkapp.moving_average(df_testend['rotRate_beta_gamma']), 75)
        t_start_rot = df_red.at[start_rot, 'relative_timestamp']
        t_end_rot = df_testend.at[end_rot, 'relative_timestamp']

        # end of standing and start of sitting
        end_stand_beta_gamma = last_peak(utils_parkapp.moving_average(df_red.loc[df_red['relative_timestamp'] >= t_start_beta_gamma, 'der_beta_gamma']), 1)
        start_sit_beta_gamma = first_peak(utils_parkapp.moving_average(df_red.loc[(df_red['relative_timestamp'] >= t_start_turn2) &
                                                                                  (df_red['relative_timestamp'] <= t_end_beta_gamma), 'der_beta_gamma']), 1)
        t_end_stand_beta_gamma = df_red.at[start_beta_gamma + end_stand_beta_gamma, 'relative_timestamp']
        t_start_sit_beta_gamma = df.at[start_turn2 + start_sit_beta_gamma, 'relative_timestamp']

        end_stand_rot = last_peak(utils_parkapp.moving_average(df_red.loc[df_red['relative_timestamp'] >= t_start_rot, 'rotRate_beta_gamma']), 70)
        start_sit_rot = first_peak(utils_parkapp.moving_average(df_red.loc[(df_red['relative_timestamp'] >= t_start_turn2) &
                                                                           (df_red['relative_timestamp'] <= t_end_rot), 'rotRate_beta_gamma']), 70)
        t_end_stand_rot = df_red.at[start_rot + end_stand_rot, 'relative_timestamp']
        t_start_sit_rot = df.at[start_turn2 + start_sit_rot, 'relative_timestamp']

        # Final times (adjust +-0.2 because the peak in the derivative/rotRate is around the center of the increase/decrease)
        if show_info:
            results = compute_all_timestamp_qualities(
                t_start_beta_gamma, t_new_start_der, t_new_start_all, t_start_rot,
                t_end_beta_gamma, t_new_end_der, t_new_end_all, t_end_rot,
                t_end_stand_beta_gamma, t_end_stand_rot,
                t_start_sit_beta_gamma, t_start_sit_rot
            )
            print_quality_report(results)

            # # Access individual values and qualities
            # t_start = results['t_start']['value']
            # t_start_confidence = results['t_start']['quality']['confidence']

        t_start = np.mean([t_start_beta_gamma - 0.2, t_new_start_der, t_new_start_all, t_start_rot - 0.2])
        t_end = np.mean([t_end_beta_gamma + 0.2, t_new_end_der, t_new_end_all, t_end_rot + 0.2])
        t_end_stand = np.mean([t_end_stand_beta_gamma, t_end_stand_rot]) + 0.2
        t_start_sit = np.mean([t_start_sit_beta_gamma, t_start_sit_rot]) - 0.2

        quality = quality + 'okresults/'
    except:
        print(1)

    return (t_start, t_end_stand, t_start_turn, t_end_turn, t_start_turn2, t_end_turn2, t_start_sit, t_end), quality

def full_algo_param_optimization(df, dataset_id,
                                 window_size,
                                 max_alpha_amplitude,
                                 max_amplitutde_start_turn_sample,
                                 mcsfd):


    if df.empty:
        return "empty df"

    # remove first 3 sec
    if dataset_id == 'parkapp':
        df = df.loc[(df['relative_timestamp'] >= 3)]
        if df.empty:
            return "empty df after removing 3 sec"

    df.reset_index(drop=True, inplace=True)
    print("FS: ", np.round(df.shape[0] / ((df.iloc[-1, 0] - df.iloc[0, 0]) / 1000), 2))

    # find the 2 turns. 20 samples correspond approx to 1/3 seconds
    alpha_ma = utils_parkapp.moving_average(df['alpha'], 20)

    result = start_change_optimization(alpha_ma, window_size=100, threshold=120)

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
        df_testend = df_red.loc[(df_red['relative_timestamp'] >= t_end_turn2 - 1)].copy()
        df_teststart = df_red.loc[(df_red['relative_timestamp'] <= t_start_turn)].copy()
        if df_testend.empty:
            return "df_test empty"

        df_testend.reset_index(drop=True, inplace=True)

        # --- Detect zero-phase regions (derivative & all) ---
        # Finding index before first turn and after the last turn
        new_start_der = find_zero_phase_end2(df_red.loc[df_red['relative_timestamp'] <= t_start_turn, 'derivative'])
        new_end_der = find_zero_phase_end_reverse2(df_testend['derivative'], 20) # Similar to previous, but scans forward instead of backward.
        t_new_start_der = df_red.at[new_start_der, 'relative_timestamp']
        t_new_end_der = df_testend.at[new_end_der, 'relative_timestamp']

        # Do the same for the “all” signal (with stronger threshold).
        new_start_all = find_zero_phase_end2(
            utils_parkapp.moving_average(df_red.loc[df_red['relative_timestamp'] <= t_start_turn, 'all']), 20, k=0.15)
        new_end_all = find_zero_phase_end_reverse2(utils_parkapp.moving_average(df_testend['all']), 20, k=0.15)
        t_new_start_all = df_red.at[new_start_all, 'relative_timestamp']
        t_new_end_all = df_testend.at[new_end_all, 'relative_timestamp']

        # Find first/last peaks in der_beta_gamma and rotRate_beta_gamma
        start_beta_gamma = first_peak(
            utils_parkapp.moving_average(df_red.loc[df_red['relative_timestamp'] <= t_start_turn, 'der_beta_gamma']))
        end_beta_gamma = last_peak(utils_parkapp.moving_average(df_testend['der_beta_gamma']))
        t_start_beta_gamma = df_red.at[start_beta_gamma, 'relative_timestamp']
        t_end_beta_gamma = df_testend.at[end_beta_gamma, 'relative_timestamp']
        start_rot = first_peak(utils_parkapp.moving_average(df_red.loc[df_red['relative_timestamp'] <= t_start_turn, 'rotRate_beta_gamma']), 75)
        end_rot = last_peak(utils_parkapp.moving_average(df_testend['rotRate_beta_gamma']), 75)
        t_start_rot = df_red.at[start_rot, 'relative_timestamp']
        t_end_rot = df_testend.at[end_rot, 'relative_timestamp']

        # end of standing and start of sitting
        end_stand_beta_gamma = last_peak(utils_parkapp.moving_average(df_red.loc[df_red['relative_timestamp'] >= t_start_beta_gamma, 'der_beta_gamma']), 1)
        start_sit_beta_gamma = first_peak(utils_parkapp.moving_average(df_red.loc[(df_red['relative_timestamp'] >= t_start_turn2) &
                                                                                  (df_red['relative_timestamp'] <= t_end_beta_gamma), 'der_beta_gamma']), 1)
        t_end_stand_beta_gamma = df_red.at[start_beta_gamma + end_stand_beta_gamma, 'relative_timestamp']
        t_start_sit_beta_gamma = df.at[start_turn2 + start_sit_beta_gamma, 'relative_timestamp']

        end_stand_rot = last_peak(utils_parkapp.moving_average(df_red.loc[df_red['relative_timestamp'] >= t_start_rot, 'rotRate_beta_gamma']), 70)
        start_sit_rot = first_peak(utils_parkapp.moving_average(df_red.loc[(df_red['relative_timestamp'] >= t_start_turn2) &
                                                                           (df_red['relative_timestamp'] <= t_end_rot), 'rotRate_beta_gamma']), 70)
        t_end_stand_rot = df_red.at[start_rot + end_stand_rot, 'relative_timestamp']
        t_start_sit_rot = df.at[start_turn2 + start_sit_rot, 'relative_timestamp']

        # Final times (adjust +-0.2 because the peak in the derivative/rotRate is around the center of the increase/decrease)
        t_start = np.mean([t_start_beta_gamma - 0.2, t_new_start_der, t_new_start_all, t_start_rot - 0.2])
        t_end = np.mean([t_end_beta_gamma + 0.2, t_new_end_der, t_new_end_all, t_end_rot + 0.2])
        t_end_stand = np.mean([t_end_stand_beta_gamma, t_end_stand_rot]) + 0.2
        t_start_sit = np.mean([t_start_sit_beta_gamma, t_start_sit_rot]) - 0.2

        return t_start, t_end_stand, t_start_turn, t_end_turn, t_start_turn2, t_end_turn2, t_start_sit, t_end

def looping_tests(all_tests):
    for test in all_tests:
        plot = False
        test.plot_labelling(method='labelling', plot=plot)
    return all_tests


def labelling_method(test, show_info=False):
    result, quality = full_algo(test.processed_data, test.dataset_id, str(test.user_id) + '_' + str(test.session_id), show_info=show_info)
    test.results['labelling'] = None

    if not isinstance(result, str):
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

    else:
        print(result)
        test.results['labelling'] = result

    return test, quality

def labelling_method_optimization(test, method):

    param_opt = running_settings.param_opt_labelling_algo
    window_segment_turns = param_opt['window_segment_turns']
    max_alpha_amplitude = param_opt['max_alpha_amplitude']
    max_amplitutde_start_turn_sample = param_opt['max_amplitutde_start_turn_sample']
    MCSFD = param_opt['MCSFD']

    results_optimization = {}

    for i, w in enumerate(window_segment_turns):
        result = full_algo_param_optimization(test.processed_data, test.dataset_id,
                                              window_size=w,
                                              max_alpha_amplitude=max_alpha_amplitude,
                                              max_amplitutde_start_turn_sample=max_amplitutde_start_turn_sample,
                                              mcsfd=MCSFD
                                              )
        test.results[method] = None

        if not isinstance(result, str):
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
            results_optimization[i] = algo_results
        else:
            print(result)
            test.results['labelling'] = result
            results_optimization[i] = result

    return test

def darioalgo_method(test):
    result, quality = utils_darioalgo.get_TUG_duration_4(test.processed_data)
    test.results['darioalgo'] = None

    if isinstance(result, dict):
        algo_results = {
            "t_start": result['startMs']/1000,
            "t_end": result['endMs']/1000
        }
        test.results['darioalgo'] = algo_results
    else:
        print(result)
        test.results['darioalgo'] = result
    return test, quality


def compute_method(test, method, show_info=False):
    print(f"Computing method: {method}, for {test.user_id}_{test.session_id}")
    quality_0 = utils_dataquality.quality_assessment(test.processed_data, test.dataset_id)
    test.quality['basic'] = quality_0

    if method == 'labelling':
        test, quality_1 = labelling_method(test, show_info=show_info)
        test.quality[method] = quality_1

    if method == 'darioalgo':
        test, quality_1 = darioalgo_method(test)
        test.quality[method] = quality_1

    if method == 'ml':
        print("Ml method.")

    pass

def compute_method_optimization(test, method):

    if method == 'labelling':
        test = labelling_method_optimization(test)

    pass

def labelling_acrossall(all_tests, method):
    # Run method
    for t in all_tests:
        t.plot_labelling(method=method, plot=False)

    return None


def parameter_optimization_labelling(all_tests, method, eval_type, gttype, dataset, title):
    # Parameter optimization for labelling method
    for t in all_tests:
        t.plot_labelling(method=method, plot=False, optimization=True)

    return None


def observe_noresult_tests(norestests, method):
    print(f"The following {len(norestests)} tests were not succesfully processed: ")
    blacklist={}
    for test in norestests:
        test_id = str(test.user_id) + '_' + str(test.session_id)
        print(f"Test {test_id}: {test.results[method]}")
        blacklist[test_id] = test.results[method]
    return blacklist