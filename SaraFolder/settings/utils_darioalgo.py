"""
function getTUGDuration4(tugData) {
    let orientationStats = new RollingStats()

    const PEAK_WINDOW_SIZE = 1000 // ms in which we look for orientation derivative peaks
    const waveMultiplierThreshold = 1.5 // multiplier of the mean orientation derivative to consider a peak as a wave

    let peaks = []
    let lastPeakMs = -PEAK_WINDOW_SIZE
    let lastPeakValue = -1
    let startingIndex = 60 // discard the first 60 rows (about 1s)
    let inAWave = false

    for (let i = 0; i<tugData.length - 4; i++) {
        orientationStats.addValue(tugData[i].orADerMean)
        if (i > startingIndex) {
            let tugDataRow = tugData[i]
            // a peak is detected if the value in the middle of the window is greater than the values before and after it
            if (tugDataRow.orADerMean >= tugData[i-1].orADerMean &&
                tugDataRow.orADerMean >= tugData[i-2].orADerMean &&
                tugDataRow.orADerMean >= tugData[i-3].orADerMean &&
                tugDataRow.orADerMean >= tugData[i-4].orADerMean &&
                tugDataRow.orADerMean >= tugData[i+1].orADerMean &&
                tugDataRow.orADerMean >= tugData[i+2].orADerMean &&
                tugDataRow.orADerMean >= tugData[i+3].orADerMean &&
                tugDataRow.orADerMean >= tugData[i+4].orADerMean)
            {

                // candidate peak detected
                if ((lastPeakMs >0) && (inAWave || (tugDataRow.msFromStart - lastPeakMs < PEAK_WINDOW_SIZE))){
                    // this is not a new candidate peak, but a continuation of the previous one
                    // this happens when there was a previous peak, and either we are in a "wave"
                    // (i.e. the orientation derivative is above the mean) or the time since the
                    // last peak is less than the PEAK_WINDOW_SIZE
                    if (tugDataRow.orADerMean > lastPeakValue) {
                        lastPeakValue = tugDataRow.orADerMean
                        lastPeakMs = tugDataRow.msFromStart
                        peaks[peaks.length - 1] = {
                            ms: lastPeakMs,
                            value: lastPeakValue
                        }
                    }
                } else {
                    // new candidate peak detected
                    lastPeakMs = tugDataRow.msFromStart
                    lastPeakValue = tugDataRow.orADerMean
                    peaks.push({
                        ms: lastPeakMs,
                        value: lastPeakValue
                    })
                }

                if (tugDataRow.orADerMean > waveMultiplierThreshold * orientationStats.getMean()) {
                    // this last peak was above the mean, so we are in a wave
                    inAWave = true
                }
            }
            if (tugDataRow.orADerMean < waveMultiplierThreshold * orientationStats.getMean()) {
                // if (inAWave) console.log(`Wave ended at ${tugDataRow.msFromStart} ms with value ${tugDataRow.orADerMean}, threshold: ${waveMultiplierThreshold * orientationStats.getMean()}`)
                // we are not in a wave anymore
                inAWave = false
            }
        }
    }

    let searchStartMs = tugData[0].msFromStart
    var searchEndMs = tugData[tugData.length - 1].msFromStart

    if (peaks.length < 2) {
        console.warn(`Not enough orientation derivative peaks found: ${peaks.length}.`)
    } else {
        // console.log('all peaks', peaks)
        let goodQuality = true


        // find the two highest peaks, but discard any peak that is earlier than a third of the total duration
        peaks.sort((a, b) => b.value - a.value)
        let peak1, peak2
        for (let i=0; i<peaks.length; i++) {
            if (peaks[i].ms > tugData[tugData.length - 1].msFromStart / 3) {
                if (!peak1) {
                    peak1 = peaks[i]
                } else if (!peak2) {
                    peak2 = peaks[i]
                    break
                }
            }
        }

        if (peak1 && peak2) {
            if (peak1.ms > peak2.ms) {
                // swap peaks if needed
                [peak1, peak2] = [peak2, peak1]
            }

            // identify quality of the measurement
            // console.log(`Found ${peaks.length} peaks`)
            let meanOrientationDerivative = orientationStats.getMean()
            // console.log(`Mean orientation derivative: ${meanOrientationDerivative.toFixed(2)}`)
            let pam = peaks.filter(p => (p.ms > 2000) && (p.value > 1.5* meanOrientationDerivative))
            // console.log(pam)
            let peaksAboveMean = pam.length
            // console.log(`Found ${peaksAboveMean} peaks above mean of ${peaks.length} peaks`)
            let peaksAboveMeanPercentage = (peaksAboveMean / peaks.length) * 100
            // console.log(`Peaks above mean percentage: ${peaksAboveMeanPercentage.toFixed(2)}%`)
            // ratio between peak 1 and mean
            let peak1Ratio = peak1.value / meanOrientationDerivative
            // console.log(`Peak 1 ratio to mean: ${peak1Ratio.toFixed(2)}`)
            // ratio between peak 2 and mean
            let peak2Ratio = peak2.value / meanOrientationDerivative
            // console.log(`Peak 2 ratio to mean: ${peak2Ratio.toFixed(2)}`)


            if (peaksAboveMeanPercentage > 50 || peak1Ratio < 2 || peak2Ratio < 2) {
                goodQuality = false
            }

            let backwardGaitDuration = peak2.ms - peak1.ms
            const standingSittingTime = 2000

            let veryslowtest = backwardGaitDuration > 4000

            if (goodQuality && !veryslowtest) {
                // we hypothesize that the forward and backward gait durations are similar
                let searchStartMs = (peak1.ms - backwardGaitDuration - standingSittingTime)
                if (searchStartMs < 0) {
                    searchStartMs = 0
                }
                searchEndMs = (peak2.ms + standingSittingTime)
                if (searchEndMs > tugData[tugData.length-1].msFromStart) {
                    searchEndMs = tugData[tugData.length-1].msFromStart
                }
                console.log(`->Good quality measurement, searching between ${searchStartMs} and ${searchEndMs}, backward gait duration: ${backwardGaitDuration} ms`)
            }
        }
    }


    const activityThreshold = 0.5

    let pastVal = 0
    let startMs = 0
    let endMs = 0

    for (let i =0; i< tugData.length; i++) {
        let line = tugData[i]
        if ((line.msFromStart>= searchStartMs) && (line.msFromStart <= searchEndMs)) {
            if ((line.accGMagnitudeVar > activityThreshold) && (pastVal < activityThreshold) && (startMs == 0)){
                startMs = line.msFromStart
            }

            // detect end of activity
            if ((line.msFromStart > 3000) && (line.accGMagnitudeVar < activityThreshold) && (pastVal > activityThreshold) && (startMs != 0)) {
                endMs = line.msFromStart
            }

            // if end of activity was detected, but the acceleration magnitude is still above the threshold,
            // we extend the end time to the last row with acceleration magnitude above the threshold
            // if (endMs != 0 && line.accGMagnitudeVar > activityThreshold) {
            //     endMs = line.msFromStart
            // }
            pastVal = line.accGMagnitudeVar
        }
    }
    if (startMs == 0) {
        startMs = searchStartMs
    }
    if (endMs == 0) {
        endMs = searchEndMs
    }

    return {
        algoName: 'combined',
        startMs: startMs,
        endMs: endMs,
        duration: endMs - startMs
    }
}

await benchmarkAlgo(getTUGDuration4)
"""

import numpy as np
import pandas as pd


class RollingStats:
    def __init__(self):
        self.values = []

    def add_value(self, value):
        self.values.append(value)

    def get_mean(self):
        return np.mean(self.values) if self.values else 0

def add_tug_features(df, window_size_1s=60, window_size_half_s=30):
    """
    Add orADerMean and accGMagnitudeVar columns to the dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        Must contain columns: 'msFromStart', 'accGX', 'accGY', 'accGZ', 'alpha'
    window_size_1s : int
        Window size for acceleration features (default 60 rows ≈ 1 second)
    window_size_half_s : int
        Window size for orientation derivative (default 30 rows ≈ 0.5 seconds)

    Returns:
    --------
    pandas.DataFrame with added columns: 'accGMagnitude', 'accGMagnitudeVar',
                                         'orADer', 'orADerMean'
    """

    df = df.copy()

    # Calculate acceleration magnitude from gravity-corrected accelerometer
    if 'accGX' in df.columns:
        df['accGMagnitude'] = np.sqrt(df['accGX'] ** 2 + df['accGY'] ** 2 + df['accGZ'] ** 2)
    else:
        df['accGMagnitude'] = np.sqrt(df['accG.x'] ** 2 + df['accG.y'] ** 2 + df['accG.z'] ** 2)

    # Calculate rolling variance of acceleration magnitude (1 second window)
    df['accGMagnitudeVar'] = df['accGMagnitude'].rolling(
        window=window_size_1s,
        min_periods=1
    ).var()

    # Calculate orientation derivative (rate of change of alpha angle)
    # First calculate time differences in seconds
    df['deltaTime'] = df['msFromStart'].diff() / 1000.0  # convert ms to seconds

    # Calculate alpha angle differences
    df['deltaAlpha'] = df['alpha'].diff()

    # Calculate derivative (degrees per second)
    # Avoid division by zero
    df['orADer'] = np.abs(df['deltaAlpha'] / df['deltaTime'].replace(0, np.nan))

    # Fill NaN values in first row with 0
    df['orADer'] = df['orADer'].fillna(0)

    # Calculate rolling mean of orientation derivative (0.5 second window)
    df['orADerMean'] = df['orADer'].rolling(
        window=window_size_half_s,
        min_periods=1
    ).mean()

    # Clean up temporary columns
    df = df.drop(columns=['deltaTime', 'deltaAlpha'])

    return df


# Usage:
# df_processed = add_tug_features(df_start)
# result = get_TUG_duration_4(df_processed)
def find_peaks_algo(df):
    # Convert dataframe to list of dicts for easier iteration
    tug_data = df.to_dict('records')

    PEAK_WINDOW_SIZE = 1000  # ms in which we look for orientation derivative peaks

    wave_multiplier_threshold = 1.5  # multiplier of the mean orientation derivative
    orientation_stats = RollingStats()

    peaks = []
    last_peak_ms = -PEAK_WINDOW_SIZE
    last_peak_value = -1
    starting_index = 60  # discard the first 60 rows (about 1s)
    in_a_wave = False

    # Find peaks
    for i in range(len(tug_data) - 4):
        orientation_stats.add_value(tug_data[i]['orADerMean'])

        if i > starting_index:
            tug_data_row = tug_data[i]

            # A peak is detected if the value in the middle is greater than values before and after
            is_peak = (
                    tug_data_row['orADerMean'] >= tug_data[i - 1]['orADerMean'] and
                    tug_data_row['orADerMean'] >= tug_data[i - 2]['orADerMean'] and
                    tug_data_row['orADerMean'] >= tug_data[i - 3]['orADerMean'] and
                    tug_data_row['orADerMean'] >= tug_data[i - 4]['orADerMean'] and
                    tug_data_row['orADerMean'] >= tug_data[i + 1]['orADerMean'] and
                    tug_data_row['orADerMean'] >= tug_data[i + 2]['orADerMean'] and
                    tug_data_row['orADerMean'] >= tug_data[i + 3]['orADerMean'] and
                    tug_data_row['orADerMean'] >= tug_data[i + 4]['orADerMean']
            )

            if is_peak:
                # Candidate peak detected
                if (last_peak_ms > 0) and (
                        in_a_wave or (tug_data_row['msFromStart'] - last_peak_ms < PEAK_WINDOW_SIZE)):
                    # Continuation of previous peak
                    if tug_data_row['orADerMean'] > last_peak_value:
                        last_peak_value = tug_data_row['orADerMean']
                        last_peak_ms = tug_data_row['msFromStart']
                        peaks[-1] = {
                            'ms': last_peak_ms,
                            'value': last_peak_value
                        }
                else:
                    # New candidate peak
                    last_peak_ms = tug_data_row['msFromStart']
                    last_peak_value = tug_data_row['orADerMean']
                    peaks.append({
                        'ms': last_peak_ms,
                        'value': last_peak_value
                    })

                if tug_data_row['orADerMean'] > wave_multiplier_threshold * orientation_stats.get_mean():
                    in_a_wave = True

            if tug_data_row['orADerMean'] < wave_multiplier_threshold * orientation_stats.get_mean():
                in_a_wave = False

    search_start_ms = tug_data[0]['msFromStart']
    search_end_ms = tug_data[-1]['msFromStart']

    if len(peaks) < 2:
        print(f"Warning: Not enough orientation derivative peaks found: {len(peaks)}")
        peak1, peak2 = None, None
    else:
        good_quality = True

        # Find the two highest peaks, discard any peak earlier than 1/3 of total duration
        peaks_sorted = sorted(peaks, key=lambda x: x['value'], reverse=True)
        peak1, peak2 = None, None

        for peak in peaks_sorted:
            if peak['ms'] > tug_data[-1]['msFromStart'] / 3:
                if peak1 is None:
                    peak1 = peak
                elif peak2 is None:
                    peak2 = peak
                    break

        if peak1 and peak2:
            # Swap peaks if needed (ensure peak1 comes before peak2)
            if peak1['ms'] > peak2['ms']:
                peak1, peak2 = peak2, peak1

            # Identify quality of measurement
            mean_orientation_derivative = orientation_stats.get_mean()
            peaks_above_mean = [
                p for p in peaks
                if (p['ms'] > 2000) and (p['value'] > 1.5 * mean_orientation_derivative)
            ]
            peaks_above_mean_percentage = (len(peaks_above_mean) / len(peaks)) * 100
            peak1_ratio = peak1['value'] / mean_orientation_derivative
            peak2_ratio = peak2['value'] / mean_orientation_derivative

            if peaks_above_mean_percentage > 50 or peak1_ratio < 2 or peak2_ratio < 2:
                good_quality = False

            backward_gait_duration = peak2['ms'] - peak1['ms']
            standing_sitting_time = 2000
            very_slow_test = backward_gait_duration > 4000

            if good_quality and not very_slow_test:
                # Hypothesize that forward and backward gait durations are similar
                search_start_ms = peak1['ms'] - backward_gait_duration - standing_sitting_time
                if search_start_ms < 0:
                    search_start_ms = 0

                search_end_ms = peak2['ms'] + standing_sitting_time
                if search_end_ms > tug_data[-1]['msFromStart']:
                    search_end_ms = tug_data[-1]['msFromStart']

                print(f"->Good quality measurement, searching between {search_start_ms} and {search_end_ms}, "
                      f"backward gait duration: {backward_gait_duration} ms")

    return search_start_ms, search_end_ms, peak1, peak2, tug_data


def get_TUG_duration_4(df):
    try:
        """
        Calculate TUG (Timed Up and Go) test duration from accelerometer data.
    
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with columns including 'msFromStart', 'orADerMean', 'accGMagnitudeVar'
            Note: You'll need to calculate these derived columns before calling this function
    
        Returns:
        --------
        dict with keys: 'algoName', 'startMs', 'endMs', 'duration'
        """
        df = add_tug_features(df)

        search_start_ms, search_end_ms, peak1, peak2, tug_data = find_peaks_algo(df)

        # Detect activity start and end
        activity_threshold = 0.5
        past_val = 0
        start_ms = 0
        end_ms = 0

        for line in tug_data:
            if search_start_ms <= line['msFromStart'] <= search_end_ms:
                # Detect start of activity
                if (line['accGMagnitudeVar'] > activity_threshold and
                        past_val < activity_threshold and
                        start_ms == 0):
                    start_ms = line['msFromStart']

                # Detect end of activity
                if (line['msFromStart'] > 3000 and
                        line['accGMagnitudeVar'] < activity_threshold and
                        past_val > activity_threshold and
                        start_ms != 0):
                    end_ms = line['msFromStart']

                past_val = line['accGMagnitudeVar']

        if start_ms == 0:
            start_ms = search_start_ms
        if end_ms == 0:
            end_ms = search_end_ms

        return {
            'startMs': start_ms,
            'endMs': end_ms,
            'duration': end_ms - start_ms
        }
    except:
        print("bug")
