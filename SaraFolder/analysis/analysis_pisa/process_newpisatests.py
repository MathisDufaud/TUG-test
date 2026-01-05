import os

import numpy as np
import pandas as pd

from SaraFolder.settings import utils_labelling, utils_plots, utils_evaluation, running_settings
from SaraFolder.settings.utils_parkaapp import utils_parkapp
from SaraFolder.settings.utils_synergy import utils_synloaders

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import matplotlib


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def plot_sensor_signals(acc, acc_unc, acc_total, gyro, orientation,
                        time_range=None, figsize=(16, 12)):
    """
    Plot all sensor signals in a comprehensive multi-panel figure.

    Parameters:
    -----------
    acc : DataFrame
        Accelerometer data (corrected)
    acc_unc : DataFrame
        Uncorrected accelerometer data
    acc_total : DataFrame
        Total accelerometer data
    gyro : DataFrame
        Gyroscope data
    orientation : DataFrame
        Orientation data (quaternions and Euler angles)
    time_range : tuple, optional
        (start, end) in seconds to zoom into specific time window
    figsize : tuple
        Figure size (width, height)
    """

    # Apply time range filter if specified
    if time_range is not None:
        start, end = time_range
        acc = acc[(acc['seconds_elapsed'] >= start) & (acc['seconds_elapsed'] <= end)]
        acc_unc = acc_unc[(acc_unc['seconds_elapsed'] >= start) & (acc_unc['seconds_elapsed'] <= end)]
        acc_total = acc_total[(acc_total['seconds_elapsed'] >= start) & (acc_total['seconds_elapsed'] <= end)]
        gyro = gyro[(gyro['seconds_elapsed'] >= start) & (gyro['seconds_elapsed'] <= end)]
        orientation = orientation[(orientation['seconds_elapsed'] >= start) & (orientation['seconds_elapsed'] <= end)]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(5, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Accelerometer (corrected) - 3 axes
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(acc['seconds_elapsed'], acc['x'], label='X', alpha=0.7, linewidth=1)
    ax1.plot(acc['seconds_elapsed'], acc['y'], label='Y', alpha=0.7, linewidth=1)
    ax1.plot(acc['seconds_elapsed'], acc['z'], label='Z', alpha=0.7, linewidth=1)
    ax1.set_ylabel('Acceleration', fontsize=10)
    ax1.set_title('Accelerometer', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Accelerometer (uncorrected) - 3 axes
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(acc_unc['seconds_elapsed'], acc_unc['x'], label='X', alpha=0.7, linewidth=1)
    ax2.plot(acc_unc['seconds_elapsed'], acc_unc['y'], label='Y', alpha=0.7, linewidth=1)
    ax2.plot(acc_unc['seconds_elapsed'], acc_unc['z'], label='Z', alpha=0.7, linewidth=1)
    ax2.set_ylabel('Acceleration', fontsize=10)
    ax2.set_title('Accelerometer (Uncalibrated)', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Accelerometer (total) - 3 axes
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(acc_total['seconds_elapsed'], acc_total['x'], label='X', alpha=0.7, linewidth=1)
    ax3.plot(acc_total['seconds_elapsed'], acc_total['y'], label='Y', alpha=0.7, linewidth=1)
    ax3.plot(acc_total['seconds_elapsed'], acc_total['z'], label='Z', alpha=0.7, linewidth=1)
    ax3.set_ylabel('Acceleration (Total)', fontsize=10)
    ax3.set_title('Accelerometer (Total)', fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Acceleration magnitude comparison
    ax4 = fig.add_subplot(gs[1, 1])
    acc_mag = np.sqrt(acc['x'] ** 2 + acc['y'] ** 2 + acc['z'] ** 2)
    acc_unc_mag = np.sqrt(acc_unc['x'] ** 2 + acc_unc['y'] ** 2 + acc_unc['z'] ** 2)
    acc_total_mag = np.sqrt(acc_total['x'] ** 2 + acc_total['y'] ** 2 + acc_total['z'] ** 2)

    ax4.plot(acc['seconds_elapsed'], acc_mag, label='Acc', alpha=0.7, linewidth=1)
    ax4.plot(acc_unc['seconds_elapsed'], acc_unc_mag, label='Acc Uncalibrated', alpha=0.7, linewidth=1)
    ax4.plot(acc_total['seconds_elapsed'], acc_total_mag, label='Acc Total', alpha=0.7, linewidth=1)
    ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, linewidth=1, label='1g')
    ax4.set_ylabel('Magnitude (g)', fontsize=10)
    ax4.set_title('Acceleration Magnitude', fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Gyroscope - 3 axes
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(gyro['seconds_elapsed'], gyro['x'], label='X', alpha=0.7, linewidth=1)
    ax5.plot(gyro['seconds_elapsed'], gyro['y'], label='Y', alpha=0.7, linewidth=1)
    ax5.plot(gyro['seconds_elapsed'], gyro['z'], label='Z', alpha=0.7, linewidth=1)
    ax5.set_ylabel('Angular velocity (rad/s)', fontsize=10)
    ax5.set_title('Gyroscope', fontweight='bold')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Gyroscope magnitude
    ax6 = fig.add_subplot(gs[2, 1])
    gyro_mag = np.sqrt(gyro['x'] ** 2 + gyro['y'] ** 2 + gyro['z'] ** 2)
    ax6.plot(gyro['seconds_elapsed'], gyro_mag, color='purple', alpha=0.7, linewidth=1)
    ax6.set_ylabel('Magnitude (rad/s)', fontsize=10)
    ax6.set_title('Gyroscope Magnitude', fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # 7. Euler angles (roll, pitch, yaw)
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.plot(orientation['seconds_elapsed'], np.rad2deg(orientation['roll']),
             label='Roll', alpha=0.7, linewidth=1)
    ax7.plot(orientation['seconds_elapsed'], np.rad2deg(orientation['pitch']),
             label='Pitch', alpha=0.7, linewidth=1)
    ax7.plot(orientation['seconds_elapsed'], np.rad2deg(orientation['yaw']),
             label='Yaw', alpha=0.7, linewidth=1)
    ax7.set_ylabel('Angle (degrees)', fontsize=10)
    ax7.set_title('Euler Angles', fontweight='bold')
    ax7.legend(loc='upper right', fontsize=8)
    ax7.grid(True, alpha=0.3)

    # 8. Quaternions
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.plot(orientation['seconds_elapsed'], orientation['qx'], label='qx', alpha=0.7, linewidth=1)
    ax8.plot(orientation['seconds_elapsed'], orientation['qy'], label='qy', alpha=0.7, linewidth=1)
    ax8.plot(orientation['seconds_elapsed'], orientation['qz'], label='qz', alpha=0.7, linewidth=1)
    ax8.plot(orientation['seconds_elapsed'], orientation['qw'], label='qw', alpha=0.7, linewidth=1)
    ax8.set_ylabel('Quaternion component', fontsize=10)
    ax8.set_title('Quaternion Orientation', fontweight='bold')
    ax8.legend(loc='upper right', fontsize=8, ncol=2)
    ax8.grid(True, alpha=0.3)

    # 9. Spectrogram of acceleration magnitude (wide plot)
    ax9 = fig.add_subplot(gs[4, :])
    from scipy import signal

    # Compute sampling frequency
    dt = np.median(np.diff(acc['seconds_elapsed']))
    fs = 1 / dt

    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(acc_mag, fs=fs, nperseg=min(256, len(acc_mag) // 4))

    im = ax9.pcolormesh(t + acc['seconds_elapsed'].iloc[0], f, 10 * np.log10(Sxx),
                        shading='gouraud', cmap='viridis')
    ax9.set_ylabel('Frequency (Hz)', fontsize=10)
    ax9.set_xlabel('Time (seconds)', fontsize=10)
    ax9.set_title('Acceleration Magnitude Spectrogram', fontweight='bold')
    ax9.set_ylim([0, min(10, fs / 2)])  # Show up to 10 Hz
    cbar = plt.colorbar(im, ax=ax9)
    cbar.set_label('Power (dB)', fontsize=9)

    # Add overall title
    time_info = f" ({time_range[0]:.1f}s - {time_range[1]:.1f}s)" if time_range else ""
    fig.suptitle(f'Sensor Signal Analysis{time_info}',
                 fontsize=14, fontweight='bold', y=0.995)

    return fig


def mergeprocess_dfs(acc, acc_unc, gyro, path_files):
    from functools import reduce
    # Prepare all dataframes with renamed columns
    dfs_to_merge = [
        acc[['time', 'accX', 'accY', 'accZ', 'timestamp']],
        acc_unc[['time', 'accGX', 'accGY', 'accGZ']],
        gyro[['time', 'rotA', 'rotB', 'rotG']],
    ]

    # Merge all at once
    df_merged = reduce(lambda left, right: left.merge(right, on='time', how='outer'), dfs_to_merge)
    # Sort by time
    df_merged = df_merged.sort_values('time').reset_index(drop=True)
    df_resampled = resample_to_60hz(df_merged)
    
    # Save TUG motion file into the folder
    df_resampled.to_csv(os.path.join(path_files, 'tug_motion.csv'), index=False)
    pass

def resample_to_60hz(df_merged):
    """
    Resample merged sensor data to 60 Hz using time-based interpolation.

    Parameters:
    -----------
    df_merged : DataFrame
        Merged dataframe with 'time' column (in nanoseconds)

    Returns:
    --------
    DataFrame resampled at 60 Hz
    """
    # Convert 'time' from nanoseconds to datetime
    df_merged['datetime'] = pd.to_datetime(df_merged['time'], unit='ns')

    # Set datetime as index
    df_resampled = df_merged.set_index('datetime')

    # Resample to 60 Hz (1/60 seconds = 16.667 ms)
    df_resampled = df_resampled.resample('16.667ms').mean(numeric_only=True)

    # Drop rows with missing values
    df_resampled = df_resampled.dropna()

    # Add msFromStart column. start corresponds to the first timestamp in the original df_merged (msFromStart = 0)
    start_time = df_resampled.index[0]
    df_resampled['msFromStart'] = (df_resampled.index - start_time).total_seconds() * 1000

    # Drop time and reset index to have a clean DataFrame
    df_resampled = df_resampled.reset_index(drop=True)
    df_resampled = df_resampled.drop(columns=['time'])

    # Bring msFromStart to the front
    cols = df_resampled.columns.tolist()
    cols = ['msFromStart'] + [col for col in cols if col != 'msFromStart']
    df_resampled = df_resampled[cols]
    
    return df_resampled


def mergeprocess_df_orientation(orientation, path_files):
    # keep only time, orA, orB, OrG
    df_orientation = orientation[['time', 'orA', 'orB', 'orG']]
    df_orientation = resample_to_60hz(df_orientation)
    # Save
    df_orientation.to_csv(os.path.join(path_files, 'tug_orientation.csv'), index=False)
    pass


def create_new_csv_pisa(path_pisa):
    for elemento in os.listdir(path_pisa):
        if 'aathe' in elemento:
            print("Processing:", elemento)
            elemento = elemento + '/PreIntervention/TUG_raw'

            # Load accelerometer, accelerometeruncalibrated, gyroscope, orientation
            path_files = os.path.join(path_pisa, elemento)

            # Load accelerometer
            acc = pd.read_csv(os.path.join(path_files, 'Accelerometer.csv'))
            acc_unc = pd.read_csv(os.path.join(path_files, 'AccelerometerUncalibrated.csv'))
            acc_total = pd.read_csv(os.path.join(path_files, 'TotalAcceleration.csv'))
            gyro = pd.read_csv(os.path.join(path_files, 'Gyroscope.csv'))
            orientation = pd.read_csv(os.path.join(path_files, 'Orientation.csv'))

            acc['timestamp'] = pd.to_datetime(acc['time'])
            acc_unc['timestamp'] = pd.to_datetime(acc_unc['time'])
            acc_total['timestamp'] = pd.to_datetime(acc_total['time'])

            if False:
                plot_sensor_signals(acc, acc_unc, acc_total, gyro, orientation)

            # Change column names to match msFromStart	accX	accY	accZ	accGX	accGY	accGZ	rotA	rotB	rotG
            acc = acc.rename(columns={'x': 'accX',
                                      'y': 'accY',
                                      'z': 'accZ'})

            acc_unc = acc_unc.rename(columns={'x': 'accGX',
                                              'y': 'accGY',
                                              'z': 'accGZ'})

            gyro = gyro.rename(columns={'x': 'rotA',
                                      'y': 'rotB',
                                      'z': 'rotG'})

            orientation = orientation.rename(columns={'roll': 'orG',
                                              'pitch': 'orB',
                                              'yaw': 'orA'})

            mergeprocess_dfs(acc, acc_unc, gyro, path_files)
            mergeprocess_df_orientation(orientation, path_files)
    pass


if __name__ == "__main__":

    create_new_csv_pisa(path_pisa = r'C:\Users\ao4518\Desktop\PHD\TUG-test\data_synpisa')

    print(1)