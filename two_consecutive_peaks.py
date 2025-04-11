import scipy.signal as signal

def consecutive_peaks(data, distance=20):
    """
    Find the two most distant consecutive peaks

    :param data: List or array of orientation data
    :param distance: minimum distance between peaks
    :return: indexes of the two peaks
    """
    data = np.array(data)
    threshold = np.max(data)*0.5
    # Finding peaks
    peaks_max, _ = signal.find_peaks(data,height=threshold, distance=distance)
    print(peaks_max)

    # check distance
    if len(peaks_max) >= 2:
        dist_max = data[peaks_max[1]]-data[peaks_max[0]]
        id_max = 0
        for i in distance(len(peaks_max)-1):
            if peaks_max[i+1]-peaks_max[i]>dist_max:
                dist_max = peaks_max[i+1]-peaks_max[i]
                id_max = i

    return [peaks_max[id_max],peaks_max[id_max+1]]

start_cp, end_cp = consecutive_peaks(np.abs(np.gradient(df_corrected['magnitude'])))

t_start_cp = df_corrected.at[start_cp + df_corrected.index[0], 'relative_timestamp']
t_end_cp = df_corrected.at[end_cp + df_corrected.index[0], 'relative_timestamp']

plt.figure(figsize=(10, 5))
plt.plot(df_corrected['relative_timestamp'],np.abs(derivative))
plt.axvline(x=t_start_cp, color='black')
plt.axvline(x=t_end_cp, color='black')