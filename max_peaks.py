def detect_two_largest_peaks(data, distance=50):
    """
    Find two max peaks

    :param data: List or array of data
    :param distance: minimum distance between peaks
    :return: indexes of the two peaks
    """
    # Finding peaks
    peaks, _ = signal.find_peaks(data, height=np.max(data)*0.5, distance=distance)

    # values of the peaks
    values = data[peaks]

    if len(peaks) < 2: # if not enough peaks
        return [0,len(data)-1]

    # Find two max
    top_index = np.argsort(values)[-2:]
    top_peaks = peaks[top_index]

    return top_peaks

turn1,turn2 = detect_two_largest_peaks(moving_average(np.abs(np.gradient(df_corrected['alpha']))))

t_turn1 = df_corrected.at[turn1 + df_corrected.index[0], 'relative_timestamp']
t_turn2 = df_corrected.at[turn2 + df_corrected.index[0], 'relative_timestamp']

plt.figure(figsize=(10, 5))
plt.plot(df_corrected['relative_timestamp'],moving_average(np.abs(np.gradient(df_corrected['alpha']))))
plt.axvline(x=t_turn1, color='black')
plt.axvline(x=t_turn2, color='black')