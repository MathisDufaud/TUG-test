def detect_consecutive_peaks(data, distance=70):
    """
    Find four consecutive peaks

    :param data: List or array of data
    :param distance: minimum distance between peaks
    :return: indexes of the peaks
    """
    # Détection des pics maximaux et minimaux
    peaks_max, _ = signal.find_peaks(data,height=np.mean(data), distance=distance)

    # Chercher une séquence valide dans les pics max ou min
    if len(peaks_max) >= 4:
        return np.sort(peaks_max)[:4]

    return []

id1,id2,id3,id4 = detect_consecutive_peaks(moving_average(df_m_red['alpha_beta']))
t1 = df_m_red.at[id1 + df_m_red.index[0], 'relative_timestamp']
t2 = df_m_red.at[id2 + df_m_red.index[0], 'relative_timestamp']
t3 = df_m_red.at[id3 + df_m_red.index[0], 'relative_timestamp']
t4 = df_m_red.at[id4 + df_m_red.index[0], 'relative_timestamp']

plt.figure(figsize=(10, 5))
plt.plot(df_m_red['relative_timestamp'],moving_average(df_m_red['alpha_beta']))
plt.axvline(x=t1, color='black')
plt.axvline(x=t2, color='black')
plt.axvline(x=t3, color='black')
plt.axvline(x=t4, color='black')