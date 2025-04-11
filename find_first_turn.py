def start_rise(data, min_duration=50,diff=100):
    data = np.array(data)

    # Find inceasing segments
    increasing = data[:-1]<data[1:]

    start_index = 0
    end_index = len(data)
    count = 0

    for i, is_increasing in enumerate(increasing):
        if is_increasing:
            count += 1
        else:
            if count >= min_duration and np.abs(data[i]-data[i-count])>diff:
                start_index = i - count + 1
                end_index = i
                break
            count = 0

    return start_index, end_index

start_turn, end_turn = start_rise(moving_average(df_corrected['alpha']))
t_start_turn = df_corrected.at[start_turn + df_corrected.index[0], 'relative_timestamp']
t_end_turn = df_corrected.at[end_turn + df_corrected.index[0], 'relative_timestamp']

plt.figure(figsize=(10, 5))
plt.plot(df_corrected['relative_timestamp'],moving_average(df_corrected['alpha']))
plt.axvline(x=t_start_turn, color='black')
plt.axvline(x=t_end_turn, color='black')