def start_drop(data, min_duration=30,diff=100):
    data = np.array(data)

    # Find decreasing segments
    decreasing = data[:-1]>data[1:]

    start_index = 0
    end_index = len(data)
    count = 0

    for i, is_decreasing in enumerate(decreasing):
        if is_decreasing:
            count += 1
        else:
            if count >= min_duration and np.abs(data[i]-data[i-count])>diff:
                start_index = i - count + 1
                end_index = i
                break
            count = 0

    return start_index, end_index

start_turn2, end_turn2 = start_drop(moving_average(df_corrected['alpha']))
t_start_turn2 = df_corrected.at[start_turn2 + df_corrected.index[0], 'relative_timestamp']
t_end_turn2 = df_corrected.at[end_turn2 + df_corrected.index[0], 'relative_timestamp']

plt.figure(figsize=(10, 5))
plt.plot(df_corrected['relative_timestamp'],moving_average(df_corrected['alpha']))
plt.axvline(x=t_start_turn2, color='black')
plt.axvline(x=t_end_turn2, color='black')