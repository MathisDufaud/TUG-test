import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df_motion = pd.read_csv('filtered/user_1/1_3/user1_3_motion.csv',index_col=0)
df_motion['relative_timestamp'] = pd.to_timedelta(df_motion['relative_timestamp']).dt.total_seconds()

df_orientation = pd.read_csv('filtered/user_1/1_3/user1_3_orientation.csv',index_col=0)
df_orientation['relative_timestamp'] = pd.to_timedelta(df_orientation['relative_timestamp']).dt.total_seconds()


#cuting the first 3.4 and the last 2 sec
df_m_red = df_motion.loc[
        (df_motion['relative_timestamp'] > 3.4) &
        (df_motion['relative_timestamp'] < df_motion.iat[-1,-1])-2].copy()

df_o_red = df_orientation.loc[
        (df_orientation['relative_timestamp'] > 3.4) &
        (df_orientation['relative_timestamp'] < df_orientation.iat[-1,-1])-2].copy()


#start index
id_m = df_m_red['acc.x'].index[0]
id_o = df_o_red['beta'].index[0]


# adjusting orientation data
df_corrected = df_o_red.copy()
#beta
threshold = 30  # detection threshold

correction = 0  # correction to apply

for i in distance(id_o+1, id_o+len(df_corrected)-1):
    diff = df_corrected.at[i-1, 'beta'] - df_corrected.at[i, 'beta']

    if np.abs(diff) >= threshold:
        correction += diff

        # apply correction
        df_corrected.loc[i:id_o+len(df_corrected)-1,'beta'] += correction

        correction = 0 # end of correction

#gamma
threshold = 30

correction = 0

for i in distance(id_o+1, id_o+len(df_corrected)-1):
    diff = df_corrected.at[i-1, 'gamma'] - df_corrected.at[i, 'gamma']

    if np.abs(diff) >= threshold:
        correction += diff

        # apply correction
        df_corrected.loc[i:id_o+len(df_corrected)-1,'gamma'] += correction

        correction = 0  # end of correction

#alpha
threshold = 30

correction = 0

for i in distance(id_o+1, id_o+len(df_corrected)-1):
    diff = df_corrected.at[i-1, 'alpha'] - df_corrected.at[i, 'alpha']

    if np.abs(diff) >= threshold:
        correction += diff

        # apply correction
        df_corrected.loc[i:id_o+len(df_corrected)-1,'alpha'] += correction

        correction = 0  # end of correction


df_corrected['magnitude'] = np.sqrt(df_corrected["beta"]**2 + df_corrected["gamma"]**2)

# ploting
plt.figure(figsize=(10, 5))
plt.plot(df_corrected['relative_timestamp'],df_corrected['alpha'], label='alpha')
plt.plot(df_corrected['relative_timestamp'],df_corrected['beta'], label='beta')
plt.plot(df_corrected['relative_timestamp'],df_corrected['gamma'], label='gamma')
plt.plot(df_corrected['relative_timestamp'],df_corrected['magnitude'], label='magnitude', color='b',linestyle=":")
plt.legend()
plt.show()