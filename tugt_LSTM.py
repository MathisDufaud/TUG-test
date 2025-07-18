import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import datetime as dt


#open data
base_path = r"C:\Users\mathi\Documents\stage\tugt_data\tugt_Data\all"

motion_files = {}
orientation_files = {}

for patient_folder in os.listdir(base_path):
    patient_path = os.path.join(base_path, patient_folder)
    if not os.path.isdir(patient_path):
        continue

    for session_folder in os.listdir(patient_path):
        session_path = os.path.join(patient_path, session_folder)
        if not os.path.isdir(session_path):
            continue

        motion_pattern = os.path.join(session_path, "*_motion.csv")
        orientation_pattern = os.path.join(session_path, "*_orientation.csv")

        motion_file_list = glob.glob(motion_pattern)
        orientation_file_list = glob.glob(orientation_pattern)

        if motion_file_list:
            df_motion = pd.read_csv(motion_file_list[0], index_col=0)
            df_motion['relative_timestamp'] = pd.to_timedelta(df_motion['relative_timestamp']).dt.total_seconds()
            motion_files[session_folder] = df_motion

        if orientation_file_list:
            df_orientation = pd.read_csv(orientation_file_list[0], index_col=0)
            df_orientation['relative_timestamp'] = pd.to_timedelta(df_orientation['relative_timestamp']).dt.total_seconds()
            orientation_files[session_folder] = df_orientation


def moving_average(data, window=5):
    serie = pd.Series(data)
    rolling_mean = serie.rolling(window, min_periods=1, center=True).mean()
    return rolling_mean


def setup_df(df_m,df_o):
    df_corrected = df_o.copy()

    for elem in ['alpha','beta','gamma']:
        base = df_corrected[elem].iloc[0]
        for i in range(len(df_corrected)):
            val = df_corrected[elem].iloc[i]
            if val > base + 50:
                df_corrected.loc[i:,elem] -= abs(val-base)
            elif val < base - 50:
                df_corrected.loc[i:,elem] += abs(val-base)
            base = df_corrected[elem].iloc[i]

    #interpolate
    df_final = df_m.copy()
    df_final['alpha'] = np.interp(df_m['relative_timestamp'],df_corrected['relative_timestamp'],df_corrected['alpha'])
    df_final['beta'] = np.interp(df_m['relative_timestamp'],df_corrected['relative_timestamp'],df_corrected['beta'])
    df_final['gamma'] = np.interp(df_m['relative_timestamp'],df_corrected['relative_timestamp'],df_corrected['gamma'])

    df_final['phase'] = np.zeros(len(df_final))

    df_final['alpha'] = moving_average(df_final['alpha'],20)
    for elem in ['beta','gamma','acc.x','acc.y','acc.z','rotRate.alpha','rotRate.beta','rotRate.gamma']:
        df_final[elem] = moving_average(df_final[elem])

    return df_final


#load time_steps
import csv

times = pd.read_csv(r"C:\Users\mathi\manual_times.csv", index_col=0)


df_fusion = {}

for key in list(motion_files.keys()):
    if key in list(skipped['skipped_keys']):
        del orientation_files[key]
        del motion_files[key]
    else:
        df_motion = motion_files[key]
        df_orientation = orientation_files[key]
        try:
            df_final = setup_df(df_motion, df_orientation)
            df_fusion[key] = df_final
        except Exception as e:
            print(f"Error on the setup for {key} : {e}")


#labels
for key, df in df_fusion.items():
    t0 = times['t_start'][key]
    t6 = times['t_end'][key]
    df.loc[(df['relative_timestamp'] >= t0) & (df['relative_timestamp'] <= t6), 'phase'] = 1


df = pd.concat(df_fusion, ignore_index=True)


#model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, Bidirectional, LSTM, TimeDistributed, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

features = ['acc.x','acc.y','acc.z','rotRate.alpha','rotRate.beta','rotRate.gamma','alpha','beta','gamma']
X = df[features]
y = df['phase']

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[features])


def create_sequences_classification(X, y, time_steps=60):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i+time_steps])
        y_seq.append(y[i:i+time_steps])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences_classification(scaled_features, df['phase'], time_steps=60)

print("X shape:", X_seq.shape)
print("y shape:", y_seq.shape)


X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)


model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(60, 9)))
model.add(Dropout(0.1))
model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
model.add(Dropout(0.1))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("best_model_v2.h5", save_best_only=True, monitor='val_loss', mode='min', verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test,y_test),
    batch_size=16,
    epochs=10,
    callbacks=[checkpoint]
)

loss, acc = model.evaluate(X_test, y_test)
print(f"Accuracy: {acc:.2f}")

preds = model.predict(X_test)
predicted_classes = np.round(preds)


#results
from matplotlib.patches import Patch
from collections import Counter


def majority_vote_predictions(y_pred_seq, total_frames, time_steps=60):
    """
    Computes the majority prediction per frame from the sequences.

    Args:
        y_pred_seq: (n_seq, time_steps) array of predictions by sequence
        total_frames: int, total number of original frames
        time_steps: int, sequence lenght

    Returns:
        y_pred_full: array of size (total_frames,) with frame prediction
    """
    votes = [[] for _ in range(total_frames)]

    for i, seq_pred in enumerate(y_pred_seq):
        for j in range(time_steps):
            if i + j < total_frames:
                votes[i + j].append(seq_pred[j])

    y_pred_full = np.array([
        Counter(v).most_common(1)[0][0] if v else 0
        for v in votes
    ])

    return y_pred_full


def reconstruct_X_from_sequences(X_seq, time_steps):
    """
    Reconstructs X_full from X_seq by inverse moving average.

    Args:
        X_seq: ndarray (n_seq, time_steps, n_features)
        time_steps: int, sequence lenght

    Returns:
        X_full_reconstructed: ndarray (n_frames, n_features)
    """
    n_seq, _, n_features = X_seq.shape
    total_frames = n_seq + time_steps - 1

    X_full = np.zeros((total_frames, n_features))
    count = np.zeros((total_frames, 1))

    for i in range(n_seq):
        X_full[i:i+time_steps] += X_seq[i]
        count[i:i+time_steps] += 1

    X_full /= count
    return X_full


X_full = reconstruct_X_from_sequences(X_test, time_steps=60)


def reconstruct_y_binary_fast(y_seq, time_steps):
    return y_seq[:,0]


y_true_full = reconstruct_y_binary_fast(y_test, time_steps=60)

predicted_classes = np.squeeze(predicted_classes, axis=-1)  # => (87171, 60)


y_pred_full = majority_vote_predictions(y_pred_seq=predicted_classes, total_frames=len(X_test), time_steps=60)


def plot_orientation_with_predictions(alpha, beta, gamma, y_pred, title="Orientation avec prédictions", figsize=(15, 6)):
    """
    Displays alpha, beta, gamma curves with colored background according to predictions (0 ou 1).

    Args:
        alpha, beta, gamma: ndarray (n_frames,), angles in degrees.
        y_pred: ndarray (n_frames,), predicted binary labels (0 or 1).
        title: title of the graph.
        figsize: size of the graph.
    """
    time = np.arange(len(alpha))

    plt.figure(figsize=figsize)

    # colors for classes
    for cls in [0, 1]:
        mask = (y_pred == cls)
        plt.fill_between(time, -180, 180, where=mask, alpha=0.2, label=f'Phase {cls}', color='red' if cls == 1 else 'gray')

    # alpha, beta, gamma
    plt.plot(time, alpha, label='Alpha', color='blue')
    plt.plot(time, beta, label='Beta', color='green')
    plt.plot(time, gamma, label='Gamma', color='orange')

    plt.title(title)
    plt.xlabel("Temps (frames)")
    plt.ylabel("Angle (°)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def clean_predictions_contextual(preds, min_length=10):
    """
    Removes small isolated sequences by replacing them with the dominant surrounding class.

    Args:
        preds: array-like (n_frames,), timestep predictions
        min_length: maximum length to consider a sequence as noise

    Returns:
        cleaned_preds: np.array cleaned
    """
    preds = np.array(preds)
    cleaned = preds.copy()

    current_label = preds[0]
    start_idx = 0

    for i in range(1, len(preds)):
        if preds[i] != current_label:
            end_idx = i
            segment_len = end_idx - start_idx

            if segment_len <= min_length:
                prev_label = preds[start_idx - 1] if start_idx - 1 >= 0 else None
                next_label = preds[end_idx] if end_idx < len(preds) else None

                # if each sides have same class, replace
                if prev_label == next_label and prev_label != current_label and prev_label is not None:
                    cleaned[start_idx:end_idx] = prev_label

            # update
            start_idx = i
            current_label = preds[i]

    return cleaned


#plot results
alpha = X['alpha']
beta = X['beta']
gamma = X['gamma']

alpha = alpha[:-59]
beta = beta[:-59]
gamma = gamma[:-59]

alpha = alpha[-len(y_pred_full):]
beta = beta[-len(y_pred_full):]
gamma = gamma[-len(y_pred_full):]

plot_orientation_with_predictions(alpha[-10000:], beta[-10000:], gamma[-10000:], y_true_full[-10000:])
plot_orientation_with_predictions(alpha[-10000:], beta[-10000:], gamma[-10000:], y_pred_full[-10000:])

y_pred_cleaned = y_pred_full
for i in range(10):
    y_pred_cleaned = clean_predictions_contextual(y_pred_cleaned, 30)
plot_orientation_with_predictions(alpha[-10000:], beta[-10000:], gamma[-10000:], y_pred_cleaned[-10000:])


def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

#print acc
acc_before = compute_accuracy(y_true_full, y_pred_full)
print(f"Accuracy: {acc_before:.4f}")

acc = compute_accuracy(y_true_full, y_pred_cleaned)
print(f"Accuracy: {acc:.4f}")