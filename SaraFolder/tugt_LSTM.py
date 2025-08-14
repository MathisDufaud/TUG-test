import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import datetime as dt
import running_settings
from SaraFolder import utils_functions

base_path = running_settings.base_path
times = pd.read_csv(base_path + os.sep + "manual_times.csv", index_col=0)


df = utils_functions.ready_df()

features = ['acc.x','acc.y','acc.z','rotRate.alpha','rotRate.beta','rotRate.gamma','alpha','beta','gamma']
X = df[features]
y = df['phase']

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[features])

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


X_full = reconstruct_X_from_sequences(X_test, time_steps=60)


y_true_full = reconstruct_y_binary_fast(y_test, time_steps=60)

predicted_classes = np.squeeze(predicted_classes, axis=-1)  # => (87171, 60)

y_pred_full = majority_vote_predictions(y_pred_seq=predicted_classes, total_frames=len(X_test), time_steps=60)


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


#print acc
acc_before = compute_accuracy(y_true_full, y_pred_full)
print(f"Accuracy: {acc_before:.4f}")

acc = compute_accuracy(y_true_full, y_pred_cleaned)
print(f"Accuracy: {acc:.4f}")