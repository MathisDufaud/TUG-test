import sys

import pandas as pd
import os

from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import BatchNormalization, Activation, Concatenate, Input, Add
from keras.src.optimizers.adam import Adam
from keras import metrics, Model
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Conv1D, Bidirectional, LSTM, TimeDistributed, Dense, Dropout


from SaraFolder.settings import utils_plots, running_settings

import matplotlib
matplotlib.use('TkAgg')

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List
import numpy as np
from matplotlib import pyplot as plt
plt.ion()

from SaraFolder.settings import utils_labelling, utils_dataquality


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        self.logging = True

    def write(self, message):
        self.terminal.write(message)
        if self.logging:
            self.log.write(message)

    def flush(self):
        self.terminal.flush()
        if self.logging:
            self.log.flush()

    def stop_logging(self):
        self.logging = False
        self.log.close()

class Results:
    def __init__(self, **entries):
        self.__dict__.update(entries)

@dataclass
class SmartphoneInfo:
    """Stores smartphone metadata."""
    model: Optional[str] = None
    os_version: Optional[str] = None
    app_version: Optional[str] = None
    sampling_rate: Optional[float] = None

# @dataclass
# class SensorData:
#     """Stores time-series sensor data."""
#     timestamps: Optional[np.ndarray] = None
#     values: Optional[np.ndarray] = None
#
#     def is_loaded(self) -> bool:
#         return self.timestamps is not None and self.values is not None
#
#     @property
#     def duration(self) -> Optional[float]:
#         if self.is_loaded():
#             return self.timestamps[-1] - self.timestamps[0]
#         return None
#
# @dataclass
# class RawSensorData:
#     """Container for raw sensor measurements."""
#     accelerometer: SensorData = field(default_factory=SensorData)
#     gyroscope: SensorData = field(default_factory=SensorData)
#     orientation: SensorData = field(default_factory=SensorData)
#
#     def is_complete(self) -> bool:
#         return (self.accelerometer.is_loaded() and
#                 self.gyroscope.is_loaded())

@dataclass
class TUGPhases:
    """Ground truth phase timings for TUG test."""
    t_start: Optional[float] = None
    t_end_stand: Optional[float] = None
    t_start_turn: Optional[float] = None
    t_end_turn: Optional[float] = None
    t_start_turn2: Optional[float] = None
    t_start_sit: Optional[float] = None
    t_end: Optional[float] = None

    def to_dict(self) -> Dict[str, tuple[float, float]]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class TUGTest:
    """
    Represents a Timed Up and Go (TUG) test session.

    A TUG test measures mobility and includes phases: sitting, standing,
    walking, turning, and sitting back down.
    """

    def __init__(
            self,
            test_id: int,
            dataset_id: str,
            user_id: int,
            session_id: int,
            gt_total_gwalk: Optional[float] = None,
            gt_total_manual: Optional[float] = None,
            gt_phases: Optional[TUGPhases] = None
    ):
        # Identifiers
        self.test_id = test_id
        self.dataset_id = dataset_id
        self.user_id = user_id
        self.session_id = session_id

        # Ground truth
        self.gt_total_manual = gt_total_manual
        self.gt_total_gwalk = gt_total_gwalk
        self.gt_phases = gt_phases or TUGPhases()

        # Metadata
        self.created_on: Optional[datetime] = None
        self.wearing_position: Optional[str] = None
        self.smartphone_info = SmartphoneInfo()
        self.context: Optional[str] = None

        # Sensor data

        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None

        # # Analysis results
        self.predicted_total: Optional[float] = None
        self.predicted_phases: Optional[Dict[str, tuple[float, float]]] = None

        ## Results
        self.results: Optional[Dict] = {}

    def plot_raw_data(self):
        """
        Plot raw sensor data.

        Args:
            sensors: List of sensors to plot (e.g., ['accelerometer', 'gyroscope'])
        """
        df_plot = self.processed_data
        if self.gt_phases is None:
            print("No phases detected, cannot plot.")
        else:
            t_start = self.gt_phases.t_start
            t_end_stand = self.gt_phases.t_end_stand
            t_start_turn = self.gt_phases.t_start_turn
            t_end_turn = self.gt_phases.t_end_turn
            t_start_turn2 = self.gt_phases.t_start_turn2
            t_start_sit = self.gt_phases.t_start_sit
            t_end = self.gt_phases.t_end

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(df_plot["relative_timestamp"], df_plot["sqrt(X²+Y²+Z²)"],
                 label="Motion (m/s²)", color="blue", linestyle="-")

        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Acceleration (m/s²)", color="blue")
        ax1.tick_params(axis='y', labelcolor="blue")
        ax3 = ax1.twinx()
        ax3.plot(df_plot["relative_timestamp"], df_plot["alpha"], label="Alpha (°)", color="red", linestyle="--", linewidth=3)
        ax3.plot(df_plot["relative_timestamp"], df_plot["beta"], label="Beta (°)", color="green", linestyle="-.", linewidth=3)
        ax3.plot(df_plot["relative_timestamp"], df_plot["gamma"], label="Gamma (°)", color="purple", linestyle=":", linewidth=3)

        ax3.plot(df_plot["relative_timestamp"], df_plot["rotRate.alpha"], label="RotRate Alpha (°)", color='darkred', linestyle="--")
        ax3.plot(df_plot["relative_timestamp"], df_plot["rotRate.beta"], label="RotRate Beta (°)", color='darkgreen', linestyle="-.")
        ax3.plot(df_plot["relative_timestamp"], df_plot["rotRate.gamma"], label="RotRate Gamma (°)", color='darkred', linestyle=":")

        if self.dataset_id == 'parkapp':
            ax1.axvspan(t_start, t_end, color="orange", alpha=0.2, label="Total duration")
            ax1.axvspan(t_start_turn, t_end_turn, color="limegreen", alpha=0.5, label="First turn")
            ax1.axvspan(t_start_turn2, t_start_sit, color="darkgreen", alpha=0.5, label="Second turn")

        ax1.grid()
        ax1.legend(loc="upper left")
        ax3.legend(loc="lower right")

        if self.gt_total_manual is not None:
            gtm = self.gt_total_manual
            if gtm>5000:
                gtm=gtm/1000
        else:
            gtm=0

        gtg = self.gt_total_gwalk
        if gtg>5000:
            gtg=gtg/1000

        plt.title(f"TUG raw data, "
                  f"{self.user_id}_{self.session_id}, "
                  f"GTmanual = {np.round(gtm, 2)},"
                  f"GTgwalk = {np.round(gtg,2)}")

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def get_summary(self) -> Dict:
        """
        Get a summary of the test results.

        Returns:
            Dictionary with test metadata and results
        """
        return {
            'test_id': self.test_id,
            'user_id': self.user_id,
            'dataset_id': self.dataset_id,
            'ground_truth_total': self.gt_total_gwalk,
            'predicted_total': self.predicted_total,
            'error': abs(self.predicted_total - self.gt_total_gwalk)
            if self.predicted_total else None,
            'wearing_position': self.wearing_position,
            'created_on': self.created_on,
            'data_loaded': self.raw_data.is_complete(),
            'data_processed': self.processed_data is not None
        }

    def plot_labelling(self, method, plot=False, optimization=False):
        if not optimization:
            if not isinstance(self.processed_data, str):
                utils_labelling.compute_method(self, method)

                if not isinstance(self.results[method], str):
                    results = Results(**self.results[method])

                    if plot:
                        df_plot = self.processed_data
                        if self.gt_phases is None:
                            print("No phases detected, cannot plot.")
                        else:
                            t_start = self.gt_phases.t_start
                            t_end_stand = self.gt_phases.t_end_stand
                            t_start_turn = self.gt_phases.t_start_turn
                            t_end_turn = self.gt_phases.t_end_turn
                            t_start_turn2 = self.gt_phases.t_start_turn2
                            t_start_sit = self.gt_phases.t_start_sit
                            t_end = self.gt_phases.t_end

                        fig, ax1 = plt.subplots(figsize=(10, 5))
                        ax1.plot(df_plot["relative_timestamp"], df_plot["sqrt(X²+Y²+Z²)"],
                                 label="Motion (m/s²)", color="blue", linestyle="-")

                        ax1.set_xlabel("Time (s)")
                        ax1.set_ylabel("Acceleration (m/s²)", color="blue")
                        ax1.tick_params(axis='y', labelcolor="blue")

                        if self.dataset_id == 'parkapp':
                            ax1.axvspan(t_start, t_end, color="orange", alpha=0.3, label="Total duration")
                            ax1.axvspan(results.t_start, results.t_end, color="red", alpha=0.2, label="Total duration - estimation")

                            ax1.axvspan(t_start_turn, t_end_turn, color="darkgreen", alpha=0.6, label="First turn")
                            ax1.axvspan(t_start_turn2, t_start_sit, color="darkgreen", alpha=0.6, label="Second turn")

                            ax1.axvspan(results.t_start_turn, results.t_end_turn, color="blue", alpha=0.4, label="First turn - estimation")
                            ax1.axvspan(results.t_start_turn2, results.t_end_turn2, color="blue", alpha=0.4, label="Second turn - estimation")

                            ax3 = ax1.twinx()
                            ax3.plot(df_plot["relative_timestamp"], df_plot["alpha"], label="Alpha (°)", color="red", linestyle="--")
                            ax3.plot(df_plot["relative_timestamp"], df_plot["beta"], label="Beta (°)", color="green", linestyle="-.")
                            ax3.plot(df_plot["relative_timestamp"], df_plot["gamma"], label="Gamma (°)", color="purple", linestyle=":")

                        if self.dataset_id == 'synergy' or self.dataset_id == 'pisa':
                            ax1.axvspan(results.t_start, results.t_end, color="red", alpha=0.2, label="Total duration - estimation")
                            ax1.axvspan(results.t_start_turn, results.t_end_turn, color="blue", alpha=0.4,
                                        label="First turn - estimation")
                            ax1.axvspan(results.t_start_turn2, results.t_end_turn2, color="blue", alpha=0.4,
                                        label="Second turn - estimation")

                            ax3 = ax1.twinx()
                            ax3.plot(df_plot["relative_timestamp"], df_plot["alpha"], label="Alpha (°)", color="red",
                                     linestyle="--")
                            ax3.plot(df_plot["relative_timestamp"], df_plot["beta"], label="Beta (°)", color="green",
                                     linestyle="-.")
                            ax3.plot(df_plot["relative_timestamp"], df_plot["gamma"], label="Gamma (°)", color="purple",
                                     linestyle=":")
                            ax3.legend(loc="lower right")

                        plt.title(f"TUG estimation, {method} approach, "
                                  f"{self.user_id}_{self.session_id}, "
                                  f"{self.dataset_id}, "
                                  f"GTmanual - Est = {np.round(self.gt_total_manual/1000 - (results.t_end-results.t_start), 2)}")

                        plt.xticks(rotation=45)

                        ax1.grid()
                        ax1.legend(loc="upper left")

                        plt.show()
                        plt.tight_layout()
        else:
            if not isinstance(self.processed_data, str):
                utils_labelling.compute_method_optimization(self, method)



    def data_quality_investigation(self, plot=False):
        print(f"Investigating data quality for {self.user_id}_{self.session_id}")

        stats = utils_dataquality.compute_test_stats(self.processed_data)

        self.data_quality_stats = stats

        pass


class MlModel:
    def __init__(
            self,
            model_name: str,
    ):
        # Identifiers
        self.model_name = model_name

    def define_model(self, window_size=60, n_features=9, architecture='', save_model=False):
        """
        Define model with multiple architecture options.

        Args:
            architecture: 'cnn_bilstm', 'transformer', 'tcn', or 'simple_lstm'
        """
        print("Selecting architecture: " + architecture)
        if architecture == 'cnn_bilstm':
            model = self._build_cnn_bilstm(window_size, n_features)
        elif architecture == 'tcn':
            model = self._build_tcn_residual(window_size, n_features)
        elif architecture == 'strongbs':
            model = self._build_strong_baseline(window_size, n_features)
        else:
            model = self._build_simple_lstm(window_size, n_features)

        # Compile with appropriate loss and metrics
        model.compile(
            optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
            loss='binary_crossentropy',
            metrics=['accuracy', metrics.Precision(), metrics.Recall()]
        )

        self.defined_model = model

        # Callbacks
        if save_model:
            path_dir = running_settings.models_path + os.sep + self.model_name
            self.defined_checkpoint = ModelCheckpoint(
                path_dir,
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            )
        return model

    def model_fit(self, X_train, y_train, X_val, y_val, plot=False,
                  information=None, epochs=5, batch_size=32, save_model=False):
        """Enhanced training with early stopping and learning rate scheduling."""
        if information:
            print("Model information:", information)

        # Additional callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            min_delta=0.001,
            cooldown=2,
            verbose=1
        )

        # Handle class imbalance if needed
        class_weights = None
        pos_ratio = np.mean(y_train)
        if pos_ratio < 0.3 or pos_ratio > 0.7:
            class_weights = {
                0: 1.0 / (1 - pos_ratio),
                1: 1.0 / pos_ratio
            }
            print(f"Using class weights: {class_weights}")

        if save_model:
            callbacks = [self.defined_checkpoint, early_stopping, reduce_lr]
        else:
            callbacks = [early_stopping, reduce_lr]

        print(f"Training with X_train shaped: {X_train.shape}")
        print(f"Evaluation with X_val shaped: {X_val.shape}")

        history = self.defined_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        self.model_history = history
        self.fitted_model = self.defined_model

        if plot:
            utils_plots.plot_training_history(history, title=self.model_name.strip('.h5') + '.jpg')

        return history

    def save_model(self, title):
        self.fitted_model.save(running_settings.models_path + os.sep + title)

    def load_model(self, title):
        self.fitted_model = load_model(running_settings.models_path + os.sep + title)

    def _build_cnn_bilstm(self, window_size, n_features):
        """Improved CNN-BiLSTM with batch normalization and regularization."""
        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', padding='same',
                   input_shape=(window_size, n_features)),
            BatchNormalization(),
            Dropout(0.2),

            Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.2),

            Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.2),

            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),

            TimeDistributed(Dense(32, activation='relu')),
            TimeDistributed(Dense(1, activation='sigmoid'))
        ])
        return model

    def _build_tcn_residual(self, window_size, n_features):
        def residual_tcn_block(x, filters, kernel_size, dilation_rate, dropout=0.2):
            conv = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(x)
            conv = BatchNormalization()(conv)
            conv = Dropout(dropout)(conv)
            conv = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(conv)
            conv = BatchNormalization()(conv)
            conv = Dropout(dropout)(conv)
            # residual
            if x.shape[-1] != filters:
                res = Conv1D(filters, 1, padding='same')(x)
            else:
                res = x
            return Activation('relu')(Add()([res, conv]))

        inp = Input(shape=(window_size, n_features))
        x = Conv1D(64, 3, padding='causal', activation='relu')(inp)
        x = residual_tcn_block(x, 64, 3, dilation_rate=1)
        x = residual_tcn_block(x, 64, 3, dilation_rate=2)
        x = residual_tcn_block(x, 128, 3, dilation_rate=4)
        x = residual_tcn_block(x, 128, 3, dilation_rate=8)
        out = TimeDistributed(Dense(1, activation='sigmoid'))(x)
        model = Model(inp, out)
        return model

    def _build_simple_lstm(self, window_size, n_features):
        """Simpler LSTM baseline."""
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(window_size, n_features)),
            Dropout(0.3),
            LSTM(32, return_sequences=True),
            Dropout(0.3),
            TimeDistributed(Dense(1, activation='sigmoid'))
        ])
        return model

    def _build_strong_baseline(self, window_size, n_features):
        inp = Input(shape=(window_size, n_features))
        # multi-scale convs
        c1 = Conv1D(64, 3, padding='same', activation='relu')(inp)
        c1 = BatchNormalization()(c1)
        c2 = Conv1D(64, 5, padding='same', activation='relu')(inp)
        c2 = BatchNormalization()(c2)
        c3 = Conv1D(64, 7, padding='same', activation='relu')(inp)
        c3 = BatchNormalization()(c3)
        x = Concatenate()([c1, c2, c3])  # shape: (T, 192)
        x = Dropout(0.2)(x)

        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.3)(x)

        # simple attention
        attn = Dense(1, activation='tanh')(x)
        attn = Activation('softmax')(attn)  # softmax along time dimension when using functional API later
        # Multiply attention weights and features
        out_seq = TimeDistributed(Dense(1, activation='sigmoid'))(x)
        model = Model(inp, out_seq)
        return model