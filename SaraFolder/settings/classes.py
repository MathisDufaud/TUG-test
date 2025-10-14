import sys

import pandas as pd

import matplotlib
matplotlib.use('TkAgg')

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
plt.ion()

from SaraFolder.settings import utils_labelling
from SaraFolder.settings.utils_pisa import utils_pisatugloaders


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
        # self.raw_data = RawSensorData()
        # self.processed_data: Optional[pd.DataFrame()] = None
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None

        # # Analysis results
        self.predicted_total: Optional[float] = None
        self.predicted_phases: Optional[Dict[str, tuple[float, float]]] = None

        ## Results
        self.results: Optional[Dict] = None
    def preprocess(self, **kwargs) -> None:
        """
        Preprocess raw sensor data.

        Args:
            **kwargs: Preprocessing parameters (e.g., filter_type, cutoff_freq)
        """
        if not self.raw_data.is_complete():
            raise ValueError("Raw data must be loaded before preprocessing")

        # Implement your preprocessing pipeline
        # Example: filtering, normalization, feature extraction
        self.processed_data = self._apply_preprocessing(**kwargs)

    def _apply_preprocessing(self, **kwargs) -> np.ndarray:
        """Internal preprocessing implementation."""
        # TODO: Implement preprocessing logic
        return self.raw_data.accelerometer.values

    def plot_raw_data(self):
        """
        Plot raw sensor data.

        Args:
            sensors: List of sensors to plot (e.g., ['accelerometer', 'gyroscope'])
        """
        if self.dataset_id != 'parkapp':
            self.processed_data = utils_pisatugloaders.process_data(self.raw_data)

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

    def __repr__(self) -> str:
        return (f"TUGTest(id={self.test_id}, user={self.user_id}, "
                f"gt_total={self.gt_total_gwalk:.2f}s)")

    def plot_labelling(self, method, plot=False):
        if self.dataset_id != 'parkapp' and self.processed_data is None:
            self.processed_data = utils_pisatugloaders.process_data(self.raw_data)
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
