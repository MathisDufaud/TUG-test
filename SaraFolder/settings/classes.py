import sys

import pandas as pd


from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path
import numpy as np


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
            gt_total: Optional[float] = None,
            gt_phases: Optional[TUGPhases] = None
    ):
        # Identifiers
        self.test_id = test_id
        self.dataset_id = dataset_id
        self.user_id = user_id
        self.session_id = session_id

        # Ground truth
        self.gt_total = gt_total
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
    @property
    def gt_total(self) -> float:
        """Total duration of the TUG test (ground truth)."""
        return self._gt_total

    @gt_total.setter
    def gt_total(self, value: float):
        if value <= 0:
            raise ValueError("Ground truth total must be positive")
        self._gt_total = value

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

    def plot_raw_data(self, sensors: Optional[List[str]] = None) -> None:
        """
        Plot raw sensor data.

        Args:
            sensors: List of sensors to plot (e.g., ['accelerometer', 'gyroscope'])
        """
        import matplotlib.pyplot as plt

        sensors = sensors or ['accelerometer', 'gyroscope']
        fig, axes = plt.subplots(len(sensors), 1, figsize=(12, 4 * len(sensors)))

        if len(sensors) == 1:
            axes = [axes]

        for ax, sensor in zip(axes, sensors):
            data = getattr(self.raw_data, sensor)
            if data.is_loaded():
                ax.plot(data.timestamps, data.values)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(f'{sensor.capitalize()} (units)')
                ax.set_title(f'{sensor.capitalize()} - Test {self.test_id}')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_phases(self) -> None:
        """Plot detected phases overlaid on sensor data."""
        import matplotlib.pyplot as plt

        if not self.raw_data.accelerometer.is_loaded():
            raise ValueError("Raw data must be loaded")

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot accelerometer magnitude
        acc_data = self.raw_data.accelerometer
        magnitude = np.linalg.norm(acc_data.values, axis=1)
        ax.plot(acc_data.timestamps, magnitude, label='Acceleration Magnitude')

        # Overlay ground truth phases if available
        if self.gt_phases:
            for phase_name, (start, end) in self.gt_phases.to_dict().items():
                ax.axvspan(start, end, alpha=0.3, label=f'GT: {phase_name}')

        # Overlay predicted phases if available
        if self.predicted_phases:
            for phase_name, (start, end) in self.predicted_phases.items():
                ax.axvline(start, color='red', linestyle='--', alpha=0.5)
                ax.axvline(end, color='red', linestyle='--', alpha=0.5)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration Magnitude')
        ax.set_title(f'TUG Test {self.test_id} - Phase Detection')
        ax.legend()
        ax.grid(True, alpha=0.3)
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
            'ground_truth_total': self.ground_truth_total,
            'predicted_total': self.predicted_total,
            'error': abs(self.predicted_total - self.ground_truth_total)
            if self.predicted_total else None,
            'wearing_position': self.wearing_position,
            'created_on': self.created_on,
            'data_loaded': self.raw_data.is_complete(),
            'data_processed': self.processed_data is not None
        }

    def __repr__(self) -> str:
        return (f"TUGTest(id={self.test_id}, user={self.user_id}, "
                f"gt_total={self.ground_truth_total:.2f}s)")
