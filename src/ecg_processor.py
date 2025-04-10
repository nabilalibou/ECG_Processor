"""
ECG Heart Rate Calculator
Implements multiple methods for heart rate detection from ECG signals.
"""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Tuple, Optional, Any
import numpy as np
from scipy import signal
import pyedflib


class HRMethod(Enum):
    """Enumeration of heart rate calculation methods."""
    PEAKS = "peaks"
    FFT = "fft"


@dataclass
class ECGConfig:
    """Configuration parameters for ECG processing."""
    # Bandpass filter parameters
    LOW_CUTOFF: float = 0.05  # Hz
    HIGH_CUTOFF: float = 40.0  # Hz
    FILTER_ORDER: int = 4

    # Peak detection parameters
    MAX_HR: float = 150  # BPM
    PEAK_HEIGHT_FACTOR: float = 0.5

    # FFT parameters
    MIN_FREQ: float = 0.5  # Corresponds to 30 BPM (0.5 Hz × 60)
    MAX_FREQ: float = 2.5  # Corresponds to 150 BPM (2.5 Hz × 60)


class ECGProcessorError(Exception):
    """Custom exception for ECGProcessor errors."""
    pass


class ECGDataNotLoadedError(ECGProcessorError):
    """Exception raised when trying to process data before loading."""
    pass


class ECGProcessor:
    """Processes ECG signals to calculate heart rate using multiple methods."""

    def __init__(self, config: Optional[ECGConfig] = None):
        """
        Initialize ECG processor.

        Args:
            config: Optional configuration parameters
        """
        self.config = config or ECGConfig()
        self._signal: Optional[np.ndarray] = None
        self._sampling_rate: Optional[float] = None
        self._filtered_signal: Optional[np.ndarray] = None

    def load_synthetic_signal(self, signal_data: np.ndarray, sampling_rate: float) -> None:
        """Loads a synthetic ECG signal directly from a NumPy array."""
        if not isinstance(signal_data, np.ndarray) or signal_data.ndim != 1:
            raise ValueError("signal_data must be a 1D NumPy array.")
        if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
            raise ValueError("sampling_rate must be a positive number.")

        self._signal = signal_data.copy()
        self._sampling_rate = float(sampling_rate)
        self._filtered_signal = None

    def load_edf_file(self, file_path: Path) -> None:
        """
        Reads ECG signal/sampling rate from an EDF file.
        """
        if not isinstance(file_path, Path):
            raise TypeError(f"Invalid file_path type: {type(file_path)}. Expected Path.")
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with pyedflib.EdfReader(str(file_path)) as edf:
                signal_data = edf.readSignal(0)
                sampling_rate = edf.getSampleFrequency(0)
        except Exception as e:
            raise IOError(f"Failed to read EDF file '{file_path}': {e}")

        self._signal = signal_data
        self._sampling_rate = sampling_rate
        self._filtered_signal = None

    def _check_data_loaded(self) -> None:
        """Check if data is loaded, raising error if not."""
        if self._signal is None or self._sampling_rate is None:
            raise ECGDataNotLoadedError(
                "ECG data has not been loaded. Call load_edf_file() or load_synthetic_signal() first."
            )

    # --- Properties ---

    @property
    def sampling_rate(self) -> float:
        """Returns the sampling rate of the loaded signal."""
        self._check_data_loaded()
        return self._sampling_rate

    @property
    def original_signal(self) -> np.ndarray:
        """Returns the original loaded signal."""
        self._check_data_loaded()
        return self._signal

    @property
    def filtered_signal(self) -> np.ndarray:
        """Get filtered signal, computing it if not already available."""
        self._check_data_loaded()

        if self.config.LOW_CUTOFF >= self.config.HIGH_CUTOFF:
            raise ValueError("LOW_CUTOFF must be less than HIGH_CUTOFF")

        if self._filtered_signal is None:
            self._filtered_signal = self._apply_bandpass_filter()
        return self._filtered_signal

    # --- Core Processing Methods ---

    def compute_heart_rate(self, method: HRMethod = HRMethod.PEAKS) -> float:
        """
        Calculate heart rate using specified method.

        Args:
            method: Method to use for heart rate calculation

        Returns:
            Heart rate in BPM
        """
        methods = {
            HRMethod.PEAKS: self._compute_hr_peaks,
            HRMethod.FFT: self._compute_hr_fft
        }
        return methods[method]()

    def _apply_bandpass_filter(self) -> np.ndarray:
        """Apply Butterworth bandpass filter to remove noise and baseline wander."""
        nyq = self._sampling_rate / 2
        b, a = signal.butter(
            self.config.FILTER_ORDER,
            [self.config.LOW_CUTOFF / nyq, self.config.HIGH_CUTOFF / nyq],
            btype='band'
        )
        return signal.filtfilt(b, a, self._signal)

    def _compute_hr_peaks(self) -> float:
        """Calculate heart rate using maxima detection (R-peaks from QRS complexes)."""
        # Minimum number of samples between peaks
        min_distance = int(60 / self.config.MAX_HR * self._sampling_rate)

        # Using the absolute refractory period of 200 ms after an R peak
        # qrs_refrac_time = 0.2  # seconds
        # min_distance = round(qrs_refrac_time * self._sampling_rate)

        # baseline + 0.5*std
        threshold = (np.mean(self.filtered_signal) +
                     self.config.PEAK_HEIGHT_FACTOR * np.std(self.filtered_signal))

        peaks: np.ndarray
        peaks, _ = signal.find_peaks(
            self.filtered_signal,
            height=threshold,
            distance=min_distance
        )

        if len(peaks) < 2:
            raise ValueError("Insufficient peaks detected in signal")

        rr_intervals = np.diff(peaks) / self._sampling_rate  # RR intervals in seconds
        return 60 / np.mean(rr_intervals)  # BPM

    def _compute_hr_fft(self) -> np.ndarray[Any, np.dtype[Any]]:
        """Calculate heart rate using FFT method."""
        fft = np.fft.rfft(self.filtered_signal)
        freq = np.fft.rfftfreq(len(self.filtered_signal), 1 / self._sampling_rate)

        # Find dominant frequency in physiological HR range
        mask = (freq >= self.config.MIN_FREQ) & (freq <= self.config.MAX_FREQ)
        peaks = np.abs(fft[mask])  # Magnitude of FFT components within the frequency range
        dominant_freq = freq[mask][np.argmax(peaks)]

        return dominant_freq * 60  # BPM

    # --- Visualization ---

    def visualize_plotly(self, save_path: Optional[Path] = None) -> None:
        """
        Create interactive plots of original and filtered signals using Plotly.

        Args:
            save_path: Optional path to save the plot as HTML
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("Please install plotly: pip install plotly")

        time = np.arange(len(self._signal)) / self._sampling_rate

        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Original ECG Signal', 'Filtered ECG Signal')
        )

        # Add original signal
        fig.add_trace(
            go.Scatter(
                x=time,
                y=self._signal,
                name='Original',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )

        # Add filtered signal
        fig.add_trace(
            go.Scatter(
                x=time,
                y=self.filtered_signal,
                name='Filtered',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            showlegend=True,
            title_text="ECG Signal Analysis",
            xaxis2_title="Time (s)",
            yaxis_title="Amplitude (mV)",
            yaxis2_title="Amplitude (mV)"
        )

        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        if save_path:
            fig.write_html(save_path)

        # Show the plot in browser
        fig.show()
