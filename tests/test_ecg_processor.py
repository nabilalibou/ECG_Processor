"""Unit tests for ECG processor."""
import numpy as np
import pytest
from pathlib import Path
from src.ecg_processor import ECGProcessor, HRMethod, ECGDataNotLoadedError

# Test constants
SAMPLING_RATE = 250.0  # Hz
DURATION = 10.0  # seconds
SIGNAL_FREQ = 1.0  # Hz (60 BPM)
EXPECTED_BPM = SIGNAL_FREQ * 60


@pytest.fixture
def synthetic_signal() -> np.ndarray:
    """Create a synthetic ECG signal (sine wave at 60 BPM)."""
    t = np.linspace(0, DURATION, int(SAMPLING_RATE * DURATION), endpoint=False)
    # Add some noise to make it slightly more realistic?
    # noise = np.random.normal(0, 0.1, signal.shape)
    return np.sin(2 * np.pi * SIGNAL_FREQ * t)


@pytest.fixture
def processor() -> ECGProcessor:
    """Create an ECGProcessor instance."""
    return ECGProcessor()


@pytest.fixture
def loaded_processor(processor, synthetic_signal) -> ECGProcessor:
    """Create an ECGProcessor with synthetic data loaded."""
    processor.load_synthetic_signal(synthetic_signal, SAMPLING_RATE)
    return processor


class TestECGProcessor:
    """Test core functionality of ECGProcessor."""

    def test_initialization(self, processor):
        """Test processor initializes with no data."""
        assert processor._signal is None
        assert processor._sampling_rate is None

    def test_load_synthetic_signal(self, processor, synthetic_signal):
        """Test loading synthetic data."""
        processor.load_synthetic_signal(synthetic_signal, SAMPLING_RATE)
        assert processor._sampling_rate == SAMPLING_RATE
        assert np.array_equal(processor._signal, synthetic_signal)

    def test_load_invalid_file(self, processor):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            processor.load_edf_file(Path("non_existent.edf"))

    def test_filter_no_data(self, processor):
        """Test filtering without data."""
        with pytest.raises(ECGDataNotLoadedError):
            _ = processor.filtered_signal

    def test_filter_validation(self, loaded_processor):
        """Test filter parameter validation."""
        # Test invalid cutoff frequencies
        loaded_processor.config.LOW_CUTOFF = 50.0
        loaded_processor.config.HIGH_CUTOFF = 40.0
        with pytest.raises(ValueError, match="LOW_CUTOFF must be less than HIGH_CUTOFF"):
            _ = loaded_processor.filtered_signal

    def test_filter_output(self, loaded_processor):
        """Test basic filtering properties."""
        filtered = loaded_processor.filtered_signal
        original = loaded_processor._signal

        assert filtered.shape == original.shape
        assert not np.array_equal(filtered, original)  # Should change signal

    def test_heart_rate_calculation(self, loaded_processor):
        """Test heart rate calculation methods."""
        # Test peaks method
        hr_peaks = loaded_processor.compute_heart_rate(HRMethod.PEAKS)
        assert hr_peaks == pytest.approx(EXPECTED_BPM, abs=1.0)

        # Test FFT method
        hr_fft = loaded_processor.compute_heart_rate(HRMethod.FFT)
        assert hr_fft == pytest.approx(EXPECTED_BPM, abs=1.0)
