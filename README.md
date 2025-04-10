# ECG Processor

A Python class for processing ECG signals and computing heart rates using Fourier transform and 
peak detection methods (Maxima detection).

## Features

- ECG signal processing:
  - Butterworth bandpass filter (0.05-40 Hz)
    - Low cutoff of 0.05Hz to remove baseline wander caused by patient breathing or movement artifacts.
    - High cutoff of 40 Hz to remove high-frequency noise while preserving the main frequency content of QRS complexes
      (10-25 Hz). The noise is generally caused by power line interference (50/60 Hz) and EMG artifacts.
- Heart rate calculation methods:
  - Time-domain R-peak detection with adaptive thresholding
  - Fourier transform frequency analysis (0.5-2.5 Hz range)
- EDF file support

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place your EDF files in the data folder
2. Run the analysis:
```bash
python main.py
```

## Testing
Run the test suite:
```bash
pytest tests/ -v
```

## Observations about the data

2nd signal '0002.edf': Poor R wave prominence compared to S negative deflection.

3rd signal '0003.edf': Fragmented QRS complexes with some notching on the upstroke of the R wave.

## Future Improvements

- Advanced baseline correction with median filter or polynomial fitting.
- Handle fragmented QRS with smoothing Savitzky-Golay filter or QRS template matching (cross-correlation).
- Implement adaptive peak detection that switches between R and S waves based on prominence metrics.
- Possibility of using prominence-based R wave detection instead of height.
- Signal segmentation for long recordings.
- Signal quality assessment (SNR-based metrics)
- Logging system


