"""
Main script to analyze ECG files using different heart rate calculation methods.
"""
from pathlib import Path
from src.ecg_processor import ECGProcessor, HRMethod


def analyze_ecg_files(data_folder: Path) -> None:
    """
    Analyze all EDF files in the given folder using both HR calculation methods.

    Args:
        data_folder: Path to folder containing ECG files
    """
    ecg_files = sorted(data_folder.glob("*.edf"))

    if not ecg_files:
        print(f"No EDF files found in {data_folder}")
        return

    for file_path in ecg_files:
        print(f"\nProcessing {file_path.name}")
        print("-" * 50)

        try:
            processor = ECGProcessor()
            processor.load_edf_file(file_path)

            # Calculate HR using peaks method
            hr_peaks = processor.compute_heart_rate(HRMethod.PEAKS)
            print(f"Peak Detection Method: {hr_peaks:.1f} BPM")

            # Calculate HR using FFT method
            hr_fft = processor.compute_heart_rate(HRMethod.FFT)
            print(f"FFT Method: {hr_fft:.1f} BPM")

            # Visualize the signal
            # processor.visualize_plotly()
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")


def main() -> None:
    # Assuming the data folder is in the same directory as the script
    data_folder = Path(__file__).parent / "data"

    if not data_folder.exists():
        print(f"Data folder not found: {data_folder}")
        return

    print("ECG Signal Analysis")
    print("=" * 50)
    analyze_ecg_files(data_folder)


if __name__ == "__main__":
    main()
