from setuptools import setup, find_packages

setup(
    name="ecg-processor",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.9.0",
        "pyedflib>=0.1.28",
        "plotly>=5.13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ],
    },
    author="nabilalibou",
    description="ECG signal processing and heart rate calculation",
    keywords="ecg, signal processing, heart rate",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)