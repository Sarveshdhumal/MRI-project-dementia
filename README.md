# Dementia MRI Preprocessing Pipeline

## Overview

This project implements a preprocessing pipeline for brain MRI scans used in dementia-related analysis. The goal is to clean raw MRI data and associated metadata, apply basic preprocessing steps, and generate outputs that are ready for further machine learning or research experiments.

The pipeline handles both medical image volumes (NIfTI format) and CSV-based metadata and produces processed images, visualizations, and cleaned datasets for downstream use.


## Project Structure

```
MRI-project-dementia/

data/
    Demographics_MRI.csv
    Records_MRI.csv

MRI_images/
    Included/
    Excluded/

outputs/
    figures/
    cleaned_metadata.csv

src/
    mri_processing.py
    preprocess_csv.py

main.py
requirements.txt
README.md
.gitignore
```

## Functionality

### MRI Processing

* Loads `.nii` and `.nii.gz` MRI volumes
* Normalizes image intensities
* Extracts axial, coronal, and sagittal slices
* Generates visualizations
* Creates binary masks
* Saves results to the outputs folder

### Metadata Processing

* Cleans demographic and record data
* Merges CSV files
* Produces a structured dataset for analysis

## Requirements

* Python 3.x
* NumPy
* Pandas
* nibabel
* Matplotlib
* SciPy

Python Environment Setup

To ensure reproducibility and avoid dependency issues, follow these steps:

1. Create a virtual environment

```
python3 -m venv venv
```

2. Activate the virtual environment

```
source venv/bin/activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

## Running the Project

After setup, run:

```
python main.py
```

The pipeline will process the data and save all results inside the `outputs/` directory.


## Outputs

The following files are generated:

* MRI slice visualizations (PNG)
* Binary masks (`.nii.gz`)
* Cleaned metadata (`cleaned_metadata.csv`)

These outputs are intended for inspection and further modeling.


## Sample Outputs

You can download all processed outputs here:

👉 [https://fie.nl.tab.digital/s/qYTBwpAHyi9iRxq](https://fie.nl.tab.digital/s/qYTBwpAHyi9iRxq)

This includes:

* Processed slice visualizations
* Binary masks
* Cleaned metadata CSV


## Author

Sarvesh Dhumal
MSc Computer Science
University of Rostock
