# Dementia MRI Preprocessing Pipeline

## Overview

This project focuses on preprocessing structural brain MRI scans together with clinical and demographic information for dementia research. The main goal is to prepare the data so that it can later be used for deep learning models to classify dementia subtypes and to segment gray and white matter.

In this work, I cleaned the metadata, normalized MRI intensities, extracted slices for visualization, generated simple brain masks, and prepared all outputs in a reproducible way using Python.


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

All processed outputs can be downloaded here:

👉 [https://fie.nl.tab.digital/s/qYTBwpAHyi9iRxq](https://fie.nl.tab.digital/s/qYTBwpAHyi9iRxq)

This includes:

* Processed slice visualizations
* Binary masks
* Cleaned metadata CSV


## Assignment Report

Task 1 – Inspection of Excluded Scans

I first inspected all MRI scans in both the Included and Excluded folders by loading and visualizing them slice by slice.

Compared to the included scans, the excluded ones clearly showed lower image quality. Some volumes had noticeably more noise and weaker contrast between brain tissues. In a few cases the brain looked partially cropped or not fully covered, and small motion artifacts were visible. These issues make it difficult to correctly separate gray and white matter and can also break simple threshold-based masking.

If such scans are used during training, the model may learn incorrect patterns or become unstable. For this reason, I believe the exclusion was intentional and correct. In medical imaging, it is usually better to keep fewer but cleaner samples rather than many low-quality ones.

Task 2 – Why NIfTI instead of DICOM

Rather than DICOM, the dataset is supplied in NIfTI format. In actuality, this greatly simplifies preprocessing.

Each slice is usually stored as a separate file in DICOM, which also contains a lot of unnecessary hospital or scanner metadata for machine learning. It can be inconvenient to manage hundreds of small files for each subject. In contrast, NIfTI stores the spatial information and the entire 3D volume in a single file. This format works directly with Python libraries like nibabel and is commonly used in research.

This makes loading, processing, and organizing the MRI data easier when NIfTI is used.

Task 3 – Creation of Cleaned Metadata CSV

I combined the two provided metadata files, Demographics_MRI.csv and Records_MRI.csv, to create a new CSV file that would be used for machine learning. The objective was to include only those subjects for whom MRI scans were available in the Included folder and to retain only the relevant information.

Initially, pandas was used to import both CSV files into Python. Both files were combined into a single table because they include data on the same topics. To keep the dataset clean and prevent confusion during modeling, duplicate or redundant columns were eliminated after merging.

I then filtered the combined table to retain only the rows that corresponded to the MRI files that were in the Included folder. This made sure that each metadata entry matched an actual MRI volume that was used for the analysis. The final dataset automatically excluded any subjects without included scans.

Explanation of Important Abbreviations

Several abbreviations are used in the metadata:

MRI_ID – Unique identifier for each MRI scan

EEG_ID – Identifier for EEG recordings (if available)

CN – Cognitively Normal subject

AD – Alzheimer’s Disease

FTD – Frontotemporal Dementia

MCI – Mild Cognitive Impairment

MMSE – Mini-Mental State Examination score, used to measure cognitive function

VISITYR – Year of clinical visit or scan acquisition

MRIAcquisitionType – MRI scan type (e.g., 3D acquisition)

RepetitionTime – MRI repetition time parameter (scanner setting)

ImagedNucleus (1H) – Indicates proton MRI imaging (hydrogen nucleus)

MagneticFieldStrength – Scanner magnetic field strength in Tesla (e.g., 1.5T or 3T)

CSF – Cerebrospinal fluid measurement or related volumetric value

Language – Language used during assessment

These features provide demographic, clinical, and scanner-related information that may be useful as input features for predictive models.

Handling Missing Data

Numerous missing values were present in the dataset in both the numerical and categorical columns. I dealt with missing values in the following manner rather than eliminating these rows, which would have decreased the size of the dataset:

The median value of the corresponding column was used to fill in numerical columns (such as Age, MMSE, RepetitionTime, or volumetric measures). The median helps preserve realistic distributions and is resilient to extreme values.

The label "Unknown" was used to fill in categorical columns (such as diagnosis, language, or acquisition type). In addition to preserving the record, this makes it obvious that the original value was absent.

After cleaning and preprocessing, the final dataset was saved as:

```
outputs/cleaned_metadata.csv
```

This cleaned CSV contains only consistent, usable, and model-ready metadata corresponding to the included MRI scans.

Task 4 – Normalization and Encoding

Before using the metadata for modeling, I applied feature preprocessing directly in the data pipeline using pandas and scikit-learn.

First, numerical columns were cleaned and converted to proper numeric types. In particular, the Age column was transformed using Min-Max scaling with MinMaxScaler, which rescales values to the range between 0 and 1. This normalization ensures that age values are comparable with other features and prevents large magnitudes from dominating the training process. Scaling improves optimization stability and helps neural networks converge faster.

Next, all categorical columns were detected automatically using their data type (object). Instead of manually specifying each feature, I applied OneHotEncoder to encode every categorical variable at once. This includes fields such as diagnosis, sex, language, scanner parameters, and other labels. One-hot encoding converts each category into separate binary columns, allowing models to interpret the data numerically without assuming any ordering between categories.

After encoding, the categorical columns were replaced with their encoded representations and concatenated back with the numerical features. The resulting dataset contains only clean numeric values, making it directly compatible with machine learning and deep learning algorithms.

Task 5 – Slice Extraction, Intensity Normalization, and Brain Mask

Since MRI volumes are three-dimensional, I selected one representative slice from each orientation: axial, sagittal, and coronal. These slices were used for visualization and simple analysis.

Before visualization, the intensities were normalized using min-max scaling. This step standardizes brightness across scans and makes comparisons easier.

To roughly separate the brain from the background, I created a simple binary mask using intensity thresholding. Voxels with non-zero intensity were treated as brain tissue, and the rest as background. Although simple, this works reasonably well for basic masking.

All raw slices, normalized slices, and masks were saved as images in the outputs folder.


## Author

Sarvesh Dhumal
MSc Computer Science
University of Rostock
