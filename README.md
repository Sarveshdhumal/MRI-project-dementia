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

I inspected all MRI scans from both the Included and Excluded folders by loading the NIfTI volumes and visually examining multiple slices from each scan. After comparing the two groups, I found that the scans in the Excluded folder appear to have lower usability for analysis, and their removal seems justified rather than accidental.

## Data_f_0001.nii
This scan shows weaker contrast between different brain tissues. The borders between gray matter, white matter, and fluid are not as clear as in the included scans. Because segmentation depends on clear tissue boundaries, this scan could cause problems during analysis or model training.

## Data_f_0002.nii
This image looks different in overall brightness and intensity compared to the included scans. Such differences can make normalization harder and may confuse a machine learning model, since it expects images with similar intensity ranges.

## Data_f_0003.nii
This scan appears slightly noisier and less sharp than the included images. Some anatomical structures are not as clearly visible, which could affect masking or tissue segmentation. Using noisy images can reduce the quality of the dataset and may negatively impact model performance.

Conclusion

Based on this inspection, excluding these scans seems justified.


Task 2 – Why NIfTI instead of DICOM

Rather than DICOM, the dataset is supplied in NIfTI format. In actuality, this greatly simplifies preprocessing.

Each slice is usually stored as a separate file in DICOM, which also contains a lot of unnecessary hospital or scanner metadata for machine learning. It can be inconvenient to manage hundreds of small files for each subject. In contrast, NIfTI stores the spatial information and the entire 3D volume in a single file. This format works directly with Python libraries like nibabel.

This makes loading, processing, and organizing the MRI data easier when NIfTI is used.

Task 3 – Creation of Cleaned Metadata CSV

I combined the clinical and demographic CSV files into a single dataset using the MRI_ID as the common identifier in order to get the metadata ready for modeling. After merging, I filtered the dataset to keep only subjects whose MRI scans were in the Included folder and eliminated duplicate columns. This guarantees that each metadata entry matches an MRI volume that is available.

Only relevant columns such as demographic variables (sex, age, education), cognitive score (MMSE), diagnosis, language, and scanner acquisition parameters that were helpful for classifying dementia were retained. To keep the dataset small and machine learning-ready, unnecessary identifiers and irrelevant technical fields were eliminated.

The following is an interpretation of a number of the dataset's abbreviations:

MRI_ID: Unique identifier of the MRI scan

MMSE: Mini-Mental State Examination score, a measure of cognitive function

years_education: Number of years of formal education

diagnosis: Clinical diagnosis (AD = Alzheimer’s Disease, FTD = Frontotemporal Dementia, CN = Cognitively Normal, MCI = Mild Cognitive Impairment)

MRIAcquisitionType: Type of MRI acquisition (e.g., 3D structural scan)

MagneticFieldStrength: MRI scanner strength in Tesla (e.g., 1.5T or 3T)

Language: Subject language code

Handling Missing Data

In order to prevent sample loss, missing values were handled carefully. The median value of each column, which is resistant to outliers, was used to fill numerical variables. When categorical variables were absent, they were labeled "Unknown." This reduced bias while maintaining the dataset's accuracy.

Only MRI subjects are included in the final, cleaned metadata file, which is prepared for additional preprocessing.

Task 4 – Normalization and Encoding

To make the metadata suitable for machine learning, I first processed the numerical and categorical features separately.

For the numerical data, I normalized the Age column using Min-Max scaling so that all values fall between 0 and 1. This helps keep the feature on a consistent scale and prevents larger numbers from having too much influence during model training. It also helps the learning process run more smoothly, especially for neural networks.

For the categorical variables such as diagnosis, language, and scanner-related settings, I converted them into numerical form using one-hot encoding. With this approach, each category is represented by its own binary column. This allows the model to use the information properly without assuming any ranking or order between categories.

After applying these steps, the final dataset contained only numerical values and could be directly used as input for machine learning or deep learning models.

Task 5 – Slice Extraction, Intensity Normalization, and Brain Mask

To understand the structure of the MRI volume, I selected one representative slice from each of the three anatomical orientations: axial, sagittal, and coronal. The axial slice shows a horizontal cross-section of the brain, the sagittal slice shows the side profile, and the coronal slice shows a front-to-back view. Viewing the brain from these three directions helps verify that the volume is correctly oriented and that the anatomical structures look consistent.

Before visualization, I applied Min–Max intensity normalization to the selected slice. This method rescales all voxel intensities so that the minimum value becomes 0 and the maximum becomes 1.​

MRI images often come from different scanners or acquisition settings, which means their raw intensity ranges can vary a lot. By normalizing intensities to a fixed range, the brightness and contrast become consistent across scans. This makes visual inspection easier and also improves stability when the images are later used in machine learning models.

I visualized both the raw axial slice and the normalized axial slice, as well as the sagittal and coronal slices. All figures include clear titles and colorbars so that the intensity values can be interpreted correctly.

To roughly separate the brain from the background, I created a simple binary brain mask using thresholding. After normalizing the full volume, voxels with intensity greater than a small threshold (for example 0.2) were labeled as brain tissue, and the rest were set to zero. This produces a basic mask that removes air and background while keeping the brain region. Although this approach is simple, it works well for initial preprocessing and visualization.

Task 6 — Saving the 3D binary mask as a NIfTI file

After generating the binary mask, I saved the full 3D mask volume as a NIfTI file. When saving the mask, I ensured that it kept the same spatial properties as the original MRI scan.

Task 7 — Median filtering and histogram comparison

I used a median filter on one axial slice from an included MRI scan to look into noise reduction. Small intensity spikes or salt-and-pepper-like noise are frequently present in medical images, which can interfere with feature extraction or segmentation.

The median filter substitutes the median value of the neighborhood (for instance, a 3x3 window) for each pixel value. The median filter eliminates isolated noisy pixels while maintaining significant anatomical boundaries, in contrast to averaging filters that may blur edges.

After applying the filter, I compared:

1. Raw slice vs filtered slice visually: The filtered image appears slightly smoother, with reduced speckle noise, while the brain structures and edges remain clearly visible. This shows that the filter successfully reduced noise without destroying important details.
2. Histograms before and after filtering: In comparison to the raw slice, the filtered image's histogram displays a slightly smoother distribution with fewer extreme intensity spikes. This validates the noise-reduction effect by showing that outlier values were suppressed.

Overall, the median filter improved image quality by removing small intensity fluctuations while preserving structural information, making the image more suitable for downstream processing or segmentation.


## Author

Sarvesh Dhumal
MSc Computer Science
University of Rostock
