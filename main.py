import os
from src.preprocess_csv import create_clean_csv
from src.mri_processing import process_subject


included_dir = "MRI_images/Included"
demo = "data/Demographics_MRI.csv"
rec = "data/Records_MRI.csv"


nii_files = [
    f for f in os.listdir(included_dir)
    if f.endswith((".nii", ".nii.gz"))
]

if len(nii_files) == 0:
    raise ValueError("No NIfTI files found inside MRI_images/Included")


included_ids = [f.split(".")[0] for f in nii_files]


create_clean_csv(
    demo,
    rec,
    included_ids,
    "outputs/cleaned_metadata.csv"
)


first_scan = os.path.join(included_dir, nii_files[0])

process_subject(first_scan, "outputs/figures")

print("✅ Pipeline finished successfully!")
