import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import os


def normalize(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def save_slice(arr, title, path):
    plt.figure(figsize=(5,5))
    plt.imshow(arr.T, cmap="gray", origin="lower")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def process_subject(nifti_path, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    img = nib.load(nifti_path)
    data = img.get_fdata()

    axial = data[:, :, data.shape[2]//2]
    coronal = data[:, data.shape[1]//2, :]
    sagittal = data[data.shape[0]//2, :, :]

    axial_n = normalize(axial)

    save_slice(axial, "Raw Axial", f"{out_dir}/raw_axial.png")
    save_slice(axial_n, "Normalized Axial", f"{out_dir}/norm_axial.png")

    save_slice(coronal, "Coronal", f"{out_dir}/coronal.png")
    save_slice(sagittal, "Sagittal", f"{out_dir}/sagittal.png")

    mask = (normalize(data) > 0.2).astype(np.uint8)

    mask_img = nib.Nifti1Image(mask, img.affine, img.header)
    nib.save(mask_img, f"{out_dir}/binary_mask.nii.gz")

    axial_med = median_filter(axial, size=3)

    save_slice(axial_med, "Median Filtered", f"{out_dir}/median.png")

    plt.figure()
    plt.hist(axial.flatten(), bins=100, alpha=0.5, label="Raw")
    plt.hist(axial_med.flatten(), bins=100, alpha=0.5, label="Median")
    plt.legend()
    plt.savefig(f"{out_dir}/histogram.png")
    plt.close()
