
import numpy as np
import nibabel as nib
import torch

from monai.transforms import NormalizeIntensity


def preprocess_mri(file_path):
    """
    Load and preprocess a 4-channel NIfTI MRI volume.

    Expected input shape: (H, W, D, 4)
    Channel order: FLAIR, T1, T1ce, T2
    """

    nii = nib.load(str(file_path))
    volume = nii.get_fdata(dtype=np.float32)

    if volume.ndim != 4:
        raise ValueError(
            f"Expected a 4D MRI volume, received shape {volume.shape}"
        )

    if volume.shape[-1] != 4:
        raise ValueError(
            "MRI must contain exactly 4 modalities in this order: "
            "FLAIR, T1, T1ce, T2."
        )

    if not np.isfinite(volume).all():
        raise ValueError("MRI contains NaN or infinite values.")

    # Convert (H, W, D, 4) to (4, H, W, D).
    volume = np.moveaxis(volume, -1, 0)

    # Same normalization used during model training.
    normalizer = NormalizeIntensity(
        nonzero=True,
        channel_wise=True
    )

    volume = normalizer(volume)

    volume = np.ascontiguousarray(volume, dtype=np.float32)

    # Convert to PyTorch tensor.
    image_tensor = torch.from_numpy(volume)

    return image_tensor, nii.affine, nii.header
