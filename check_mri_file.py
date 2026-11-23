from pathlib import Path
import nibabel as nib

file_path = Path(__file__).parent / "sample" / "BRATS_001.nii.gz"

print("File size:", file_path.stat().st_size)

nii = nib.load(str(file_path))

print("MRI loaded successfully!")
print("MRI shape:", nii.shape)