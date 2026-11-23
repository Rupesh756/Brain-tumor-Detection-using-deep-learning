
from pathlib import Path
import numpy as np

from inference.model import load_model
from modules.preprocessing import preprocess_mri
from modules.prediction import predict_segmentation


# Update this path to your local 4D NIfTI MRI file.
MRI_PATH = Path(__file__).parent / "sample" / "BRATS_001.nii.gz"

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "best_model_v3.pth"

# 1. Load trained model.
model, device = load_model(CHECKPOINT_PATH)

# 2. Preprocess MRI.
image_tensor, affine, header = preprocess_mri(MRI_PATH)

print("\nPreprocessed MRI shape:", tuple(image_tensor.shape))

# 3. Generate segmentation.
prediction = predict_segmentation(model, image_tensor, device)

print("Prediction shape:", prediction.shape)
print("Predicted classes:", np.unique(prediction))

# 4. Calculate tumour volumes.
voxel_volume_mm3 = abs(np.linalg.det(affine[:3, :3]))

class_names = {
    1: "Edema",
    2: "Non-enhancing tumour",
    3: "Enhancing tumour"
}

print("\nTumour volume statistics:")

for class_id, class_name in class_names.items():

    voxel_count = int(np.sum(prediction == class_id))

    volume_cm3 = voxel_count * voxel_volume_mm3 / 1000

    print(
        f"{class_name}: "
        f"{voxel_count} voxels, "
        f"{volume_cm3:.3f} cm³"
    )

print("\nInference test completed successfully!")
