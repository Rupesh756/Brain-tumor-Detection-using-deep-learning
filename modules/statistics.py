
import numpy as np


CLASS_NAMES = {
    1: "Edema",
    2: "Non-enhancing Tumour",
    3: "Enhancing Tumour"
}


def calculate_tumour_statistics(prediction, affine):
    """
    Calculate tumour volume for each predicted class.

    Returns volumes in cm³.
    """

    voxel_volume_mm3 = abs(np.linalg.det(affine[:3, :3]))

    statistics = {}

    for class_id, class_name in CLASS_NAMES.items():

        voxel_count = int(np.sum(prediction == class_id))

        volume_cm3 = (
            voxel_count * voxel_volume_mm3 / 1000
        )

        statistics[class_name] = {
            "voxels": voxel_count,
            "volume_cm3": round(float(volume_cm3), 3)
        }

    statistics["Total Tumour"] = {
        "voxels": sum(
            item["voxels"] for item in statistics.values()
        ),
        "volume_cm3": round(
            sum(
                item["volume_cm3"]
                for item in statistics.values()
            ),
            3
        )
    }

    return statistics
