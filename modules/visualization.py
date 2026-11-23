
import numpy as np
import matplotlib.pyplot as plt


CLASS_COLORS = {
    1: "#E5484D",
    2: "#E5A000",
    3: "#2685E8"
}


def create_overlay(mri_volume, prediction, view="axial", slice_index=None):

    if view == "axial":
        image = mri_volume[:, :, mri_volume.shape[2] // 2 if slice_index is None else slice_index, 0]
        mask = prediction[:, :, mri_volume.shape[2] // 2 if slice_index is None else slice_index]

    elif view == "coronal":
        index = mri_volume.shape[1] // 2 if slice_index is None else slice_index
        image = mri_volume[:, index, :, 0]
        mask = prediction[:, index, :]

    elif view == "sagittal":
        index = mri_volume.shape[0] // 2 if slice_index is None else slice_index
        image = mri_volume[index, :, :, 0]
        mask = prediction[index, :, :]

    else:
        raise ValueError("Invalid view")

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(image.T, cmap="gray", origin="lower")

    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)

    for class_id, color in CLASS_COLORS.items():
        hex_color = color.lstrip("#")
        rgb = tuple(
            int(hex_color[i:i+2], 16) / 255
            for i in (0, 2, 4)
        )

        selected = mask == class_id
        overlay[selected] = (*rgb, 0.55)

    ax.imshow(
        np.transpose(overlay, (1, 0, 2)),
        origin="lower"
    )

    ax.set_title(f"{view.capitalize()} View")
    ax.axis("off")

    return fig
