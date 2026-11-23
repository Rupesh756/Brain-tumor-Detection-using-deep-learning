
from pathlib import Path
import tempfile

import streamlit as st
import nibabel as nib
import numpy as np

from inference.model import load_model
from modules.preprocessing import preprocess_mri
from modules.prediction import predict_segmentation
from modules.statistics import calculate_tumour_statistics
from modules.visualization import create_overlay


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumour Segmentation",
    page_icon="🧠",
    layout="wide"
)

st.title("3D Brain Tumour Segmentation")
st.caption("Deep Learning Based MRI Analysis using 3D U-Net")

st.divider()


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

CHECKPOINT_PATH = (
    Path(__file__).parent
    / "checkpoints"
    / "best_model_v3.pth"
)


@st.cache_resource
def get_model():
    return load_model(CHECKPOINT_PATH)


try:
    model, device = get_model()
    st.success("3D U-Net model loaded successfully.")
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


# --------------------------------------------------
# MRI UPLOAD
# --------------------------------------------------

st.subheader("Upload MRI Scan")

uploaded_file = st.file_uploader(
    "Upload a 4-channel NIfTI MRI volume",
    type=["nii", "gz"]
)

if uploaded_file is not None:

    if not uploaded_file.name.endswith((".nii", ".nii.gz")):
        st.error("Please upload a valid .nii or .nii.gz file.")
        st.stop()

    if st.button("Run Tumour Segmentation", type="primary"):

        with tempfile.TemporaryDirectory() as temp_dir:

            temp_path = Path(temp_dir) / uploaded_file.name

            temp_path.write_bytes(uploaded_file.getvalue())

            try:

                with st.spinner("Preprocessing MRI..."):

                    image_tensor, affine, header = preprocess_mri(
                        temp_path
                    )

                st.write(
                    "Preprocessed MRI shape:",
                    tuple(image_tensor.shape)
                )

                with st.spinner(
                    "Running 3D U-Net segmentation. "
                    "CPU inference may take several minutes..."
                ):

                    prediction = predict_segmentation(
                        model,
                        image_tensor,
                        device
                    )

                # Convert channel-first MRI back to H,W,D,4.
                mri_volume = np.moveaxis(
                    image_tensor.numpy(),
                    0,
                    -1
                )

                statistics = calculate_tumour_statistics(
                    prediction,
                    affine
                )

                # Save results in session state.
                st.session_state["prediction"] = prediction
                st.session_state["mri_volume"] = mri_volume
                st.session_state["statistics"] = statistics
                st.session_state["affine"] = affine
                st.session_state["header"] = header

                st.success("Segmentation completed successfully!")

            except Exception as error:
                st.error(f"Segmentation failed: {error}")


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if "prediction" in st.session_state:

    prediction = st.session_state["prediction"]
    mri_volume = st.session_state["mri_volume"]
    statistics = st.session_state["statistics"]

    st.divider()

    st.header("Segmentation Results")

    # ----------------------------------------------
    # VOLUME STATISTICS
    # ----------------------------------------------

    st.subheader("Tumour Volume Statistics")

    columns = st.columns(4)

    names = [
        "Edema",
        "Non-enhancing Tumour",
        "Enhancing Tumour",
        "Total Tumour"
    ]

    for column, name in zip(columns, names):

        column.metric(
            name,
            f"{statistics[name]['volume_cm3']:.3f} cm³"
        )

    # ----------------------------------------------
    # VISUALIZATION
    # ----------------------------------------------

    st.divider()
    st.subheader("MRI Segmentation Visualization")

    views = ["axial", "coronal", "sagittal"]

    for view in views:

        if view == "axial":
            maximum = prediction.shape[2] - 1
        elif view == "coronal":
            maximum = prediction.shape[1] - 1
        else:
            maximum = prediction.shape[0] - 1

        index = st.slider(
            f"{view.capitalize()} Slice",
            min_value=0,
            max_value=maximum,
            value=maximum // 2,
            key=f"{view}_slice"
        )

        figure = create_overlay(
            mri_volume,
            prediction,
            view,
            index
        )

        st.pyplot(figure)
        import matplotlib.pyplot as plt
        plt.close(figure)

    # ----------------------------------------------
    # DOWNLOAD SEGMENTATION
    # ----------------------------------------------

    st.divider()
    st.subheader("Download Segmentation")

    output_nii = nib.Nifti1Image(
        prediction.astype(np.uint8),
        st.session_state["affine"],
        st.session_state["header"]
    )

    with tempfile.NamedTemporaryFile(
        suffix=".nii.gz",
        delete=False
    ) as output_file:

        output_path = output_file.name

    try:
        nib.save(output_nii, output_path)

        with open(output_path, "rb") as f:
            st.download_button(
                "Download Predicted Segmentation",
                data=f.read(),
                file_name="brain_tumour_segmentation.nii.gz",
                mime="application/gzip"
            )
    finally:
        Path(output_path).unlink(missing_ok=True)
