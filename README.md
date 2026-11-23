# Brain Tumor Detection and Segmentation Using Deep Learning

A deep-learning project for brain tumor analysis using MRI scans. This repository includes 2D classification experiments, 3D brain tumor segmentation using MONAI and 3D U-Net, and a Streamlit application for interactive MRI segmentation and visualization.

## Project Overview

Brain tumors can be difficult to identify and analyze from MRI scans. This project explores deep-learning techniques to classify and segment brain tumors using MRI data.

The main focus is **3D brain tumor segmentation using a 3D U-Net architecture**, trained on the Medical Segmentation Decathlon (MSD) BrainTumour dataset.

The Streamlit application allows users to upload a 4D MRI volume, generate tumor segmentation predictions, visualize the results in different anatomical views, and calculate tumor region volumes.

## Key Features

- 3D brain tumor segmentation using MONAI and 3D U-Net.
- Four MRI modalities: FLAIR, T1, T1ce, and T2.
- Segmentation of three tumor subregions:
  - Edema
  - Non-enhancing tumor
  - Enhancing tumor
- Sliding-window inference for large 3D MRI volumes.
- Interactive Streamlit interface.
- Axial, coronal, and sagittal segmentation visualization.
- Tumor region volume calculation in cubic centimeters (cm³).
- Export predicted segmentation as a NIfTI file.
- Model evaluation using Dice similarity coefficient.
- Grad-CAM experiments for model interpretability.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| Medical Imaging | MONAI, NiBabel |
| Model Architecture | 3D U-Net |
| Image Processing | NumPy, SciPy |
| Visualization | Matplotlib |
| Web Application | Streamlit |
| Other Experiments | EfficientNetB0 |

## Dataset

**Medical Segmentation Decathlon – Task01_BrainTumour**

The dataset contains multimodal 3D brain MRI scans.

Dataset organization:

```text
Task01_BrainTumour/
├── imagesTr/
├── labelsTr/
├── imagesTs/
└── dataset.json
```

Dataset details:

- 484 labeled training cases.
- 289 unlabeled testing cases.
- Four MRI modalities per case.
- Three tumor subregions and background.

### MRI Modalities

| Channel | MRI Modality |
|---|---|
| 0 | FLAIR |
| 1 | T1 |
| 2 | T1ce (Post-contrast T1) |
| 3 | T2 |

### Segmentation Labels

| Label | Region |
|---|---|
| 0 | Background |
| 1 | Edema |
| 2 | Non-enhancing tumor |
| 3 | Enhancing tumor |

Dataset source: [Medical Segmentation Decathlon](http://medicaldecathlon.com/)

## Model Architecture

The primary segmentation model is a 3D U-Net implemented using MONAI.

**Model configuration:**

- Spatial dimensions: 3D
- Input channels: 4
- Output channels: 4
- Feature channels: (16, 32, 64, 128)
- Strides: (2, 2, 2)
- Residual units: 2
- Loss function: DiceCELoss
- Optimizer: RMSprop
- Inference: Sliding Window Inference
- ROI size: 96 × 96 × 96
- Window overlap: 0.25

The model predicts a segmentation mask containing background and three tumor subregions.

## Model Performance

The following results were obtained on the 97-case validation split.

| Tumor Region | Mean Dice Score |
|---|---:|
| Edema | 0.7796 |
| Non-enhancing Tumor | 0.5745 |
| Enhancing Tumor | 0.7276 |
| Macro Average | 0.6939 |

The best validation checkpoint achieved an overall validation Dice score of approximately **0.7001**.

These are experimental validation results and do not represent clinical performance.

## Streamlit Application

The project includes an interactive web application for running 3D brain tumor segmentation locally.

### Application Features

1. Upload a 4D NIfTI MRI file (`.nii` or `.nii.gz`).
2. Load the trained 3D U-Net model.
3. Preprocess the four MRI modalities.
4. Generate segmentation predictions.
5. Visualize segmentation overlays in axial, coronal, and sagittal views.
6. Calculate individual tumor region volumes.
7. Download the predicted segmentation as a NIfTI file.

### Application Workflow

```text
MRI Upload
    |
    v
MRI Preprocessing
    |
    v
Trained 3D U-Net
    |
    v
Sliding Window Inference
    |
    v
Tumor Segmentation
    |
    v
Visualization and Volume Analysis
    |
    v
Download NIfTI Prediction
```

## Project Structure

```text
Brain-tumor-Detection-using-deep-learning/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── inference/
│   ├── __init__.py
│   └── model.py
│
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── prediction.py
│   ├── visualization.py
│   └── statistics.py
│
├── 3D_model.ipynb
├── BrainTumor_3D_MONAI_All_Dataset_Training.ipynb
└── EfficientNetB0.ipynb
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Rupesh756/Brain-tumor-Detection-using-deep-learning.git
cd Brain-tumor-Detection-using-deep-learning
```

### 2. Create a virtual environment

For Windows:

```powershell
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install PyTorch CPU

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install project dependencies

```powershell
python -m pip install numpy==2.2.6 scipy==1.15.3 pandas==2.2.3 scikit-learn==1.6.1
python -m pip install monai==1.6.2 nibabel streamlit==1.45.1 matplotlib==3.10.7 Pillow==11.3.0 plotly
```

## Model Checkpoint Setup

The trained checkpoint is not included in this repository because of its file size.

Place the trained model checkpoint in the following location:

```text
checkpoints/
└── best_model_v3.pth
```

The application loads the checkpoint from:

```text
checkpoints/best_model_v3.pth
```

You must obtain the trained checkpoint separately before running the application.

## Run the Application

After installing the dependencies and placing the checkpoint in the correct location, execute:

```powershell
streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

Upload a compatible four-modality 3D NIfTI MRI volume to perform segmentation.

## Future Improvements

- Improve segmentation performance for smaller tumor regions.
- Evaluate generalization across different MRI datasets and scanners.
- Integrate more advanced 3D segmentation architectures.
- Improve model inference speed and memory efficiency.
- Add more comprehensive Grad-CAM visualization.
- Explore tumor localization and volumetric analysis.
- Evaluate performance on external datasets.

## Disclaimer

This project is developed for academic and research purposes. Its predictions are not intended to replace professional medical diagnosis or clinical decision-making.

## Author

**Rupesh Kumar**

Computer Science and Engineering Student

GitHub: [@Rupesh756](https://github.com/Rupesh756)

Project Repository: [Brain Tumor Detection and Segmentation Using Deep Learning](https://github.com/Rupesh756/Brain-tumor-Detection-using-deep-learning)