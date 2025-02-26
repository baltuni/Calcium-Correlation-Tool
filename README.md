<p align="center">
  <a href="https://github.com/baltuni/Calcium-Correlation-Tool">
    <img loading="lazy" alt="CCT" src="https://github.com/baltuni/Calcium-Correlation-Tool/blob/main/readme_logo.png" width="100%"/>
  </a>
</p>

# Calcium Correlation Tool

This repository provides a pipeline for **cell segmentation, tracking, and correlation analysis** in time-lapse microscopy data. Utilizing deep learning models like **Cellpose**, it enables precise segmentation and tracking of cells across frames, offering insights into their morphological changes and fluorescence intensity variations over time. The workflow includes:

- **Automated segmentation** using deep learning.
- **Interactive refinement** via **Napari**.
- **Tracking cell movement** across frames.
- **Correlation analysis** to uncover dynamic cellular behaviors.

## 🏗 Repository Overview

| 📂 File                      | 📌 Description                                           |
| :--------------------------- | :-----------------------------------------------         |
| 0_segmentation_utils.py      | Core segmentation & tracking functions (Cellpose)        |
| 1_generate_masks.ipynb       | Generates segmentation masks from `.tif` images          |
| 2_tracking_refinement.ipynb  | Filters, refines, and visualizes tracked cells           |
| 3_correlation_analysis.ipynb | Performs correlation analysis on tracked cell properties |

<details>

<summary>Detailed file descriptions</summary>

### `0_segmentation_utils.py`
- Contains helper functions for cell segmentation.
- Uses **Cellpose** for deep-learning-based segmentation.
- Provides functions for:
  - Image preprocessing before segmentation.
  - Running segmentation models.
  - Post-processing segmented masks (e.g., filtering, smoothing).
  - Visualizing segmentation results.

### `1_generate_masks.ipynb`
- Implements the segmentation pipeline for generating cell masks.
- Steps include:
  - Loading time-lapse microscopy data.
  - Running segmentation using **Cellpose**.
  - Saving segmented masks for further analysis.
  - Interactive refinement using **Napari** for manual corrections.

### `2_tracking_refinement.ipynb`
- Tracks cell movements across time-lapse frames.
- Key functionalities:
  - Assigns unique IDs to segmented cells in consecutive frames.
  - Corrects tracking errors using heuristic rules.
  - Provides visualization tools for overlaying tracked paths.
  - Allows interactive refinement of cell tracks in **Napari**.

### `3_correlation_analysis.ipynb`
- Analyzes dynamic cell behaviors through correlation studies.
- Includes:
  - Extracting fluorescence intensity over time for each cell.
  - Computing correlations between intensity signals of different cells.
  - Filtering significant correlations for network analysis.
  - Visualizing correlation networks with **Napari**.

</details>

---

## 🛠️ Installation

### 🏡 Environment Setup

Run the following code in Anaconda PowerShell Prompt to create the conda environment:

```bash
conda create -n cctenv python=3.9 numpy scipy pandas matplotlib tqdm scikit-image tifffile napari cellpose opencv pillow networkx
```

After installing the environment run the following code to activate it:

```bash
conda activate cctenv
```

---

## 🖥️ Segmentation Setup

### 🗂️ Workflow

It is recommended to download `Workflow.zip` and extract it somewhere on your workstation for better directory structure.

<details>

<summary>Click to expand</summary>

```bash
User                                    # YourName
  │
  ├── masks_tracked                     # folders with masked .tif files are stored here
  │   ├── Model3                        # an instance of such a folder
  │   └── YourNewModel                  # your instance of such a folder
  │
  ├── pkl_data                          # folders with .pkl files are stored here
  │   ├── Model3                        # an instance of such a folder
  │   └── YourNewModel                  # your instance of such a folder
  │
  ├── plots                             # plots are saved here
  │   └── latex                         # tex files for latex tables are saved here
  │
  ├── raw_data                          # folders with full framed raw .tif files
  │   ├── Model3                        # an instance of such a folder
  │   └── YourNewModel                  # your instance of such a folder
  │
  ├── saved_models                      # folders with cellpose models are stored here
  │   ├── Model3                        # an instance of such a folder
  │   └── YourNewModel                  # your instance of such a folder
  │
  └── training_images_for_cellpose      # used for cellpose training
      ├── cellpose_train                # store files from label and raw here
      │   └── models                    # cellpose model gets saved here
      │
      ├── label                         # labeled raw single frame .tif files
      └── raw                           # raw single frame .tif files
```

</details>

### 🩻 Image Conversion

Use ImageJ to convert your `.czi` images to `.tif` files and save them in `...\User\raw_data`.
Make sure you save single-framed images from your `.tif` files in in `...\User\training_images_for_cellpose\raw`.

### 🪄 Manual Segmentation

Before training, **create labels using Napari** (make sure you have the conda environment activated):

```bash
napari
```

Once open, **drag raw images** into Napari and manually create labels.

### 📐 Creating a Model

Ensure the single-framed images (both raw and labeled) are in `...\User\training_images_for_cellpose\`, then run:
```bash
python -m cellpose --train --use_gpu --verbose --n_epochs 2000 --dir ...\User\training_images_for_cellpose\cellpose_train\ --img_filter _ --mask_filter _label --pretrained_model None
```

---

## 📚 Running the Notebooks

To start VSCode, make sure you have the conda environment activated so that you can run the following code in the Anaconda PowerShell Prompt:

```bash
code
```

<details>

<summary>Step-by-step instructions</summary>

### **📕 Generate Segmentation Masks**
**Execute** `1_generate_masks.ipynb` **to segment cells in your microscopy data:**
- Open the notebook in **VSCode**.
- Load raw microscopy images from the specified directory.
- Define segmentation parameters (`occurrence_limit`, `dist_limit`).
- Use **Cellpose** to segment cells.
- Run the segmentation pipeline and generate cell masks.
- Save the segmented masks for further analysis.

### **📗 Analyze & Refine Masks**
**Execute** `2_tracking_refinement.ipynb` **to track and refine cell segmentation:**
- Load segmented masks from the previous step.
- Use a tracking algorithm to assign unique IDs to cells across frames.
- Apply filtering (`get_common_cells`) to remove unreliable labels.
- Use **Napari** for manual correction of incorrectly placed labels.
- Ensure temporal consistency of cell masks.
- Save the refined segmentation and tracking results.

### **📘 Perform Correlation Analysis**
**Execute** `3_correlation_analysis.ipynb` **to analyze fluorescence intensity correlations:**
- Load `.pkl` files containing tracked cell data.
- Load raw microscopy images and corresponding segmented masks.
- Compute fluorescence intensity variations over time for each cell.
- Perform correlation analysis between different cells.
- Visualize and filter significant correlations using **Napari**.
- Save correlation plots and tables for further analysis.

</details>

---

## 🧰 Usage

This pipeline is designed for **time-lapse microscopy image analysis**, specifically:
- **Tracking cell movement and morphological changes**.
- **Measuring fluorescence intensity variations over time**.
- **Detecting correlations between cellular behaviors**.

---

## 🤝 Contributing

**We welcome contributions! Here's how you can help:**

### 🐞 Reporting Issues
- If you find a bug, please **open an issue** [here](https://github.com/baltuni/Calcium-Correlation-Tool/issues).
- Provide **steps to reproduce the issue**, expected behavior, and actual behavior.

### 🪛 Improving the Code
- **Fork** the repository.
- Create a **feature branch**: `git checkout -b feature-name`
- **Commit** your changes: `git commit -m "Added new feature"`
- **Push** your branch: `git push origin feature-name`
- Open a **pull request**

---
