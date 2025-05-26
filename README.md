<div align="center">
  <br>
  </br>
<img src="https://github.com/baltuni/Calcium-Correlation-Tool/blob/main/assets/readme_logo.png" alt="CCT Logo"  width="600"/>
  <br>
  </br>
<h1>Calcium Correlation Tool</h1>
</div>

This repository provides a pipeline for **cell segmentation, tracking, and correlation analysis** in time-lapse microscopy data. Utilizing deep learning models like **Cellpose**, it enables precise segmentation and tracking of cells across frames, offering insights into their morphological changes and fluorescence intensity variations over time. The workflow includes:

- **Automated segmentation** using deep learning.
- **Interactive refinement** via **Napari**.
- **Tracking cell movement** across frames.
- **Correlation analysis** to uncover dynamic cellular behaviors.

## 🏗 Repository Overview

| 📂 File                      | 📌 Description                                          |
| :--------------------------- | :-----------------------------------------------         |
| 0_cct_utils.py               | Core segmentation, tracking, and analysis functions      |
| 1_generate_masks.ipynb       | Generates segmentation masks from `.tif` images          |
| 2_tracking_refinement.ipynb  | Filters, refines, and saves tracked cell data            |
| 3_correlation_analysis.ipynb | Performs correlation analysis on tracked cell properties |

<details>

<summary>Detailed file descriptions</summary>

### `0_cct_utils.py`
- Contains core functions for cell segmentation, tracking, and correlation analysis.
- Uses **Cellpose** for deep-learning-based segmentation.
- Provides functions for:
  - Preprocessing and running segmentation models.
  - Tracking cells across frames (`track_Y`, `get_tracked_masks`).
  - Filtering cell pairs by distance, smoothing intensities, and calculating cross-correlations.
  - Building null distributions for statistical thresholding.
  - Generating LaTeX tables and network visualizations of correlations.
  - Calculating and visualizing centers of mass (COM) of cells.

### `1_generate_masks.ipynb`
- Implements the segmentation pipeline for generating cell masks.
- Steps include:
  - Loading time-lapse microscopy data.
  - Splitting the data into chunks for processing.
  - Running segmentation using **Cellpose** on each chunk.
  - Relabeling masks across chunks to ensure consistent tracking.
  - Saving segmented masks as `.tif` files.

### `2_tracking_refinement.ipynb`
- Refines segmented masks and tracks cells over time.
- Key functionalities:
  - Filters cells by occurrence in frames (`get_common_cells`).
  - Adds a **Napari Points Layer** for manual removal of incorrectly placed labels.
  - Calculates cell intensities and centers of mass (COM) using `cct_utils`.
  - Saves refined tracking data into `.pkl` files for correlation analysis.

### `3_correlation_analysis.ipynb`
- Analyzes dynamic cell behaviors through correlation studies.
- Includes:
  - Loading `.pkl` tracking data and corresponding raw images and masks.
  - Normalizing and smoothing cell intensity traces (`apply_smoothing_to_normalized`).
  - Calculating cross-correlations between cells, filtering by distance, and identifying significant correlations.
  - Building null distributions and determining statistical thresholds (`get_significant_correlation_threshold`).
  - Generating LaTeX tables summarizing top and bottom correlations.
  - Visualizing correlation networks and COMs in **Napari**.
  - **Note**: Requires `.pkl` files generated from `2_tracking_refinement.ipynb`.

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

It is recommended to download [`Workflow.zip`](https://github.com/baltuni/Calcium-Correlation-Tool/blob/main/assets/Workflow.zip) and extract it somewhere on your workstation for better directory structure.

<details>

<summary>Click to expand</summary>

```bash
User                                    # YourName
  │
  ├── masks_tracked                     # folders with masked .tif files are stored here
  │   ├── ModelAB1                      # an instance of such a folder
  │   └── YourNewModel                  # your instance of such a folder
  │
  ├── pkl_data                          # folders with .pkl files are stored here
  │   ├── ModelAB1                      # an instance of such a folder
  │   └── YourNewModel                  # your instance of such a folder
  │
  ├── plots                             # plots are saved here
  │   └── latex                         # tex files for latex tables are saved here
  │
  ├── raw_data                          # folders with full framed raw .tif files
  │   ├── ModelAB1                      # an instance of such a folder
  │   └── YourNewModel                  # your instance of such a folder
  │
  ├── saved_models                      # folders with cellpose models are stored here
  │   ├── ModelAB1                      # an instance of such a folder
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
- Run the segmentation pipeline using **Cellpose**, splitting large files into chunks.
- Relabel masks across chunks for consistent cell tracking.
- Run the segmentation pipeline and generate cell masks.
- Save segmented masks as `.tif` files.

### **📗 Track & Refine Masks**
**Execute** `2_tracking_refinement.ipynb` **to track and refine cell segmentation:**
- Load segmented masks from the previous step.
- Filter cells by occurrence (`get_common_cells`).
- Add a **Napari Points Layer** for manual removal of missegmented labels.
- Calculate cell intensities and centers of mass using `cct_utils`.
- Save the tracking and refinement results as `.pkl` files for correlation analysis.
- Important: Ensure you have run `1_generate_masks.ipynb` before this.

### **📘 Correlation Analysis**
**Execute** `3_correlation_analysis.ipynb` **to analyze fluorescence intensity correlations:**
- Load `.pkl` tracking data and corresponding raw images/masks.
- Normalize and smooth intensity traces for each cell.
- Calculate cross-correlations and identify significant correlations via null distribution.
- Generate LaTeX tables of top and bottom correlation pairs.
- Visualize correlation networks and centers of mass in **Napari**.
- Important: Ensure you have run `2_tracking_refinement.ipynb` before this.

</details>

---

## 🧰 Usage

This pipeline is designed for **time-lapse microscopy image analysis**, specifically:
- **Tracking cell movement and morphological changes**.
- **Measuring fluorescence intensity variations over time**.
- **Detecting correlations and constructing interaction networks between cells.**.

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
