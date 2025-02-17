# README - Calcium Correlation Tool 🔬
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)  
[![Cellpose](https://img.shields.io/badge/cellpose-2.0-orange.svg)](https://github.com/MouseLand/cellpose)

This repository provides a pipeline for **cell segmentation, tracking, and correlation analysis** in time-lapse microscopy data. Utilizing deep learning models like **Cellpose**, it enables precise segmentation and tracking of cells across frames, offering insights into their **morphological changes and fluorescence intensity variations** over time. The workflow includes **automated segmentation, interactive refinement via Napari, and correlation analysis** to uncover dynamic cellular behaviors.

![Image](https://github.com/baltuni/Calcium-Correlation-Tool/blob/main/021024_cluster_3_25.png?raw=true)

## 🏗 Repository Overview

| 📂 File                      | 📌 Description                                           |
| :--------------------------- | :-----------------------------------------------         |
| 0_segmentation_utils.py      | Core segmentation & tracking functions (Cellpose)        |
| 1_generate_masks.ipynb       | Generates segmentation masks from `.tif` images          |
| 2_tracking_refinement.ipynb  | Filters, refines, and visualizes tracked cells           |
| 3_correlation_analysis.ipynb | Performs correlation analysis on tracked cell properties |

<details>

<summary>File descriptions</summary>

### `0_segmentation_utils.py`
Core functions for **image segmentation and tracking** using a pre-trained Cellpose model:
- **`get_segmentation`** – Segments images using Cellpose.
- **`get_tracked_masks`** – Tracks segmented cells across time.
- **Utilities** – Image preprocessing and saving segmented masks.

### `1_generate_masks.ipynb`
Generates segmentation masks and tracks cells in time-lapse microscopy images:
- Loads raw `.tif` image stacks.
- Applies **Cellpose** for segmentation.
- Calls `get_tracked_masks` to maintain consistent cell labels.
- Saves segmentation masks for further analysis.

### `2_tracking_refinement.ipynb`
Processes segmentation results and refines tracked cell data:
- Loads segmentation masks and raw images.
- Extracts **common cells** across frames.
- Filters out inconsistently segmented cells.
- Provides **interactive visualization via Napari**.
- Computes **cell perimeter, radius, and mass** over time.

### `3_correlation_analysis.ipynb`
Performs correlation analysis on **tracked cell properties**:
- Loads **pre-processed tracking data** (`.pkl` files).
- Computes **cross-correlations** between cell properties.
- Visualizes raw images and segmented masks.
- Saves **correlation plots** for further analysis.

</details>

## 🛠️ Installation

### 🏡 Environment Setup

* Run the following code in Anaconda PowerShell Prompt to create the conda environment
```bash
conda create -n cctenv python=3.9 numpy scipy pandas matplotlib tqdm scikit-image tifffile napari cellpose opencv pillow networkx
```

* After installing the environment run the following code to activate it:
```bash
conda activate cctenv
```

## 🖥️ Segmentation Setup

### 🗂️ Workflow

* It is recommended to download `Workflow.zip` and extract it somewhere on your workstation for better directory structure.

<details>

<summary>Click to expand</summary>

```bash
User                                    # YourName
  │
  ├── masks_tracked                     # folders with masked .tif files are stored here
  │   ├── Model3                        # an instance of such a folder
  │   └── YourNewModel                  # your instance of such a folder
  │
  ├── output_path                       # folders with .pkl files are stored here
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

### 🪄 Manual Segmentation

* Before training, **create labels using Napari** (make sure you have the conda environment activated):
```bash
napari
```
Once open, **drag raw images** into Napari and manually create labels.

### 📐 Creating a Model

* Ensure the single-frame labeled images and single-frame raw images are in `cellpose_train`, then run:
```bash
python -m cellpose --train --use_gpu --verbose --n_epochs 2000 --dir D:\User\training_images_for_cellpose\cellpose_train\ --img_filter _ --mask_filter _label --pretrained_model None
```

## 📚 Running the Notebooks

* To start VSCode, make sure you have the conda environment activated so that you can run the following code in the Anaconda PowerShell Prompt:
```bash
code
```

<details>

<summary>Step-by-step description</summary>

### **📕 Generate Segmentation Masks**

Run `1_generate_masks.ipynb` to:
- Load raw images and apply **Cellpose segmentation**.
- Track segmented cells across frames.
- Save results as `.tif` files.

### **📗 Analyze & Refine Masks**

Run `2_tracking_refinement.ipynb` to:
- Load segmentation masks and filter unreliable cells.
- Identify **common cells** across frames.
- Compute **cell properties** (area, radius, intensity).
- Visualize results interactively in **Napari**.
- Save processed data for correlation analysis.

### **📘 Perform Correlation Analysis**

Run `3_correlation_analysis.ipynb` to:
- Load tracking data (`.pkl` files).
- Extract **fluorescence intensity and movement metrics**.
- Compute **cross-correlations** between cell behaviors.
- Generate **correlation heatmaps and statistical summaries**.
- Export results for further analysis.

</details>

## 🧰 Usage

This pipeline is designed for **time-lapse microscopy image analysis**, specifically:
- **Tracking cell movement and morphological changes**.
- **Measuring fluorescence intensity variations over time**.
- **Detecting correlations between cellular behaviors**.

## 🤝 Contributing

**We welcome contributions!** Here's how you can help:

### 🐞 Reporting Issues
- If you find a bug, please **open an issue** [here](https://github.com/baltuni/Calcium-Correlation-Tool/issues).
- Provide **steps to reproduce the issue**, expected behavior, and actual behavior.

### 🛠 Improving the Code
- **Fork** the repository.
- Create a **feature branch**: `git checkout -b feature-name`
- **Commit** your changes: `git commit -m "Added new feature"`
- **Push** your branch: `git push origin feature-name`
- Open a **pull request**
