# README - Calcium Correlation Tool (Cell Segmentation & Tracking Analysis)

This repository provides a pipeline for **cell segmentation, tracking, and correlation analysis** in time-lapse microscopy data. Utilizing deep learning models like **Cellpose**, it enables precise segmentation and tracking of cells across frames, offering insights into their **morphological changes and fluorescence intensity variations** over time. The workflow includes **automated segmentation, interactive refinement via Napari, and correlation analysis** to uncover dynamic cellular behaviors.

![Image](https://github.com/baltuni/Calcium-Correlation-Tool/blob/main/021024_cluster_3_25.pdf?raw=true)

## Repository Overview

### `0_python_test.py`
Core functions for **image segmentation and tracking** using a pre-trained Cellpose model:
- **`get_segmentation`** – Segments images using Cellpose.
- **`get_tracked_masks`** – Tracks segmented cells across time.
- **Utilities** – Image preprocessing and saving segmented masks.

### `1_max_create_masks.ipynb`
Generates segmentation masks and tracks cells in time-lapse microscopy images:
- Loads raw `.tif` image stacks.
- Applies **Cellpose** for segmentation.
- Calls `get_tracked_masks` to maintain consistent cell labels.
- Saves segmentation masks for further analysis.

### `4_analyse_output_radius_circle_mass_modified.ipynb`
Processes segmentation results and refines tracked cell data:
- Loads segmentation masks and raw images.
- Extracts **common cells** across frames.
- Filters out inconsistently segmented cells.
- Provides **interactive visualization via Napari**.
- Computes **cell perimeter, radius, and mass** over time.

### `6_correlation.ipynb`
Performs correlation analysis on **tracked cell properties**:
- Loads **pre-processed tracking data** (`.pkl` files).
- Computes **cross-correlations** between cell properties.
- Visualizes raw images and segmented masks.
- Saves **correlation plots** for further analysis.

## Installation

### Environment Setup
Activate the Conda environment before running the notebooks:
```bash
conda activate cellpose
```

### Manual Segmentation
Before training, **create labels using Napari**:
```bash
napari
```
Once open, **drag raw images** into Napari and manually create labels.

### Creating a Model
Ensure labeled images and raw images are in `cellpose_train`, then run:
```bash
python -m cellpose --train --use_gpu --verbose --n_epochs 2000 --dir D:\User\training_images_for_cellpose\cellpose_train\ --img_filter _ --mask_filter _label --pretrained_model None
```

### Kernel Requirement
Use the following Conda kernel when running notebooks:
**`cellpose (Python 3.8.16)`**

## Running the Notebooks

### **1. Generate Segmentation Masks**
Run `1_max_create_masks.ipynb` to:
- Load raw images and apply **Cellpose segmentation**.
- Track segmented cells across frames.
- Save results as `.tif` files.

### **2. Analyze & Refine Masks**
Run `4_analyse_output_radius_circle_mass_modified.ipynb` to:
- Load segmentation masks and filter unreliable cells.
- Identify **common cells** across frames.
- Compute **cell properties** (area, radius, intensity).
- Visualize results interactively in **Napari**.
- Save processed data for correlation analysis.

### **3. Perform Correlation Analysis**
Run `6_correlation.ipynb` to:
- Load tracking data (`.pkl` files).
- Extract **fluorescence intensity and movement metrics**.
- Compute **cross-correlations** between cell behaviors.
- Generate **correlation heatmaps and statistical summaries**.
- Export results for further analysis.

## Usage
This pipeline is designed for **time-lapse microscopy image analysis**, specifically:
- **Tracking cell movement and morphological changes**.
- **Measuring fluorescence intensity variations over time**.
- **Detecting correlations between cellular behaviors**.

## Contributing
Contributions are welcome! Submit issues or pull requests to improve this workflow.
