# README - Calcium Correlation Tool (Cell Segmentation and Tracking Analysis)

This repository contains Jupyter Notebooks and Python scripts for cell segmentation, tracking, and analysis using deep learning models (Cellpose) and various image processing techniques. The workflows are designed for time-lapse microscopy data, enabling users to analyze cell behavior over time.

## Files Overview

### 1. `0_python_test.py`
This script contains core functions for segmenting images using a pre-trained Cellpose model and tracking cell movements across frames. It includes:
- **`get_segmentation`**: Segments images using Cellpose.
- **`get_tracked_masks`**: Tracks segmented cells across time.
- **Utilities for image preprocessing and saving segmented masks**.

### 2. `1_max_create_masks.ipynb`
This notebook generates segmentation masks for time-lapse microscopy images and tracks cells across frames. It:
- Loads raw microscopy image stacks (`.tif`).
- Applies **Cellpose** for segmentation.
- Calls `get_tracked_masks` to maintain consistent cell labels over time.
- Saves the segmentation masks for further analysis.

### 3. `4_analyse_output_radius_circle_mass_modified.ipynb`
This notebook processes the segmentation results and refines tracked cell data. It:
- Loads segmentation masks and raw images.
- Extracts **common cells** (cells that persist across frames).
- Filters out inconsistently segmented cells.
- Provides **interactive visualization with Napari** for manual corrections.
- Calculates **cell perimeter, radius, and mass** over time.

### 4. `6_correlation.ipynb`
This notebook performs correlation analysis on tracked cell properties, such as fluorescence intensity and movement patterns. It:
- Loads **pre-processed tracking data** (`.pkl` files) from previous steps.
- Computes **cross-correlations** between cell properties.
- Visualizes the raw image and filtered segmentation masks.
- Saves **correlation plots** for further analysis.

## Installation

### Creating a Model
Ensure you have the raw and labeled images in the `cellpose_train` folder before you run the following code to create the model:
```bash
python -m cellpose --train --use_gpu --verbose --n_epochs 2000 --dir D:\Bestun\training_images_for_cellpose\cellpose_train\ --img_filter _ --mask_filter _label --pretrained_model None
```

### Prerequisites
Ensure you pick the following kernel when running the notebooks:

`cellpose (Python 3.8.16)`

### Running the Notebooks
1. **Generate segmentation masks:**
   - Run `1_max_create_masks.ipynb` to create cell segmentation masks.
2. **Analyze and refine masks:**
   - Run `4_analyse_output_radius_circle_mass_modified.ipynb` to refine segmentation and filter out unreliable cells.
3. **Perform correlation analysis:**
   - Run `6_correlation.ipynb` to analyze time-series cell behavior.

## Usage
This pipeline is designed for analyzing time-lapse microscopy images, particularly for:
- **Tracking cell movement and shape changes.**
- **Measuring fluorescence intensity variations over time.**
- **Detecting correlations between cellular behaviors.**

## Contributing
Feel free to submit issues or pull requests to improve this workflow.
