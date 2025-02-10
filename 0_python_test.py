import numpy as np
from datetime import date
from scipy import ndimage
from tifffile import imwrite
from os.path import exists
import logging
from tqdm import tqdm

# Set up logging
log_filename = "tracking_log.txt"
logging.basicConfig(filename=log_filename, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def get_cell_labels(masks):
    """ Obtain a list of cell labels """
    if len(masks.shape) == 3:
        return [np.unique(mask[mask != 0]) for mask in masks]
    return np.unique(masks[masks != 0])

def get_centers_of_mass(masks, cell_labels=None):
    """ Get centers of mass for each cell """
    if cell_labels is None:
        cell_labels = get_cell_labels(masks)
    if len(masks.shape) == 3:
        return [np.array(ndimage.center_of_mass(mask, mask, labels)) for mask, labels in zip(masks, cell_labels)], cell_labels
    return np.array(ndimage.center_of_mass(masks, masks, cell_labels)), cell_labels

def assign_random_cell_labels(mask):
    """ Assign random cell labels to mask """
    labels = np.unique(mask[mask != 0])
    np.random.shuffle(labels)
    randomized_mask = np.zeros_like(mask)
    for lbl, rnd_lbl in zip(np.unique(mask[mask != 0]), labels):
        randomized_mask[mask == lbl] = rnd_lbl
    return randomized_mask

def _save_masks(masks, name=None, savedir=None):
    """ Save masks as single .tif file """
    name = name or str(date.today())
    if savedir:
        if not exists(savedir):
            os.makedirs(savedir)
        imwrite(f"{savedir}/{name}_masks.tif", masks)
    else:
        imwrite(f"{name}_masks.tif", masks)

def get_tracked_masks(masks, dist_limit=20, backtrack_limit=5, random_labels=False, save=False, name=None, savedir=None):
    """ Track cells across frames and maintain consistent labeling. """
    logging.info(f"Tracking started. Input masks shape: {masks.shape}")
    tracked_masks = np.zeros_like(masks)
    COMs, roi_labels = get_centers_of_mass(masks)
    logging.info(f"Number of frames with centers of mass: {len(COMs)}")
    
    if random_labels:
        tracked_masks[0] = assign_random_cell_labels(masks[0])
    else:
        tracked_masks[0] = masks[0]
    
    used_labels = set(np.unique(tracked_masks[0]))
    for imnr in tqdm(range(len(masks)), desc="Tracking masks", unit=" frames"):
        logging.info(f"Processing frame {imnr}...")
        if imnr == 0:
            continue
        new_cells = 0
        ROI_labels_imnr = roi_labels[imnr]
        if len(COMs[imnr]) == 0:
            logging.warning(f"No centers of mass detected in frame {imnr}. Skipping frame.")
            continue
        for COM_idx, COM_label in zip(range(len(COMs[imnr])), ROI_labels_imnr):
            ref_im_idx = -10
            for k in range(1, backtrack_limit):
                if imnr - k < 0:
                    break
                distances = np.linalg.norm(np.array(COMs[imnr - k]) - np.array(COMs[imnr][COM_idx]), axis=1)
                if np.min(distances) < dist_limit:
                    ref_im_idx = imnr - k
                    mcc = COMs[ref_im_idx][np.argmin(distances)]
                    cell_label = tracked_masks[ref_im_idx][round(mcc[0]), round(mcc[1])]
                    logging.info(f"Frame {imnr}: Matched {COM_label} with label {cell_label} in frame {ref_im_idx}")
                    break
            if ref_im_idx == -10:
                new_cells += 1
                cell_label = np.max(tracked_masks[:imnr]) + new_cells
                while cell_label in used_labels:
                    cell_label += 1
                logging.info(f"Frame {imnr}: No match found. Assigned new label {cell_label}")
            used_labels.add(cell_label)
            tracked_masks[imnr][masks[imnr] == COM_label] = cell_label
        logging.info(f"Frame {imnr} processed. Unique labels: {np.unique(tracked_masks[imnr])}")
    if save:
        _save_masks(tracked_masks, name=name, savedir=savedir)
        logging.info(f"Masks saved to {savedir} with name {name}_masks.tif")
    logging.info("Tracking complete.")
    return tracked_masks
