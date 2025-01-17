from cellpose import models
from cellpose.io import imread
import numpy as np
from os import makedirs
from datetime import date
from scipy import ndimage
from tifffile import imwrite
from os.path import basename, splitext, exists
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import convolve2d
import pandas as pd

def get_segmentation(image_path, model_path, diam=40, save=False, savedir=None,
                     track=True, name=None):
    """ segment an image
    
    Takes an image stack and segments each image separately with
    the given model.
    Parameters
    ---------------
    image_path: str or 3D array
        the path to the image to be segmented or the 3D array of the image.
        The 3D array should follow the structure given from open_image_stack
        in order to be segmented correctly.
    model_path: str
        the path to the model to use for segmenting
    diam: float (optional)
        approximate diameter of the cells in pixels. Recommended to use the
        automatic estimation in the cellpose API.
    save: bool (optional)
        if True, the masks will be saved as a single .tif file
    savedir: str (optional)
        the path to the directory in which to save the generated masks.
    track: bool (optional)
        if True, the cells are tracked using the standard variables in
        get_tracked_cells. For tuning of the tracking, use get_tracked_cells
        directly.
    name: str (optional)
        name of the .tif file to which the masks will be saved. "_masks.tif"
        is always added to the end of the name. If None, the name will be
        "[image name]_masks.tif". See _save_masks() for further details.
    Returns
    ---------------
    masks: 3D array
        the generated masks, where the area of each cell has a unique integer
        value (the cell label) greater than or eaqual to one and the
        background has the value 0.
    """

    model = models.CellposeModel(gpu = True, pretrained_model=model_path)
    if type(image_path)==str:
        imgs = open_image_stack(image_path)
    else:
        imgs = image_path
    masks, flows, styles = model.eval(imgs, diameter=diam, channels=[0,0],
                                      flow_threshold=0.4, do_3D=False)
    masks = np.array(masks)

    if save==True and name==None:
        name=splitext(basename(image_path))[0]
    if track:
        masks = get_tracked_masks(masks=masks, save=save, name=name, savedir=savedir)
    if save and not track:
        _save_masks(savedir=savedir, masks=masks,
                    name=name)
    return masks


def open_image_stack(image_path):
    """ open a .tif image stack
    
    Modifies the data to work with the segmentation function.
    Parameters
    ---------------
    image_path: str
        the path to the image to be opened
    Returns
    ---------------
    stack: 3D array
        the image stack as a 3D array
    """
    img = imread(image_path)
    stack = [np.array(i) for i in img] # Necessary for correct segmentation
    return stack


def _save_masks(masks, name=None, savedir=None):
    """ save masks as single .tif file

    Parameters
    ---------------
    masks: 3D array
        the masks to be saved.
    name: str (optional)
        name of the file to be saved. Default name is 
        '[today's date]_masks.tif'. '_masks.tif' is always added to the name.
    savedir: str (optional)
        the directory in which to save the file. Default to save in current
        folder.
    Returns
    ---------------
        None
    """
    if name == None:
        name = str(date.today())
    if savedir == None:
        imwrite(name + "_masks.tif", masks)
    else:
        if exists(savedir) == False:
            makedirs(savedir)
        imwrite(savedir+"/"+name+"_masks.tif", masks)
    return None


def open_masks(file_path):
    """ open a mask file
    
    This function only exists because of previous problems with reading the
    mask files. It is actally really simple and the function is probably 
    totally unnecessary.
    Parameters
    ---------------
    file_path: str
        the path to the .tif file of the segmented images (masks).
    Returns
    ---------------
    masks: 3D array
        an array with all the masks for the images.
    """
    masks = imread(file_path)
    return masks


def extract_images_for_training(images_path, interval=50, savedir=None,
                                name=None):
    """Extracts images with a certain interval from image stack.
    
    Parameters
    ---------------
    images_path: str
        the path to the image stack (.tif file) to extract the images
        from
    interval: int (optional)
        every [interval]'th image will be extracted
    savedir: str (optional)
        directory in which to save the extracted images. Default to save
        the images in current directory.
    name: str (optional)
        name of the files. To the name, the image number will be added.
        Default to save files with the name of the image stack file,
        followed by the image number, e.g. ctl0035. 
    Returns
    ---------------
        None
    """

    stack = open_image_stack(images_path)
    im_idxs = np.arange(0,len(stack),interval)
    if name == None:
        name=splitext(basename(images_path))[0]
    for i in im_idxs:
        if savedir == None:
            imwrite(name + str("{0:03}".format(i)) + ".tif", stack[i])
        else:
            if exists(savedir) == False:
                makedirs(savedir)
            imwrite(savedir + "\\" + name + str("{0:03}".format(i)) + ".tif",
                    stack[i])
        
    return None


def get_cell_labels(masks):
    """ obtain a list of cell labels

    Finds all labels in each segmented image and returns these as a 
    2D array consisting of lists with all labels for each image.
    Parameters
    ---------------
    masks: 2D or 3D array
        previously generated masks
    Returns
    ---------------
    all_labels: 1D or 2D list
        If masks are 2D, a list of all labels is returned.
        If masks are 3D, a list of this kind of lists is returned.
    """
    if len(masks.shape) == 3:
        all_labels = []
        for mask in masks:
            labels = np.unique(mask[mask!=0])
            all_labels.append(labels)
        # all_labels = np.array(all_labels)
    if len(masks.shape) == 2:
        all_labels = np.unique(masks[masks!=0])
    return all_labels


def get_centers_of_mass(masks, cell_labels=None):
    """ get centers of mass for each cell

    Calculates the coordinates for the center of mass for each cell in each
    image.
    Parameters
    ---------------
    masks: 2D or 3D array
        previously generated segmentation mask(s)
    Returns
    ---------------
    COMs: 1D or 2D list of tuples
        all coordinates of centers of mass, sorted by image if masks are 3D
    labels: 1D or 2D list
        the labels of all the cells, in the same order as the centers of mass
    """
    if cell_labels is None:
        cell_labels = get_cell_labels(masks)
    if len(masks.shape) == 3:
        COMs = []
        for i in range(masks.shape[0]):
            COMs_i = np.array(ndimage.center_of_mass(masks[i], masks[i], cell_labels[i]))
            COMs.append(COMs_i)
    if len(masks.shape) == 2:
        COMs = np.array(ndimage.center_of_mass(masks, masks, cell_labels))
    return COMs, cell_labels

'''''''''''
def get_tracked_masks(masks, dist_limit=20, backtrack_limit=15, random_labels=False,
                      save=False, name=None, savedir=None):
    """ track the cells

    Tracks cells and returns a list of masks, where each separate 
    cell is given the same label (integer number) in every mask.
    Parameters
    ---------------
    masks: 3D array
        previously generated segmentation of cells
    dist_limit: int (optional)
        the longest distance, in pixels, that the center of mass is allowed
        to move from one image to the next for it to still count as the same
        cell. Make bigger if cells are moving a lot.
    backtrack_limit: int (optional)
        the maximum number of images back that the algorithm will search
        through to find a center of mass within the distance limit
        (dist_limit). Make smaller if cells are moving a lot.
    random_labels: bool
        if True, the cells will be assigned random lables from the start,
        rather than keeping the labels from the first image in masks
    save: bool (optional)
        if True, the masks are saved as a single .tif file
    name: str (optional)
        name of the file to be saved. Default name is '[today's date]'.
        '_masks.tif' is always added to the name.
    savedir: str (optional)
        the directory in which to save the file. Default to save in current
        folder.
    Returns
    ---------------
    tracked_masks: 3D array
        an aray with the same masks as given as input, but updated labels to
        match between images.
    """
    tracked_masks = np.zeros_like(masks)
    COMs, roi_labels = get_centers_of_mass(masks)

    if random_labels:
        tracked_masks[0] = assign_random_cell_labels(masks[0])
    else:
        tracked_masks[0] = masks[0]

    for imnr in range(1, len(masks)):
        new_cells = 0
        ROI_labels_imnr = roi_labels[imnr]
        for COM_idx, COM_label in zip(range(len(COMs[imnr])), ROI_labels_imnr):
            ref_im_idx = -10
            for k in range(1,backtrack_limit):
                # Get all distances between centers of mass of one
                # image and the one before.
                if imnr-k<0:
                    break
                distances = np.linalg.norm(np.array(COMs[imnr-k])
                                           - np.array(COMs[imnr][COM_idx]),
                                           axis=1)
                # If the smallest one is smaller than the dist_limit, exit loop
                if np.min(distances) < dist_limit:
                    ref_im_idx = imnr-k
                    mcc = COMs[ref_im_idx][np.argmin(distances)] # matched
                                                                 # cell
                                                                 # coordinate
                    cell_label = tracked_masks[ref_im_idx][
                        round(mcc[0])][round(mcc[1])]
                    break
            # If no matching cell is found in previous images:
            if ref_im_idx == -10:
                new_cells += 1
                cell_label = (np.max(tracked_masks[:imnr].flatten())
                             + new_cells)
            # Give area in new mask value corresponding to matched cell
            roi_coords = np.argwhere(masks[imnr].flatten() == COM_label)
            np.put(tracked_masks[imnr], roi_coords, cell_label)

    if save:
        # savedir = "NewMasks_"+str(date.today())
        _save_masks(tracked_masks, name=name, savedir=savedir)

    return tracked_masks

'''''''''

#Version of get_tracked_masks that has been debugged to avoid assigning the same label to multiple cells
import sys

#Changed the backtrack_limit to 5 (from 15)
def get_tracked_masks(masks, dist_limit=20, backtrack_limit=5, random_labels=False,
                      save=False, name=None, savedir=None):
    """ Track the cells
    Tracks cells and returns a list of masks, where each separate 
    cell is given the same label (integer number) in every mask.
    """
    # Debug: Check the shape of the input masks
    print(f"Input masks shape: {masks.shape}", flush=True)
    print(f"Unique labels in the first frame: {np.unique(masks[0])}", flush=True)
    sys.stdout.flush()

    tracked_masks = np.zeros_like(masks)
    COMs, roi_labels = get_centers_of_mass(masks)

    # Debug: Inspect the output of get_centers_of_mass
    print(f"Number of frames with centers of mass: {len(COMs)}", flush=True)
    for i, centers in enumerate(COMs):
        print(f"Frame {i}: {len(centers)} centers of mass", flush=True)

    print(f"Number of frames with ROI labels: {len(roi_labels)}", flush=True)
    for i, labels in enumerate(roi_labels):
        print(f"Frame {i}: ROI labels - {labels}", flush=True)
    sys.stdout.flush()

    if random_labels:
        tracked_masks[0] = assign_random_cell_labels(masks[0])
    else:
        tracked_masks[0] = masks[0]

    # Debug: Check initialized tracked_masks
    print(f"Tracked masks initialized for frame 0. Unique labels: {np.unique(tracked_masks[0])}", flush=True)
    sys.stdout.flush()

    # Set to keep track of used labels across all frames
    used_labels = set(np.unique(tracked_masks[0]))
    print(f"Initial used labels: {used_labels}", flush=True)
    sys.stdout.flush()

    # Loop through all frames starting from the second frame
    for imnr in range(1, len(masks)):
        print(f"\nProcessing frame {imnr}...", flush=True)
        sys.stdout.flush()
        new_cells = 0

        # Debug: Check ROI labels for the current frame
        ROI_labels_imnr = roi_labels[imnr]
        print(f"Frame {imnr} has {len(ROI_labels_imnr)} ROIs: {ROI_labels_imnr}", flush=True)

        if len(COMs[imnr]) == 0:
            print(f"No centers of mass detected in frame {imnr}. Skipping frame.", flush=True)
            sys.stdout.flush()
            continue

        # Process each center of mass and corresponding ROI
        for COM_idx, COM_label in zip(range(len(COMs[imnr])), ROI_labels_imnr):
            print(f"  Processing COM {COM_label} (index {COM_idx}) in frame {imnr}", flush=True)
            sys.stdout.flush()
            ref_im_idx = -10

            # Check for matches in previous frames
            for k in range(1, backtrack_limit):
                if imnr - k < 0:
                    break
                distances = np.linalg.norm(np.array(COMs[imnr - k])
                                           - np.array(COMs[imnr][COM_idx]),
                                           axis=1)
                print(f"    Checking frame {imnr - k}, distances: {distances}", flush=True)
                sys.stdout.flush()
                if np.min(distances) < dist_limit:
                    ref_im_idx = imnr - k
                    mcc = COMs[ref_im_idx][np.argmin(distances)]  # matched cell coordinates
                    cell_label = tracked_masks[ref_im_idx][
                        round(mcc[0])][round(mcc[1])]
                    print(f"    Matched with cell label {cell_label} in frame {ref_im_idx}", flush=True)
                    sys.stdout.flush()
                    break

            # If no matching cell is found, assign a new label
            if ref_im_idx == -10:
                new_cells += 1
                cell_label = np.max(tracked_masks[:imnr].flatten()) + new_cells

                # Ensure the new label is unique
                while cell_label in used_labels:
                    cell_label += 1
                print(f"    No match found. Assigned new label {cell_label}", flush=True)
                sys.stdout.flush()

            # Print the assigned label
            print(f"  Final label for ROI {COM_label}: {cell_label}", flush=True)
            sys.stdout.flush()

            # Update the set of used labels
            used_labels.add(cell_label)

            # Update the tracked masks for the current frame
            roi_coords = np.argwhere(masks[imnr].flatten() == COM_label)
            np.put(tracked_masks[imnr], roi_coords, cell_label)

        # Debug: Check labels in the current frame after processing
        print(f"Frame {imnr} processed. Unique labels: {np.unique(tracked_masks[imnr])}", flush=True)
        sys.stdout.flush()

    if save:
        _save_masks(tracked_masks, name=name, savedir=savedir)
        print(f"Masks saved to {savedir} with name {name}_masks.tif", flush=True)
        sys.stdout.flush()

    print("\nTracking complete.", flush=True)
    sys.stdout.flush()
    return tracked_masks




def get_tracked_masks_area_overlap(masks, overlap_limit=0.5, backtrack_limit=15,
                       random_labels=False, save=False, name=None,
                       savedir=None):
    """ track the cells

    Tracks cells and returns a list of masks, where each separate 
    cell is given the same label (integer number) in every mask.
    Parameters
    ---------------
    masks: 3D array
        previously generated segmentation of cells
    overlap_limit: float (optional)
        the fraction of overlap cells need to have to be considered the same
        cell.
    backtrack_limit: int (optional)
        the maximum number of images back that the algorithm will search
        through to find a center of mass within the distance limit
        (dist_limit). Make smaller if cells are moving a lot.
    random_labels: bool
        if True, the cells will be assigned random lables from the start,
        rather than keeping the labels from the first image in masks
    save: bool (optional)
        if True, the masks are saved as a single .tif file
    name: str (optional)
        name of the file to be saved. Default name is '[today's date]'.
        '_masks.tif' is always added to the name.
    savedir: str (optional)
        the directory in which to save the file. Default to save in current
        folder.
    Returns
    ---------------
    tracked_masks: 3D array
        an aray with the same masks as given as input, but updated labels to
        match between images.
    """
    tracked_masks = np.zeros_like(masks)

    if random_labels:
        tracked_masks[0] = assign_random_cell_labels(masks[0])
    else:
        tracked_masks[0] = masks[0]

    for imnr in range(1, len(masks)):
        new_cells = 0
        # Use overlap of area of cells instead of center of mass.
        for lbl in np.unique(masks[imnr][masks[imnr]!=0]):
            cell_area = np.count_nonzero(masks[imnr]==lbl)
            min_overlap = round(overlap_limit*cell_area)
            # print(min_overlap)
            for k in range(1, backtrack_limit+1):
                overlap_values = tracked_masks[imnr-k][masks[imnr]==lbl]
                counts = np.bincount(overlap_values)
                if max(counts)>=min_overlap and np.argmax(counts)!=0:
                    new_cell_value = np.argmax(counts)
                    break
                if k==backtrack_limit or imnr-k==0:    # No matching cell found
                    new_cells += 1
                    new_cell_value = np.max(tracked_masks[:imnr].flatten())+new_cells
                    break
            coords = np.argwhere(masks[imnr].flatten()==lbl)
            np.put(tracked_masks[imnr], coords, new_cell_value)

    if save:
        _save_masks(tracked_masks, name=name, savedir=savedir)

    return tracked_masks


# def get_cell_intensities(cell_label, tracked_cells, images, T=10,
    #                      normalize=None, hpf=0.0005, lpf=0.017, n=None, cutoff_peakheight=None):
    # """ get the intensities of a specified cell in all images
    
    # Calculates the intensity of a specified cell in each image.
    # If the cell does not appear in one of the images, the intensity
    # will be given by linear interpolation of existing values.
    # Parameters
    # ---------------
    # cell_label: int
    #     the label of the cell for which the intensities should be calculated
    # tracked_cells: 3D array
    #     previously tracked masks from which to get the cell locations
    # images: 3D array
    #     The images from which the tracked cells mask was generated and the
    #     intensity should be retrieved from
    # T : int or float
    #     period of imaging (time between images) in seconds
    # normalize: bool (optional)
    #     'minmax'
    #     if True, the intensities are min-max normalized to a range of 0 to 1
    # hpf: bool (optional)
    #     whether to do high-pass filtering or not
    # lpf: bool (optional)
    #     whether to do low-pass filtering or not
    # Returns
    # ---------------
    # mean_intensities: 1D array
    #     mean (normalized) intensity of the cell for each image
    # """
    # images_count = len(images)
    # intensities = np.zeros(images_count)

    # for i in range(images_count):
    #     intsies = images[i][tracked_cells[i]==cell_label]
    #     if np.any(intsies):
    #         intensities[i] = np.mean(intsies)
    #     else:
    #         intensities[i] = np.NaN

    # lost_intsies_idx = np.where(np.isnan(intensities))[0]
    # lost_intsies_idx = np.array_split(lost_intsies_idx,
    #                             np.where(np.diff(lost_intsies_idx) != 1)[0]+1)
    
    # # Use segmented area to get mean intensity
    # if len(lost_intsies_idx[0])>0:
    #     for idxs in lost_intsies_idx:
    #         start_idx = idxs[0]-1
    #         end_idx = idxs[-1]+1
    #         if start_idx<0:
    #             area = tracked_cells[end_idx]==cell_label
    #         elif end_idx>=len(images):
    #             area = tracked_cells[start_idx]==cell_label
    #         else:
    #             area = np.multiply(tracked_cells[start_idx]==cell_label,
    #                             tracked_cells[end_idx]==cell_label)
    #         for i in idxs:
    #             intensities[i] = np.mean(images[i][area])
            


    # # Extrapolate data
    # # intensities = pd.Series(intensities).interpolate().tolist()

    # if n:
    #     for i in range(n):
    #         intensities[i] = np.mean(intensities[0:i+n])
    #     for i in range(4,len(intensities)):
    #         intensities[i] = np.mean(intensities[i-n:i+n])
    # if hpf:
    #     hpf_cutoff_freq = hpf
    #     freqs = np.fft.fftfreq(len(intensities), T)
    #     filter_mask = np.abs(freqs) > hpf_cutoff_freq
    #     intensities_fft = np.fft.fft(intensities)
    #     filtered_signal = np.real(np.fft.ifft(intensities_fft*filter_mask))
    #     intensities = filtered_signal
    # if lpf:
    #     lpf_cutoff_freq = lpf
    #     freqs = np.fft.fftfreq(len(intensities), T)
    #     filter_mask = np.abs(freqs) < lpf_cutoff_freq
    #     intensities_fft = np.fft.fft(intensities)
    #     filtered_signal = np.real(np.fft.ifft(intensities_fft*filter_mask))
    #     intensities = filtered_signal
    # if normalize=='minmax':
    #     intensities = (intensities - np.min(intensities))/\
    #         (np.max(intensities)-np.min(intensities))
    # if cutoff_peakheight!=None:
    #     for i in range(len(intensities)):
    #         if intensities[i]<cutoff_peakheight:
    #             intensities[i] = 0
    # return np.array(intensities)

def get_cell_intensities(cell_label, tracked_cells, images, T=10):
    """ get the intensities of a specified cell in all images
    
    Calculates the intensity of a specified cell in each image.
    If the cell does not appear in one of the images, the intensity
    will be given by linear interpolation of existing values.
    Parameters
    ---------------
    cell_label: int
        the label of the cell for which the intensities should be calculated
    tracked_cells: 3D array
        previously tracked masks from which to get the cell locations
    images: 3D array
        The images from which the tracked cells mask was generated and the
        intensity should be retrieved from
    T : int or float
        period of imaging (time between images) in seconds
    Returns
    ---------------
    mean_intensities: 1D array
        mean (normalized) intensity of the cell for each image
    """
    images_count = len(images)
    intensities = np.zeros(images_count)

    for i in range(images_count):
        intsies = images[i][tracked_cells[i]==cell_label]
        if np.any(intsies):
            intensities[i] = np.mean(intsies)
        else:
            intensities[i] = np.NaN

    lost_intsies_idx = np.where(np.isnan(intensities))[0]
    lost_intsies_idx = np.array_split(lost_intsies_idx,
                                np.where(np.diff(lost_intsies_idx)!=1)[0]+1)
    
    # Use segmented area to get mean intensity
    if len(lost_intsies_idx[0])>0:
        for idxs in lost_intsies_idx:
            start_idx = idxs[0]-1
            end_idx = idxs[-1]+1
            if start_idx<0:
                area = (tracked_cells[end_idx]==cell_label)
            elif end_idx>=len(images):
                area = (tracked_cells[start_idx]==cell_label)
            else:
                area = np.multiply(tracked_cells[start_idx]==cell_label,
                                tracked_cells[end_idx]==cell_label)
            for i in idxs:
                intensities[i] = np.mean(images[i][area])
    return np.array(intensities)


def assign_random_cell_labels(mask):
    """ assign random cell labels to mask. I didn't find much use in this.

    Please, only provide one single mask, not a whole 3D array.
    Zero labels remain zero.
    Parameters
    ---------------
    mask: 2D array
        the mask for which the labels should be changed
    Returns
    ---------------
    randomized_mask: 2D array
        a mask with the labels shuffled around, but the general
        segmentation still remaining intact
    """
    labels = np.unique(mask[mask!=0])
    random_labels = labels.copy()
    np.random.shuffle(random_labels)
    randomized_mask = np.zeros_like(mask)
    for lbl, rnd_lbl in zip(labels, random_labels):
        lbl_coords = np.argwhere(mask.flatten() == lbl)
        np.put(randomized_mask, lbl_coords, rnd_lbl)
    return randomized_mask


def plot_cell_intensities(cell_labels, tracked_cells, images, T=10,
                          normalize=True, hpf=False, lpf=False):
    """ plot relative mean intensities for specified cells
    
    Parameters
    ---------------
    cell_labels: list of ints
        list of of the labels of the cells which intensities should be plotted
    tracked_cells: 3D array
        tracked segmentation mask
    images: 3D array
        the images from which the intensities should be taken
    T : int or float
        period of imaging (time between images) in seconds
    normalize: bool (optional)
        if True, the intensities are min-max normalized to a range of 0 to 1
    hpf: bool (optional)
        whether to do high-pass filtering or not
    lpf: bool (optional)
        whether to do low-pass filtering or not
    Returns
    ---------------
    None
    """
    image_count = len(images)
    x = np.linspace(0,0+(T*image_count), image_count, endpoint=False)

    plt.figure()

    for c in cell_labels:
        y = get_cell_intensities(c, tracked_cells, images, normalize, hpf, lpf)
        plt.plot(x, y, label="Cell " + str(c))

    plt.ylabel("Relative intensity")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.show() 
    return None


def get_correlation_matrix(tracked_cells, images, cell_labels=None,
                           plot=True):
    """ calculate and plot correlation matrix
    
    Uses cross correlation to generate a matrix of correlations between the
    cells given by cell_labels.
    Parameters
    ---------------
    tracked_cells: 3D array
        previously tracked segmentation mask
    images: 3D array
        images from which the tracked_cells segmentation mask was generated
    cell_labels: list (optional)
        the cells for which the correlation should be calculated. Default all cells.
    plot: bool (optional)
        whether to plot the resulting matrix or not.
    Returns
    ---------------
    corrcoefs: 2D array
        the correlation coefficients as a matrix
    """
    intensities = []
    if not cell_labels:
        cell_labels = range(1, np.max(tracked_cells)+1)
    corrcoefs = np.zeros((len(cell_labels), len(cell_labels)))
    
    for cell_label, j in zip(cell_labels, range(len(cell_labels))):
        intensities.append(get_cell_intensities(cell_label, tracked_cells,
                                                images))
        for i in range(j+1):
            corrcoefs[i, j] = np.correlate(intensities[i], intensities[j])

    if plot:
        if len(cell_labels) > 100:
            idx = np.round(np.linspace(0, len(list(cell_labels)) - 1, 10,
                                       dtype='int'))
            tick_labels = [str(c) for c in idx]
        else:
            idx = cell_labels
            tick_labels = [str(c) for c in cell_labels]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(corrcoefs)
        plt.title("Correlation coefficients")
        fig.colorbar(cax, label="Correlation", orientation="vertical")#, fraction=0.046, pad=0.04)
        ax.set_xticks(idx)
        ax.set_yticks(idx)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        plt.show()

    return corrcoefs


def get_common_cells(tracked_masks, occurrence=80):
    """ get the cells that appear in at least [occurrence] percent of images
    
    Parameters
    ---------------
    tracked_masks: 3D array
        previously tracked segmentation masks
    occurrence: int (optional)
        the smallest percent of images that the cell is allowed to appear
        in to still be taken into account
    Returns
    ---------------
    commons: list
        a list of labels of the cells that fullfill the given requirements of
        how many images they need to appear in
    counts: list
        a list of the number of times a specific cell appears in the images,
        in the same order as 'commons'
    """

    cell_labels = get_cell_labels(tracked_masks)
    commons = []
    counts = []
    limit = (occurrence/100.)*len(tracked_masks)
    cell_labels_flat = np.array([i for image in cell_labels for i in image])

    for i in np.unique(cell_labels_flat):
        count = np.count_nonzero(cell_labels_flat==i)
        if count>=limit:
            commons.append(i)
            counts.append(count)
    return np.array(commons), np.array(counts)


def plot_xcorr_vs_distance(ref_cell, tracked_masks, images, perc_req = 100,
                           normalize=False, hpf=0.0005, lpf=0.017, plot=True,
                           maxdt=200, T=10, n=None, cutoff_peakheight=None, mode="max", speed=20):
    """ plot cross correlation as a function of distance from a reference cell

    The reference cell has to appear in the first image.
    Parameters
    ---------------
    ref_cell: int
        the label of the cell that the distances should be compared with
    tracked_masks: 3D array
        previously tracked segmentation masks
    images: 3D array
        images from which the tracked_cells segmentation mask was generated
    perc_req: int
        requirement on how many percent of the images the cells have to be in
        in order to be included in the calculations of cross correlation
    normalize: bool (optional)
        if True, the intensities are normalized so that the intensities of
        different cells match better
    hpf: bool (optional)
        whether to do high-pass filtering or not
    lpf: bool (optional)
        whether to do low-pass filtering or not
    plot: bool (optional)
        whether to plot the results or not
    mode:
        'max': take the maximal normalized cross-correlation for each time point
        'lin': linear increase of time difference for cross correlation.
    speed:
        pixels/s
    Returns
    ---------------
    dist_sort: list
        a list of all the distances between the reference cell and the other
        cells, sorted by distance
    xcorr_list:list
        normalized cross correlation between reference cell and the other
        cells, sorted by distance from reference cell.
    """
    cell_lbls, count = get_common_cells(tracked_masks, perc_req)
    coms, cell_lbls = get_centers_of_mass(tracked_masks[0], cell_lbls)
    com_ref = coms[cell_lbls==ref_cell]

    dists = np.linalg.norm(np.array(coms)-np.array(com_ref), axis=1)

    dists_sort, cell_lbls_sort = (list(t) for t in 
                                  zip(*sorted(zip(dists, cell_lbls))))
    xcorr_list = []
    dtmax_list = []
    ref_cell_intensity = get_cell_intensities(ref_cell, tracked_masks, images, T=T,
                                              normalize=normalize, hpf=hpf, lpf=lpf, n=n, cutoff_peakheight=cutoff_peakheight)

    for i, cell in enumerate(cell_lbls_sort):
        intensity = get_cell_intensities(cell, tracked_masks, images, T, normalize, hpf, lpf, n, cutoff_peakheight)
        xcorr = get_cc(ref_cell_intensity, intensity, maxdt)
        if mode=='max':
            xcorr_list.append(np.max(xcorr))
            dtmax_list.append((np.argmax(xcorr)-maxdt)*T)
        if mode=='lin':
            dt = round(dists_sort[i]/speed)
            dtmax_list.append(dt)
            xcorr_list.append(xcorr[round(dt/T+maxdt)])


    def plot_data(data_list, ylabel, title):
        plt.figure()
        plt.scatter(dists_sort, data_list, marker='.',color="blue")
        plt.xlabel("Distance from reference cell [pixels]")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()

    if plot:
        plot_data(xcorr_list,"Maximal cross-correlation", "Cross-correlation as a function of distance from cell " + str(ref_cell))
        plot_data(dtmax_list,"dt at maximal cross-correlation", "dt at maximal cross-correlation as a function of distance from cell " + str(ref_cell))

    return dists_sort, xcorr_list


def plot_xcorr_vs_distance_new(ref_cell, intensities, reference_mask):
    """ plot cross correlation as a function of distance from a reference cell

    The reference cell has to appear in the first image.
    Parameters
    ---------------
    ref_cell: int
        the label of the cell that the distances should be compared with
    tracked_masks: 3D array
        previously tracked segmentation masks
    images: 3D array
        images from which the tracked_cells segmentation mask was generated
    perc_req: int
        requirement on how many percent of the images the cells have to be in
        in order to be included in the calculations of cross correlation
    normalize: bool (optional)
        if True, the intensities are normalized so that the intensities of
        different cells match better
    hpf: bool (optional)
        whether to do high-pass filtering or not
    lpf: bool (optional)
        whether to do low-pass filtering or not
    plot: bool (optional)
        whether to plot the results or not
    mode:
        'max': take the maximal normalized cross-correlation for each time point
        'lin': linear increase of time difference for cross correlation.
    speed:
        pixels/s
    Returns
    ---------------
    dist_sort: list
        a list of all the distances between the reference cell and the other
        cells, sorted by distance
    xcorr_list:list
        normalized cross correlation between reference cell and the other
        cells, sorted by distance from reference cell.
    """
    cell_lbls, count = get_common_cells(tracked_masks, perc_req)
    coms, cell_lbls = get_centers_of_mass(tracked_masks[0], cell_lbls)
    com_ref = coms[cell_lbls==ref_cell]

    dists = np.linalg.norm(np.array(coms)-np.array(com_ref), axis=1)

    dists_sort, cell_lbls_sort = (list(t) for t in 
                                  zip(*sorted(zip(dists, cell_lbls))))
    xcorr_list = []
    dtmax_list = []
    ref_cell_intensity = get_cell_intensities(ref_cell, tracked_masks, images, T=T,
                                              normalize=normalize, hpf=hpf, lpf=lpf, n=n, cutoff_peakheight=cutoff_peakheight)

    for i, cell in enumerate(cell_lbls_sort):
        intensity = get_cell_intensities(cell, tracked_masks, images, T, normalize, hpf, lpf, n, cutoff_peakheight)
        xcorr = get_cc(ref_cell_intensity, intensity, maxdt)
        if mode=='max':
            xcorr_list.append(np.max(xcorr))
            dtmax_list.append((np.argmax(xcorr)-maxdt)*T)
        if mode=='lin':
            dt = round(dists_sort[i]/speed)
            dtmax_list.append(dt)
            xcorr_list.append(xcorr[round(dt/T+maxdt)])


    def plot_data(data_list, ylabel, title):
        plt.figure()
        plt.scatter(dists_sort, data_list, marker='.',color="blue")
        plt.xlabel("Distance from reference cell [pixels]")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()

    if plot:
        plot_data(xcorr_list,"Maximal cross-correlation", "Cross-correlation as a function of distance from cell " + str(ref_cell))
        plot_data(dtmax_list,"dt at maximal cross-correlation", "dt at maximal cross-correlation as a function of distance from cell " + str(ref_cell))

    return dists_sort, xcorr_list


def plot_xcorr_map(tracked_masks, images, mode='single', ref_cell=1, occurrence=97, ref_image_idx=0,
                   normalize=False, hpf=False, lpf=False, show_labels=False):
    """ plot a map of the cross correlation per cell
    
    Parameters
    ---------------
    tracked_masks: 3D array
        previously tracked segmentation masks

    images: 3D array
        images from which tracked_masks was generated

    mode: {'single', 'total_sum', 'nearest_neighbor'} (optional)
        'single':
            By default, mode is 'single'. This returns a map with the
            cross correlations between a reference cell (ref_cell) and every
            other cell. With this mode, ref_cell must be given.
        'total_sum':
            Mode 'total_sum' returns a map with the summed cross correlation
            with all cells for each cell.
        'nearest_neighbor':
            Mode 'nearest_neighbor' returns a map where each cell gets a value
            corresponding to the weighted sum of the cross correlations with
            the cells sharing a border with it.

    ref_cell: int (optional)
        index of the reference cell when using mode 'single'.

    occurrence: int (optional)
        the percentage of images a cell has to appear in in order to be used
        in the cross correlation.

    ref_image_idx: int (optional)
        index of the image from tracked_masks to be used for the map

    normalize: bool (optional)
        if True, the intensities are normalized so that the intensities of
        different cells match better

    hpf: bool (optional)
        whether to do high-pass filtering or not

    lpf: bool (optional)
        whether to do low-pass filtering or not

    show_labels: bool (optional)
        whether to show the labels of the cells or not

    Returns
    ---------------
    matrix: 2D array
        the first mask from tracked_masks, but the values of the cells
        have been exchanged for the corresponding correlation coefficients.
    """

    ref_image = tracked_masks[ref_image_idx]
    if mode == "nearest_neighbor":
        ref_image = np.pad(ref_image, 1, 'constant', constant_values=0)
    cell_labels, xxx = get_common_cells(tracked_masks, occurrence)
    matrix = np.zeros_like(ref_image, dtype=float)
    correlation_mean = 0
    correlation_variance = 0
    intensities = []
    correlations_list = []
    for lbl in cell_labels:
        intensities.append(get_cell_intensities(lbl, tracked_masks, images,
                                                normalize, hpf, lpf))

    if mode=="single":
        ref_cell_intensity = get_cell_intensities(ref_cell, tracked_masks,
                                                  images, normalize, hpf, lpf)
        for lbl in cell_labels:
            intensity = intensities[cell_labels==lbl]
            xcorr = np.corrcoef(ref_cell_intensity, intensity)[0,1]
            matrix[ref_image==lbl] = xcorr
            correlations_list.append(xcorr)
        correlation_mean = np.mean(np.array(correlations_list))
        correlation_variance = np.var(np.array(correlations_list))

    if mode=="nearest_neighbor":        # CONTROL SOMEHOW IF THIS ALL ACTUALLY WORKS!
        xcorr_matrix = np.corrcoef(intensities)
        conv_krnl = np.ones([3,3])
        start_cell_labels = cell_labels
        for lbl in start_cell_labels:
            if not any(ref_image.flatten()==lbl):
                cell_labels = np.delete(cell_labels, cell_labels==lbl)
                continue
            mask = convolve2d(ref_image==lbl, conv_krnl, mode="same")
            all_border_values = ref_image[mask!=0]
            all_border_values = all_border_values[all_border_values!=lbl]
            # We only want to take the elements in cell_labels into account
            border_values = all_border_values[np.in1d(all_border_values,
                                                      start_cell_labels)]
            bv_count = len(border_values)
            correlations = 0
            for i in np.unique(border_values):
                weight = float(np.count_nonzero(border_values==i))/\
                    float(bv_count)
                weighted_corr = weight*float(xcorr_matrix[start_cell_labels==lbl,
                                                          start_cell_labels==i])
                correlations += weighted_corr
            matrix[ref_image==lbl] = correlations
            if (len(all_border_values)-bv_count) < 0.3*len(all_border_values):
                correlations_list.append(correlations)
        correlation_mean = np.mean(np.array(correlations_list))
        correlation_variance = np.var(np.array(correlations_list))

    if mode=="total_sum":
        xcorr = np.corrcoef(intensities)
        corr_elems_count = len(cell_labels)-1 # We don't correlate with self
        for i, lbl in enumerate(cell_labels):
            # Get average correlation with all cells except self
            mean_corr = (float(np.sum(xcorr[i])-1))/(float(corr_elems_count))
            matrix[ref_image==lbl] = mean_corr
            correlations_list.append(mean_corr)
        correlations_list = np.array(correlations_list)
        correlation_mean = np.mean(correlations_list)
        correlation_variance = np.var(correlations_list)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    masked_matrix = np.ma.masked_where(matrix == 0, matrix)
    cmap = cm.hot   # This doesn't seem to do anything
    cmap.set_bad(color='white')
    cax = ax.imshow(masked_matrix)
    if mode=='total_sum':
        plt.title("Summed correlation with all cells")
        fig.colorbar(cax, label="Sum of correlation coefficients")
    if mode=='single':
        plt.title("Correlation to cell no. " + str(ref_cell))
        fig.colorbar(cax, label="Correlation coefficient")
    if mode=='nearest_neighbor':
        plt.title("Correlation to nearest neighboring cells")
        fig.colorbar(cax, label="Correlation coefficient")

    # # Add annotation to all the cells in the image
    if show_labels:
        coms, lbls = get_centers_of_mass(ref_image)
        coms_commons = []
        for lbl in cell_labels:
            coms_commons.append(list(coms[lbls==lbl]))
        y = [com[0][0] for com in coms_commons]
        x = [com[0][1] for com in coms_commons]
        plt.scatter(x,y, marker='.', color="red")
        for i, lbl in enumerate(cell_labels):
            ax.annotate(lbl, (x[i], y[i]))

    plt.show()
    return matrix, correlation_mean, correlation_variance


def get_cc(intensities1, intensities2, max_dt):
    """Returns the list of correlations and corresponding time shift.
    The time shift is in respect to intensities1."""
    def cc(f1,f2):
        return np.sum((f1-np.mean(f1))*(f2-np.mean(f2)))
    
    def ncc(f1, f2):
        return cc(f1,f2)/np.sqrt(cc(f1,f1)*cc(f2,f2))

    correlation_list = []
    dt_list = []
    for dt in range(max_dt,0,-1):
        dt_list.append(-dt)
        correlation_list.append(ncc(intensities1[:-dt],intensities2[dt:]))
    correlation_list.append(ncc(intensities1,intensities2))
    dt_list.append(0)
    for dt in range(1, max_dt+1):
        dt_list.append(dt)
        correlation_list.append(ncc(intensities1[dt:],intensities2[:-dt]))

    return correlation_list, dt_list


def plot_xcorr_map_new(tracked_masks, images, mode="single", ref_cell_lbl=1,
                       occurrence=97, ref_image_idx=0, max_dt=200,
                       normalize=False, hpf=0.0005, lpf=0.017, T=10,
                       n=None, cutoff_peakheight=None, plot=True,
                       show_labels=False):
    
    def plot_data(data,show_labels,cbar_label,ref_image,cell_labels):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        data_masked = np.ma.masked_where(data==0, data)
        cmap = cm.hot   # This doesn't seem to do anything
        cmap.set_bad(color='white')
        cax = ax.imshow(data_masked)
        fig.colorbar(cax, label=cbar_label)
        if show_labels:
            coms, lbls = get_centers_of_mass(ref_image, cell_labels)
            y,x = zip(*coms)
            plt.scatter(x,y, marker='.', color="red")
            for i, lbl in enumerate(lbls):
                ax.annotate(lbl, (x[i], y[i]))
        plt.show()

    cell_labels, counts = get_common_cells(tracked_masks, occurrence)
    intsies_lbl_dict = {}
    for lbl in cell_labels:
        intsy = get_cell_intensities(lbl, tracked_masks, images, T, normalize, hpf, lpf, n, cutoff_peakheight)
        intsies_lbl_dict[lbl] = intsy
    
    if mode == "single":
        cc_dict = {}
        dtmax_dict = {}

        for lbl in cell_labels:
            cc, dt = get_cc(intsies_lbl_dict[lbl], intsies_lbl_dict[ref_cell_lbl], max_dt)
            cc_dict[lbl] = np.max(cc)
            dtmax_dict[lbl] = dt[np.argmax(cc)]

        if plot:
            ref_image = tracked_masks[ref_image_idx]
            cc_plot = np.zeros_like(ref_image, 'float')
            dtmax_plot = np.zeros_like(ref_image, 'float')

            for lbl in cell_labels:
                cc_plot[ref_image==lbl] = cc_dict[lbl]
                dtmax_plot[ref_image==lbl] = dtmax_dict[lbl]

            plot_data(cc_plot,show_labels,"Maximal cross-correlation",ref_image,cell_labels)
            plot_data(dtmax_plot,show_labels,"dt at maximal cross-correlation",ref_image,cell_labels)

        return cc_dict, dtmax_dict, cc_plot, dtmax_plot
        

# def main():
#     path0815 = "//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Master2023/Recordings/2023-08-15/"
#     name_ctl = "ctl-ftrc2-dl12-btl15-diam325"
#     images_path_ctl0815 = path0815 + "15_08_ctl.tif"
#     masks_path_ctl0815 = path0815 + name_ctl + "_masks.tif"
#     images_ctl0815 = open_image_stack(images_path_ctl0815)
#     masks_ctl0815 = open_masks(masks_path_ctl0815)
#     common_cellsctl0815, counts = get_common_cells(masks_ctl0815, occurrence=50)
#     save_path = "//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Master2023/Recordings/Intensity-data/ctl"

#     ctl0815_dict = {}
#     for c in common_cellsctl0815:
#         y = get_cell_intensities(c, masks_ctl0815, images_ctl0815, normalize=False, hpf=None, lpf=None)
#         # Find the first image the cell appears in
#         im_id = np.where(masks_ctl0815)[0][0]
#         c_o_m = get_centers_of_mass(masks_ctl0815[im_id], c)
#         ctl0815_dict[str(c_o_m)] = y
#     df = pd.DataFrame(data=ctl0815_dict)
#     df.to_csv(save_path+"/ctl0815.csv")



# if __name__ == "__main__":
#     main()
