import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import napari
from datetime import date
from scipy import ndimage
from tifffile import imwrite
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from matplotlib import cm, colors
from PIL import Image
from scipy.ndimage import center_of_mass

# General Utilities
def is_valid_com(com):
    return not np.isnan(com[0]) and not np.isnan(com[1]) and not np.array_equal(com, [0, 0])

def extract_unique_labels_from_pairs(cell_pairs):
    unique_labels = set()
    for pair in cell_pairs:
        unique_labels.update(pair)
    return list(unique_labels)

def rolling_window_average_with_padding(intensities, window_size):
    padded_intensities = np.pad(intensities, (window_size//2, window_size//2), mode='edge')
    filtered_intensities = np.convolve(padded_intensities, np.ones(window_size)/window_size, mode='same')
    return filtered_intensities[window_size//2: -window_size//2]

def apply_smoothing_to_normalized(intensities_normalized, window_size, smoothing_method='padding', poly_order=3):
    filtered_intensities = {}
    for cell_id, intensities in intensities_normalized.items():
        if smoothing_method == 'padding':
            filtered_intensities[cell_id] = rolling_window_average_with_padding(intensities, window_size)
        else:
            filtered_intensities[cell_id] = intensities
    return filtered_intensities

# Segmentation and Mask Utilities
def fill_in_missing_labels_tracking(Y, max_frame_gap=1):
    Y_filled = Y.copy()
    n_frames = Y.shape[0]
    unique_labels = np.unique(Y)
    unique_labels = unique_labels[unique_labels != 0]

    # Build trajectories for each label
    trajectories = {}
    for label in unique_labels:
        centroids = []
        for f in range(n_frames):
            mask = (Y[f] == label)
            if np.any(mask):
                com = center_of_mass(mask)
                centroids.append((f, com))
        trajectories[label] = centroids

    # Attempt to fill missing labels
    for label, traj in trajectories.items():
        present_frames = set(f for f, _ in traj)
        for f in range(n_frames):
            if f not in present_frames:
                # Find nearest frame with this label
                neighbor_frames = [f2 for f2, _ in traj if abs(f - f2) <= max_frame_gap]
                if not neighbor_frames:
                    continue
                # Choose closest in time
                closest_f = min(neighbor_frames, key=lambda nf: abs(nf - f))
                # Predict centroid
                predicted_com = [c for fr, c in traj if fr == closest_f][0]
                predicted_com_int = tuple(map(int, predicted_com))

                # Check if predicted location is free
                if Y_filled[f][predicted_com_int] == 0:
                    mask_neighbor = (Y[closest_f] == label)
                    Y_filled[f][mask_neighbor] = label
                elif Y_filled[f][predicted_com_int] != label:
                    # Conflict: Decide based on trajectory density
                    # Here we skip, but this could be improved
                    print(f"Conflict: label {label} in frame {f}")
    return Y_filled

def get_cell_labels(masks):
    if len(masks.shape) == 3:
        return [np.unique(mask[mask != 0]) for mask in masks]
    return np.unique(masks[masks != 0])

def get_centers_of_mass(masks, cell_labels=None):
    if cell_labels is None:
        cell_labels = get_cell_labels(masks)
    if len(masks.shape) == 3:
        return [np.array(ndimage.center_of_mass(mask, mask, labels)) for mask, labels in zip(masks, cell_labels)], cell_labels
    return np.array(ndimage.center_of_mass(masks, masks, cell_labels)), cell_labels

def assign_random_cell_labels(mask):
    labels = np.unique(mask[mask != 0])
    np.random.shuffle(labels)
    randomized_mask = np.zeros_like(mask)
    for lbl, rnd_lbl in zip(np.unique(mask[mask != 0]), labels):
        randomized_mask[mask == lbl] = rnd_lbl
    return randomized_mask

def _save_masks(masks, name=None, savedir=None):
    name = name or str(date.today())
    if savedir:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        output_file = os.path.join(savedir, name)
    imwrite(output_file, masks, imagej=True, metadata={'axes': 'TYX'})
    print(f"Final mask saved to: {output_file}")

def get_tracked_masks(masks, dist_limit=None, backtrack_limit=None, random_labels=False, save=False, name=None, savedir=None):
    tracked_masks = np.zeros_like(masks)
    COMs, roi_labels = get_centers_of_mass(masks)
    
    if random_labels:
        tracked_masks[0] = assign_random_cell_labels(masks[0])
    else:
        tracked_masks[0] = masks[0]
    
    used_labels = set(np.unique(tracked_masks[0]))
    
    for imnr in tqdm(range(len(masks)), desc="Tracking frames", unit=" frames"):
        if imnr == 0:
            continue
        new_cells = 0
        ROI_labels_imnr = roi_labels[imnr]
        if len(COMs[imnr]) == 0:
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
                    break
            if ref_im_idx == -10:
                new_cells += 1
                cell_label = np.max(tracked_masks[:imnr]) + new_cells
                while cell_label in used_labels:
                    cell_label += 1
            used_labels.add(cell_label)
            tracked_masks[imnr][masks[imnr] == COM_label] = cell_label

    if save:
        _save_masks(tracked_masks, name=name, savedir=savedir)
    
    return tracked_masks

def get_common_cells(tracked_masks, occurrence=80):
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

# Cell Intensity Extraction
def get_cell_intensities(c, Y, X):
    intensities = np.full(X.shape[0], np.NaN)
    for t in range(X.shape[0]):
        mask = (Y[t] == c)
        if np.any(mask):
            intensities[t] = np.nanmean(X[t][mask])
    mean_intensity = np.nanmean(intensities)
    intensities[np.isnan(intensities)] = mean_intensity
    return intensities

def get_cell_intensities_circle(c, Y, X, radius):
    intensities = np.full(X.shape[0], np.NaN)
    com_mask = np.zeros(Y.shape, dtype=np.uint8)
    com_coords = [None] * X.shape[0]
    for t in range(X.shape[0]):
        mask = (Y[t] == c)
        if np.any(mask):
            com = ndimage.center_of_mass(mask)
            if not np.isnan(com[0]):
                com_coords[t] = com
                xx, yy = np.mgrid[:X.shape[1], :X.shape[2]]
                circle = (xx - com[0]) ** 2 + (yy - com[1]) ** 2
                com_mask[t] = circle < radius ** 2
                intensities[t] = np.nanmean(X[t][com_mask[t] == 1])
    mean_intensity = np.nanmean(intensities)
    intensities[np.isnan(intensities)] = mean_intensity
    com_mask[com_mask != 0] = c
    return com_mask, intensities, com_coords

# Center of Mass Calculation
def calculate_valid_center_of_mass(Y, target_cell, frame_idx, max_frame_gap=5):
    def calculate_center_of_mass_for_cell_at_frame(Y, target_cell, frame_idx):
        frame_mask = Y[frame_idx]
        mask = (frame_mask == target_cell)
        if np.any(mask):
            return ndimage.center_of_mass(mask)
        return np.nan, np.nan
    com = calculate_center_of_mass_for_cell_at_frame(Y, target_cell, frame_idx)
    if is_valid_com(com):
        return com
    for gap in range(1, max_frame_gap + 1):
        for new_frame in [frame_idx + gap, frame_idx - gap]:
            if 0 <= new_frame < Y.shape[0]:
                com = calculate_center_of_mass_for_cell_at_frame(Y, target_cell, new_frame)
                if is_valid_com(com):
                    return com
    return [0, 0]

def calculate_and_save_cell_coms_at_frame(Y, unique_cell_labels, frame_idx, max_frame_gap=5):
    """Calculate and store the center of mass for each unique cell at a specific frame, using neighboring frames if needed."""
    cell_coms = {}
    for target_cell in unique_cell_labels:
        com = calculate_valid_center_of_mass(Y, target_cell, frame_idx, max_frame_gap)
        
        # Ensure the COM is stored as a tuple (x, y)
        if is_valid_com(com):
            cell_coms[target_cell] = tuple(com)
    
    return cell_coms

# Correlation Calculation
def calculate_cross_correlation(cell1, cell2, max_lag, filtered_intensities, min_overlap):
    intensities1 = filtered_intensities[cell1]
    intensities2 = filtered_intensities[cell2]
    if np.any(np.isnan(intensities1)) or np.any(np.isnan(intensities2)):
        return {
            'label1': cell1,
            'label2': cell2,
            'correlations': np.nan,
            'time_lags': np.arange(-max_lag, max_lag + 1)
        }
    correlation_data = {
        'label1': cell1,
        'label2': cell2,
        'correlations': np.full(2 * max_lag + 1, np.nan),
        'time_lags': np.arange(-max_lag, max_lag + 1)
    }
    for lag in correlation_data['time_lags']:
        if lag == 0:
            num_pairs = len(intensities1)
            if num_pairs >= min_overlap:
                correlation_data['correlations'][max_lag] = np.mean(intensities1 * intensities2)
        elif lag > 0:
            num_pairs = len(intensities1) - lag
            if num_pairs >= min_overlap:
                correlation_data['correlations'][lag + max_lag] = np.sum(intensities1[:-lag] * intensities2[lag:]) / num_pairs
        else:
            num_pairs = len(intensities1) + lag
            if num_pairs >= min_overlap:
                correlation_data['correlations'][lag + max_lag] = np.sum(intensities1[-lag:] * intensities2[:lag]) / num_pairs
    return correlation_data

def filter_cell_pairs_by_distance(cell1, cell2, com_dict, threshold):
    coms1 = com_dict[cell1].get('com_coords', None)
    coms2 = com_dict[cell2].get('com_coords', None)
    if coms1 is None or coms2 is None:
        return False
    distances = [
        euclidean(c1, c2) for c1, c2 in zip(coms1, coms2) if c1 is not None and c2 is not None
    ]
    return bool(distances) and np.mean(distances) <= threshold

def build_null_distribution(filtered_pairs, filtered_intensities, max_lag, num_permutations=1000, min_overlap=10):
    null_distribution = []
    for _ in tqdm(range(num_permutations), desc="Computing Null Distribution"):
        shuffled_intensities = {
            cell: np.random.permutation(intensity) for cell, intensity in filtered_intensities.items()
        }
        for cell1, cell2 in filtered_pairs:
            shuffled_corr = calculate_cross_correlation(cell1, cell2, max_lag, shuffled_intensities, min_overlap)
            if shuffled_corr is not None:
                max_corr = np.nanmax(shuffled_corr['correlations'])
                null_distribution.append(max_corr)
    return np.array(null_distribution)

def get_significant_correlation_threshold(sorted_correlations, null_distribution, p_value_threshold=0.05):
    """
    Find significant correlation threshold from null distribution and print it.
    """
    for _, correlation in sorted_correlations:
        p_value = np.mean(null_distribution >= correlation)
        if p_value < p_value_threshold:
            print(f"Significant threshold: {correlation} (p-value: {p_value})")
            return correlation, p_value
    print("No significant threshold found with p-value below the threshold.")
    return None, None

# Visualization Utilities
def save_raw_frames_as_pdfs(file_name, raw_file, save_dir):
    with tifffile.TiffFile(raw_file) as tif:
        frames = tif.asarray()

    total_frames = frames.shape[0]
    os.makedirs(save_dir, exist_ok=True)

    # Dynamically select 3 indices: first, middle, last
    frame_indices = [0, (total_frames - 1) // 2, total_frames - 1]
    suffixes = ['t0', 't1', 't2']

    saved_files = []
    for idx, suffix in zip(frame_indices, suffixes):
        frame = frames[idx]

        plt.figure(figsize=(6, 6))
        plt.imshow(frame, cmap='gray')
        plt.axis('off')

        output_path = os.path.join(save_dir, f'{file_name}_{suffix}.pdf')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        saved_files.append(output_path)

    print(f"Saved frames for {file_name}:")
    for f in saved_files:
        print(f" - {f}")

def save_napari_export(viewer, filename, scale_factor=1.0):
    # Export the figure from the viewer (not a layer)
    image = viewer.export_figure(path=filename, scale_factor=scale_factor, flash=False)
    print(f"Screenshot saved as {filename}")

def plot_network_and_cell_coms_in_napari(
    X, correlations_above_threshold, filtered_cell_comms, non_network_cell_coms_at_frame,
    frame_idx, export_path=None, plot_file_path=None
):
    viewer = napari.Viewer()
    viewer.add_image(X[frame_idx], name=f'Frame {frame_idx}', blending='additive')
    network_cells = set()
    edges = []
    edge_colors = []
    for (pair, corr_value) in correlations_above_threshold:
        cell1, cell2 = pair
        if cell1 in filtered_cell_comms and cell2 in filtered_cell_comms:
            com1, com2 = filtered_cell_comms[cell1], filtered_cell_comms[cell2]
            edges.append([com1, com2])
            network_cells.update([cell1, cell2])
            edge_colors.append(float(corr_value))
    if edge_colors:
        norm = colors.Normalize(vmin=min(edge_colors), vmax=max(edge_colors))
        colormap = cm.viridis
        edge_colors_rgb = [colormap(norm(corr))[:3] for corr in edge_colors]
    else:
        edge_colors_rgb = []
    edge_lines = [
        [[com1[0], com1[1]], [com2[0], com2[1]]]
        for com1, com2 in edges
        if not np.any(np.isnan(com1)) and not np.any(np.isnan(com2))
    ]
    if edge_lines:
        viewer.add_shapes(edge_lines, shape_type='line', edge_color=edge_colors_rgb, edge_width=1, name='Cell Network')
    if filtered_cell_comms:
        points = [filtered_cell_comms[cell] for cell in network_cells if cell in filtered_cell_comms]
        labels = [str(cell) for cell in network_cells if cell in filtered_cell_comms]
        if points:
            layer = viewer.add_points(points, size=5, face_color='grey', name='Filtered Cell COMs', text=labels)
            layer.text_color = 'white'
    non_network_coords = [(com[0], com[1]) for com in non_network_cell_coms_at_frame.values()]
    non_network_labels = [str(cell) for cell in non_network_cell_coms_at_frame.keys()]
    if non_network_coords:
        layer = viewer.add_points(non_network_coords, size=5, face_color='red', name='Non-network Cell COMs', text=non_network_labels)
        layer.text_color = 'white'
    if export_path and plot_file_path:
        screenshot = viewer.screenshot()
        image = Image.fromarray(screenshot)
        bbox = image.convert('L').getbbox()
        cropped = image.crop(bbox)
        cropped.save(plot_file_path, format='PNG')
    napari.run()
    if export_path and plot_file_path:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(plt.imread(plot_file_path))
        ax.set_xticks([])
        ax.set_yticks([])
        if edge_colors:
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)
        plt.savefig(plot_file_path)
        plt.show()
        plt.close(fig)

def shift_intensity_curve(intensities, time_lag):
    num_frames = len(intensities)
    shifted_intensity = np.full(num_frames, np.nan)
    if time_lag > 0:
        shifted_intensity[time_lag:] = intensities[:-time_lag]
    elif time_lag < 0:
        shifted_intensity[:time_lag] = intensities[-time_lag:]
    else:
        shifted_intensity = intensities.copy()
    return shifted_intensity

def plot_correlation_curve(correlation_data, frame_rate, cell1, cell2, max_lag):
    time_lags_in_seconds = correlation_data['time_lags'] * (1 / frame_rate)
    plt.figure(figsize=(12, 6))
    plt.plot(time_lags_in_seconds, correlation_data['correlations'],
             label=f'Correlation for Cell {cell1} vs Cell {cell2}', color='green')
    plt.title(f'Cross-Correlation Curve for Cells {cell1} and {cell2}')
    plt.xlabel('Time Lag (seconds)')
    plt.ylabel('Cross-Correlation')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_intensity_curves(cell1, cell2, correlation_data, frame_rate, intensities_filtered):
    intensities1 = intensities_filtered[cell1]
    intensities2 = intensities_filtered[cell2]
    max_corr_index = np.nanargmax(correlation_data['correlations'])
    max_time_lag = correlation_data['time_lags'][max_corr_index]
    shifted_intensities2 = shift_intensity_curve(intensities2, max_time_lag)

    if max_time_lag > 0:
        start_idx, end_idx = max_time_lag, len(intensities1)
    elif max_time_lag < 0:
        start_idx, end_idx = 0, len(intensities1) + max_time_lag
    else:
        start_idx, end_idx = 0, len(intensities1)

    valid_intensities1 = intensities1[start_idx:end_idx]
    valid_intensities2 = shifted_intensities2[start_idx:end_idx]
    valid_time_axis = np.arange(start_idx, end_idx) / frame_rate / 60

    plt.figure(figsize=(12, 6))
    plt.plot(valid_time_axis, valid_intensities1, label=f'Cell {cell1} Intensity', color='blue')
    plt.plot(valid_time_axis, valid_intensities2, label=f'Cell {cell2} Intensity (shifted)', color='orange')
    plt.title(f'Normalized Intensities for Cells {cell1} and {cell2} (Time Lag = {max_time_lag} frames)')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Normalized Intensity')
    plt.legend()
    plt.grid(True)
    plt.show()

# Table Generation
def generate_latex_table(top_pairs, bottom_pairs, title, label, file_name, latex_path, min_correlations_filtered):
    """
    Generate and save a LaTeX table combining Top 10 and Bottom 10 correlations with consistent spacing.
    """
    latex_table = rf"""\begin{{table}}[H]
\centering
\caption{{{f"{file_name.replace('_', ' ').replace('CON', 'Control').replace('OUA', 'Ouabain-treated').replace('cluster', 'Cluster ')} - {title}"}}}
    \begin{{tabularx}}{{\textwidth}}{{CCCCCCCC}}
    \toprule[0.5pt]\toprule[0.5pt]\toprule[0.5pt]
    \multicolumn{{4}}{{c}}{{\textit{{Top 10}}}} & \multicolumn{{4}}{{c}}{{\textit{{Bottom 10}}}} \\
    \cmidrule(lr){{1-4}} \cmidrule(lr){{5-8}}
    \multicolumn{{1}}{{C}}{{\itshape $C_1$}} & 
    \multicolumn{{1}}{{C}}{{\itshape $C_2$}} & 
    \multicolumn{{1}}{{C}}{{\itshape \makecell{{Max\\Corr}}}} & 
    \multicolumn{{1}}{{C}}{{\itshape \makecell{{Min\\Corr}}}} & 
    \multicolumn{{1}}{{C}}{{\itshape $C_1$}} & 
    \multicolumn{{1}}{{C}}{{\itshape $C_2$}} & 
    \multicolumn{{1}}{{C}}{{\itshape \makecell{{Max\\Corr}}}} & 
    \multicolumn{{1}}{{C}}{{\itshape \makecell{{Min\\Corr}}}} \\
    \midrule[0.5pt]\midrule[0.5pt]
"""
    max_len = max(len(top_pairs), len(bottom_pairs))
    for idx in range(max_len):
        if idx < len(top_pairs):
            cell1_t, cell2_t = top_pairs[idx][0]
            max_t = top_pairs[idx][1]
            min_t = next(min_corr for (c1, c2), min_corr in min_correlations_filtered if c1 == cell1_t and c2 == cell2_t)
            row_top = f"        \\multicolumn{{1}}{{T}}{{{cell1_t}}} &\n        \\multicolumn{{1}}{{T}}{{{cell2_t}}} &\n        \\multicolumn{{1}}{{T}}{{{max_t:.4f}}} &\n        \\multicolumn{{1}}{{T}}{{{min_t:.4f}}}"
        else:
            row_top = "        \\multicolumn{1}{T}{} &\n        \\multicolumn{1}{T}{} &\n        \\multicolumn{1}{T}{} &\n        \\multicolumn{1}{T}{}"

        if idx < len(bottom_pairs):
            cell1_b, cell2_b = bottom_pairs[idx][0]
            max_b = bottom_pairs[idx][1]
            min_b = next(min_corr for (c1, c2), min_corr in min_correlations_filtered if c1 == cell1_b and c2 == cell2_b)
            row_bottom = f"        \\multicolumn{{1}}{{T}}{{{cell1_b}}} &\n        \\multicolumn{{1}}{{T}}{{{cell2_b}}} &\n        \\multicolumn{{1}}{{T}}{{{max_b:.4f}}} &\n        \\multicolumn{{1}}{{T}}{{{min_b:.4f}}}"
        else:
            row_bottom = "        \\multicolumn{1}{T}{} &\n        \\multicolumn{1}{T}{} &\n        \\multicolumn{1}{T}{} &\n        \\multicolumn{1}{T}{}"

        if idx < max_len - 1:
            latex_table += f"{row_top} &\n{row_bottom} \\\\\n        \\midrule\n"
        else:
            latex_table += f"{row_top} &\n{row_bottom} \\\\"

    latex_table += rf"""
    \midrule[0.5pt]\midrule[0.5pt]\midrule[0.5pt]
    \end{{tabularx}}
\label{{tab:{label}}}
\end{{table}}"""
    latex_file_name = f"{file_name}_correlations.tex"
    latex_file_path = os.path.join(latex_path, latex_file_name)
    with open(latex_file_path, "w") as file:
        file.write(latex_table)
    print(f"LaTeX table saved at: {latex_file_path}")