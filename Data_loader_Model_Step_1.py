import os
import numpy as np
import json
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def load_json_data(json_file):
    """
    Load nodal points data from a JSON file.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

def load_data(image_folder, mask_epi_folder, cont_epi_folder, seg_folder, json_file, target_size=(256, 256)):
    """
    Load images, masks, nodal points, contours, and segmentations.

    Args:
    - image_folder: Path to the folder with images.
    - mask_epi_folder: Path to the folder with masks.
    - cont_epi_folder: Path to the folder with contours.
    - seg_folder: Path to the folder with segmentations.
    - json_file: Path to the JSON file containing nodal points.
    - target_size: Target size for resizing images.

    Returns:
    - images, masks_epi, nodal_points, contours_epi, segmentations: Arrays of processed data.
    """
    images, masks_epi, nodal_points, contours_epi, segmentations = [], [], [], [], []
    nodal_points_data = load_json_data(json_file)

    for filename in sorted(os.listdir(image_folder)):
        img_path = os.path.join(image_folder, filename)
        mask_epi_path = os.path.join(mask_epi_folder, filename)
        cont_epi_path = os.path.join(cont_epi_folder, filename)
        seg_path = os.path.join(seg_folder, filename)

        if not os.path.exists(mask_epi_path):
            continue

        if filename not in nodal_points_data or not nodal_points_data[filename][0]:
            continue

        points = np.array(nodal_points_data[filename][0]) / np.array(target_size)

        img = img_to_array(load_img(img_path, target_size=target_size, color_mode='grayscale'))
        mask_epi = img_to_array(load_img(mask_epi_path, target_size=target_size, color_mode='grayscale'))

        cont_epi = (img_to_array(load_img(cont_epi_path, target_size=target_size, color_mode='grayscale'))
                    if os.path.exists(cont_epi_path) else np.zeros(target_size + (1,)))
        seg = img_to_array(load_img(seg_path, target_size=target_size, color_mode='grayscale'))

        images.append(img)
        masks_epi.append(mask_epi)
        nodal_points.append(points)
        contours_epi.append(cont_epi)
        segmentations.append(seg)

    return np.array(images), np.array(masks_epi), np.array(nodal_points), np.array(contours_epi), np.array(segmentations)
