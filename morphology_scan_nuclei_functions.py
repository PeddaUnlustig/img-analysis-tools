import numpy as np
import cv2
import pandas as pd
import image_utils  # script from indra

def scan_slides(sampler):
    xs, ys, keys = sampler.get()
    image_results_list = list()
    for idx, nucleus in enumerate(xs):
        key = keys[idx]
        contour = nucleus["contour"]
        area = cv2.contourArea(np.array(contour))
        image_results_list.append(scan_nucleus(contour, key, idx, area))
    return image_results_list


def scan_nucleus(contour, file_key, idx, area):
    cnt = np.array(contour)
    perimeter = cv2.arcLength(cnt, True)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)

    x, y, w, h = cv2.boundingRect(cnt)

    rect = cv2.minAreaRect(cnt)
    min_xy, min_wh, min_angle = rect

    mom = cv2.moments(cnt)
    center_x = int(mom["m10"] / mom["m00"])
    center_y = int(mom["m01"] / mom["m00"])
    center_x -= x
    center_y -= y

    return [
        file_key,
        idx,
        center_x,
        center_y,
        w,
        h,
        min_wh[0],
        min_wh[1],
        perimeter,
        area,
        hull_perimeter,
        hull_area,
    ]


#######################################################################################
##########äääää##### Additional code for UNET Morph analysis ##########################


def scan_unet_tuple(data: tuple):
    assert len(data[0]) == len(data[1])

    image_results_list = list()
    for i, bitmap in enumerate(data[1]):
        # contour = get_boundary_points(bitmap)
        contours, bin_img = find_contours(bitmap, 1)
        if not contours:  # If no contours are found, skip or handle accordingly
            print(f"No contours found for image {i}")
            image_results_list.append(None)  # Or handle as needed
            continue

        contour = max(contours, key=cv2.contourArea)
        contour = contour.astype(np.float32)

        area = cv2.contourArea(np.array(contour))
        image_results_list.append(scan_nucleus(contour, data[0][i], i, area))

    morph_df = pd.DataFrame(
        image_results_list,
        columns=[
            "file_key",
            "idx",
            "center_x",
            "center_y",
            "w",
            "h",
            "min_wh[0]",
            "min_wh[1]",
            "perimeter",
            "area",
            "hull_perimeter",
            "hull_area",
        ],
    )

    return morph_df


# Contour function from other script
def get_boundary_points(binary_array):
    # Get array dimensions
    rows, cols = binary_array.shape

    # Create an output array to store boundary points
    boundary = np.zeros_like(binary_array)

    # Pad the array with zeros to handle edges easily
    padded = np.pad(binary_array, pad_width=1, mode="constant", constant_values=0)

    # Check each point that is 1 in the original array
    for i in range(rows):
        for j in range(cols):
            if binary_array[i, j] == 1:
                if (
                    padded[i + 1 - 1, j + 1] == 0  # up
                    or padded[i + 1 + 1, j + 1] == 0  # down
                    or padded[i + 1, j + 1 - 1] == 0  # left
                    or padded[i + 1, j + 1 + 1] == 0
                ):  # right
                    boundary[i, j] = 1

    # Get coordinates of boundary points
    boundary_coords = np.where(boundary == 1)
    return list(zip(boundary_coords[0], boundary_coords[1]))


def find_contours(img, threshold):
    gray = img
    blur = cv2.GaussianBlur(src=gray, ksize=(3, 3), sigmaX=0)
    (t, binary) = cv2.threshold(
        src=blur, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY
    )
    bin_img = image_utils.convert_8b(binary)
    (cnts, hierarchy) = cv2.findContours(
        image=bin_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
    )
    return cnts, bin_img
