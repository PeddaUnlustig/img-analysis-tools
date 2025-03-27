from pathlib import Path
import pickle
import numpy as np
import os
from compressed_sampler import SampleManager
import prep_ktb_nusp_cellvit as pknc
import json
import math
from scipy.ndimage import label


def identify_holes(data: dict):
    arrays_with_holes_count = 0
    keys_list = list()

    for key, values in data.items():
        binary_image = values["bbox_bitmap"]
        inverted_image = 1 - binary_image

        # Step 2: Label connected regions of 0s
        labeled_array, num_features = label(inverted_image)

        # Step 3: Check for holes
        # A hole is a region of 0s that does not touch the border
        has_hole = False
        image_shape = binary_image.shape
        for i in range(1, num_features + 1):
            region = labeled_array == i
            # Check if the region touches the border
            touches_border = (
                np.any(region[0, :])  # Top border
                or np.any(region[-1, :])  # Bottom border
                or np.any(region[:, 0])  # Left border
                or np.any(region[:, -1])  # Right border
            )
            if not touches_border:
                has_hole = True
                break  # No need to check further; we found at least one hole

        # Step 4: Increment the counter if the array has at least one hole
        if has_hole:
            arrays_with_holes_count += 1
            keys_list.append(key)
    print("Total length: ", len(data))
    print(arrays_with_holes_count)

    return keys_list


def create_geojson_from_contour(data: dict, output_file: str, contour_column="contour"):
    nuclei_list = list()
    for nucleus_data in data.values():
        contour = nucleus_data[contour_column]
        sorted_contour = sort_points_clockwise(contour)
        sorted_contour.append(sorted_contour[0])
        nuclei_list.append([sorted_contour])

    geojson_placeholder = []
    template_multipolygon = {
        "type": "Feature",
        "id": "ID",
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": nuclei_list,
        },
        "properties": {
            "objectType": "annotation",
            "classification": {"name": "Tool", "color": [92, 20, 186]},
        },
    }
    geojson_placeholder.append(template_multipolygon)
    with open(output_file, "w") as f:
        json.dump(geojson_placeholder, f, indent=2)

    return True


def display_points(points):
    # Find the max dimensions to create grid
    max_x = max(x for x, y in points)
    max_y = max(y for x, y in points)
    grid = np.full((max_x + 1, max_y + 1), " . ", dtype=object)

    for i, (x, y) in enumerate(points):
        for j, row in enumerate(grid):
            if j == x:
                for k, col in enumerate(row):
                    if k == y:
                        row[k] = f"{i:2}"
    # Print grid
    for row in grid:  # Transpose to correct orientation
        print(" ".join(row))
    return None


def sort_points_clockwise(points):
    # Compute centroid to use as reference
    cx = sum(x for x, y in points) / len(points)
    cy = sum(y for x, y in points) / len(points)

    # Define sorting key based on polar angle
    def angle(p):
        x, y = p
        return math.atan2(y - cy, x - cx)

    # Sort points by angle in clockwise order (descending)
    points.sort(key=angle, reverse=True)
    # points = [tuple(x) for x in points]
    return points


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
                # Check 4-connected neighbors in padded array
                # (i+1, j+1) is the corresponding position in padded array
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


def dict_to_geojson_exact_outline(
    nuclei_dict, output_file, debug=False, color: int = 0
):
    """
    Convert a dictionary of nuclei data to a GeoJSON file with exact outline of 1s.

    Args:
        nuclei_dict (dict): Dictionary with structure {"nuclei_id": {"centroid": (x,y), "bbox_bitmap": np.array}}
        output_file (str): Path to save the GeoJSON file
        debug (bool): If True, print debug information
    """
    geojson = []
    colors = {
        0: [92, 20, 186],
        1: [255, 0, 0],
        2: [34, 221, 77],
        3: [35, 92, 236],
        4: [254, 255, 0],
        5: [255, 159, 68],
    }
    overall_coordinates = list()

    for nucleus_id, data in nuclei_dict.items():
        centroid = data["centroid"]
        bitmap = data["bbox_bitmap"].astype(float)

        # Normalize bitmap if needed
        if bitmap.max() > 1:
            bitmap = bitmap / 255

        # Get exact boundary points
        boundary_points = get_boundary_points(bitmap)

        if not boundary_points:
            if debug:
                print(f"No boundary found for {nucleus_id}")
            continue

        boundary_points = sort_points_clockwise(boundary_points)

        # Convert to absolute coordinates
        coordinates = []
        bitmap_height, bitmap_width = bitmap.shape
        for x, y in boundary_points:
            abs_x = x + (centroid[0] - bitmap_width / 2)
            abs_y = y + (centroid[1] - bitmap_height / 2)
            coordinates.append([float(abs_x), float(abs_y)])

        # Close the polygon
        coordinates.append(coordinates[0])
        if debug:
            print(f"Nucleus {nucleus_id}:")
            print(f"Bitmap shape: {bitmap.shape}")
            print(f"Centroid: {centroid}")
            print(f"Number of boundary points: {len(boundary_points)}")
            print(f"Bitmap sample:\n{bitmap}")

        dummy_list = list()
        dummy_list.append(coordinates)

        overall_coordinates.append(dummy_list)

    print(len(overall_coordinates))

    template_multipolygon = {
        "type": "Feature",
        "id": "ID",
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": overall_coordinates,
        },
        "properties": {
            "objectType": "annotation",
            "classification": {"name": "Tool", "color": colors[color]},
        },
    }

    geojson.append(template_multipolygon)

    with open(output_file, "w") as f:
        json.dump(geojson, f, indent=2)

    return True


def load_wsi_pickle(input_path, wsi_name):
    assert os.path.exists(input_path)
    with open(input_path, "rb") as file:
        sampler = pickle.load(file)

    # Select only the relevant data from that one WSI
    nucleus_dict = dict()
    for i, nucleus_name in enumerate(sampler[0]):
        if nucleus_name.split(" ")[0].split("/")[0] == wsi_name:
            x, y = pknc.parse_coord_old(nucleus_name)
            array = np.squeeze(sampler[1][i])

            if sampler[1][i].shape[2] != 1:
                print(sampler[1][i].shape)
                raise ValueError("Array is not in shape as expected: (x,y,1)")

            value_dict = {"centroid": (x, y), "bbox_bitmap": array}
            nucleus_dict.update({sampler[0][i]: value_dict})

    return nucleus_dict


def load_wsi_with_sampler(input_path, wsi_name):
    assert os.path.exists(input_path)
    wsi_name = wsi_name + "_results"
    path = input_path / wsi_name
    sampler = SampleManager(path)
    sampler.load_samples()
    xs, ys, keys = sampler.get()

    values_list = list()
    for nucleus_data in xs:
        values_list.append(
            {
                "centroid": nucleus_data["centroid"],
                "bbox_bitmap": nucleus_data["bbox_bitmap"],
                "contour": nucleus_data["contour"],
            }
        )
    cell_dict = dict(zip(keys, values_list))

    return cell_dict


if __name__ == "__main__":
    os.chdir(
        "/home/ubuntu/NucMore-Projects/fabio/helper_scripts"
    )  # used when debugging with VS-Code and directory is not set correctly...

    pickle_unet_path = Path(
        "../nucmor/indra/new_unet_scores_KTB/classifier/detections-nuc1-4-picked-mask.pickle"
    )
    cvpp_output_path = Path("../nucmor/results/H_and_E/")
    wsi_name = "K102664"

    # unet_pickle_K102664_data = load_wsi_pickle(
    #     pickle_unet_path, wsi_name
    # )  # Something to make the formatting look nicer
    cvpp_256_K102664_data = load_wsi_with_sampler(
        cvpp_output_path / "CVpp_256_05_KTB", wsi_name
    )
    # print("Data loaded.")
    # cvpp_SAM_05_K102664_data = load_wsi_with_sampler(
    #     cvpp_output_path / "CVpp_SAM_05_KTB/no_resize", wsi_name
    # )
    # cvpp_SAM_025_K102664_data = load_wsi_with_sampler(
    #     cvpp_output_path / "CVpp_SAM_025_KTB/no_resize", wsi_name
    # )

    ########################################################
    # Execute the geojson export
    output_path = Path("../temp/geojson_output/")

    # dict_to_geojson_exact_outline(
    #     unet_pickle_K102664_data,
    #     output_path / "UNET_exact_K102664.geojson",
    #     color=0,
    # )
    # print("Export 1")
    # dict_to_geojson_exact_outline(
    #     cvpp_256_K102664_data,
    #     output_path / "CVpp_256_exact_K102664.geojson",
    #     color=1,
    # )
    # print("Export 2")

    # dict_to_geojson_exact_outline(
    #     cvpp_SAM_05_K102664_data,
    #     output_path / "CVpp_SAM_05_exact_K102664.geojson",
    #     color=2,
    # )
    # print("Export 3")

    # create_geojson_from_contour(
    #     cvpp_256_K102664_data,
    #     output_file=output_path / "CVpp_256_contour_K102664.geojson",
    # )

    # dict_to_geojson_exact_outline(
    #     cvpp_SAM_025_K102664_data,
    #     output_path / "CVpp_SAM_025_exact_K102664.geojson",
    #     color=3,
    # )
    print("Export 4")

    ##############
    # Identifying holes
    keys_with_holes = identify_holes(cvpp_256_K102664_data)

    print("Done")
