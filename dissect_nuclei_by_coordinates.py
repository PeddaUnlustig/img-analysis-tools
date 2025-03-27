from compressed_sampler import SampleManager
import pandas as pd
import os
import pickle
from collections import defaultdict
from pathlib import Path

#####################################################################################
# NOTE:
# This file was created to export only the nuclei, that are within a certain range of coordinates. 
# The UMCG dataset had several tissues in one WSI --> For our analysis only liver was needed.
# The script needs a previously created CSV file as input, which WSI are affected and the
# respective coordinates. 
######################################################################################

def parse_range(coord_str):
    """Convert string like '109k-134k', '>35k', '<34k' to numerical min/max values"""
    coord_str = coord_str.strip().replace("k", "000")

    if "-" in coord_str:  # Range case: '109000-134000'
        min_val, max_val = map(int, coord_str.split("-"))
        return min_val, max_val
    elif ">" in coord_str:  # Greater than case: '>35000'
        min_val = int(coord_str.replace(">", "")) + 1
        return min_val, float("inf")
    elif "<" in coord_str:  # Less than case: '<34000'
        max_val = int(coord_str.replace("<", "")) - 1
        return float("-inf"), max_val
    else:  # Single value case
        val = int(coord_str)
        return val, val


def ranges_overlap(min1, max1, min2, max2):
    """Check if two ranges overlap"""
    return min1 <= max2 and min2 <= max1


def extract_tile_coordinates(
    file_list: list[Path],
) -> tuple[tuple[int, int], tuple[int, int]]:
    tile_coords = [
        tuple(
            map(
                int,
                Path(p)
                .stem.split("_tile_")[1]
                .replace(".tiff_celldict", "")
                .split("_"),
            )
        )
        for p in file_list
    ]
    y_coords, x_coords = zip(*tile_coords)
    x_unique, y_unique = set(x_coords) - {0}, set(y_coords) - {0}
    return (min(x_unique), min(y_unique)), (max(x_unique), max(y_unique))


def filter_nuclei_by_range(nuclei_list, df_row, xy_coord, xy_tile_dimensions):
    """
    Filter nuclei list to include only those with centroids within the specified ranges.

    Args:
        nuclei_list: List of dictionaries containing nucleus data
        df_row: DataFrame row with x and y range strings
        xy_coord: Tuple of (x, a) top-left coordinates of the tile
        xy_tile_dimensions: Tuple of (width, height) tile dimensions

    Returns:
        List of nuclei dictionaries within the specified ranges
    """
    # Parse dataframe ranges
    x_min, x_max = parse_range(df_row["x"])
    y_min, y_max = parse_range(df_row["y"])

    # Calculate tile boundaries
    tile_x_start, tile_y_start = xy_coord
    tile_x_end = tile_x_start + xy_tile_dimensions[0] - 1
    tile_y_end = tile_y_start + xy_tile_dimensions[1] - 1

    # Effective ranges are the intersection of df ranges and tile boundaries
    effective_x_min = max(x_min, tile_x_start)
    effective_x_max = min(x_max, tile_x_end)
    effective_y_min = max(y_min, tile_y_start)
    effective_y_max = min(y_max, tile_y_end)

    # Filter nuclei based on centroid
    filtered_list = []
    for nucleus in nuclei_list:
        # Convert local centroid to global coordinates
        local_x, local_y = nucleus["centroid"]
        offset_x, offset_y = nucleus["offset_global"]
        # Calculate global coordinates
        # Global = local + patch_offset + global_offset + tile_offset
        global_x = local_x + tile_x_start  # + offset_x
        global_y = local_y + tile_y_start  # + offset_y

        # Check if global centroid is within effective ranges
        if (
            effective_x_min <= global_x <= effective_x_max
            and effective_y_min <= global_y <= effective_y_max
        ):
            filtered_list.append(nucleus)

    return filtered_list


# Use this list, if you need to exclude certain WSI from the export process
## e.g.: If the pickles have already been created, and dont need to be recreated.
exclude_wsi_list = []

############################################################################

def concat_split_pickle_files(pickle_save_dir, cellvit_outdir, split_wsi_csv_name):
    assert os.path.exists(pickle_save_dir)
    assert os.path.exists(cellvit_outdir)
    os.chdir(pickle_save_dir)
    pickle_files = defaultdict(list)

    split_wsi_df = pd.read_csv(
        pickle_save_dir / ".." / split_wsi_csv_name, header=0, delimiter=";"
    )
    print(f"Loaded csv with shape: {split_wsi_df.shape}")

    # Collect all pickle files
    for root, _, files in os.walk(cellvit_outdir):
        for file in files:
            if file.endswith("_celldict.pickle"):
                # Extract WSI-name from filename format: {WSI-name}_tile_{x}_{y}.tiff_celldict.pickle --> MAKE SURE THEY ARE SAVED IN THIS FORMAT AND NOT y,x! 
                parts = file.split("_tile_")
                if len(parts) == 2:
                    wsi_name = parts[0]
                    pickle_files[wsi_name].append(Path(root) / file)

    # Process each WSI group
    print(f"Processing {len(pickle_files)} WSIs for output.")
    for wsi_name, file_list in pickle_files.items():

        if wsi_name in exclude_wsi_list:
            continue

        combined_list = list()
        tile_coordinates_list = list()

        # Check if wsi in table:
        checker = None
        for df_index, wsi_path in enumerate(split_wsi_df.path):
            wsi_path = Path(wsi_path)
            if wsi_path.stem == wsi_name:
                checker = split_wsi_df.iloc[df_index, 1]
                break

        if checker == None:
            raise KeyError("WSI not in csv!")

        # # get dimension of tiles, if necessary
        if checker == 1:
            print("Filtering out nuclei, as multiple slides are in image.")
            xy_tile_dimensions, (max_x, max_y) = extract_tile_coordinates(file_list)

        # Read and combine all pickles for this WSI
        for pickle_path in file_list:
            xy_coord = (
                pickle_path.stem.split("_tile_")[1]
                .replace(".tiff_celldict", "")
                .split("_")
            )
            xy_coord = (int(xy_coord[0]), int(xy_coord[1]))

            # check if tile is in the coordinates:
            if checker == 1:
                df_x_min, df_x_max = parse_range(split_wsi_df["x"].iloc[df_index])
                df_y_min, df_y_max = parse_range(split_wsi_df["y"].iloc[df_index])
                tile_x_start, tile_y_start = xy_coord
                tile_x_end = tile_x_start + (xy_tile_dimensions[0] - 1)  # width
                tile_y_end = tile_y_start + xy_tile_dimensions[1] - 1  # height
                x_overlap = ranges_overlap(tile_x_start, tile_x_end, df_x_min, df_x_max)
                y_overlap = ranges_overlap(tile_y_start, tile_y_end, df_y_min, df_y_max)
                if x_overlap == False or y_overlap == False:
                    continue

            # Storing the coordinates data to use them later.
            with open(pickle_path, "rb") as f:
                cell_dict = pickle.load(f)

            if checker == 1:
                filtered_nuclei = filter_nuclei_by_range(
                    cell_dict,
                    split_wsi_df.iloc[df_index, :],
                    xy_coord,
                    xy_tile_dimensions,
                )
                combined_list.extend(filtered_nuclei)
                celldict_len = len(filtered_nuclei)
            elif checker == 0:
                combined_list.extend(cell_dict)
                celldict_len = len(cell_dict)
            else:
                raise ValueError(
                    "Checker is None or another value - something is wrong!"
                )

            tile_coordinates_list.append((celldict_len, xy_coord))

        nucmore_samples = SampleManager(filename=f"{wsi_name}_results")
        print(f"Sample Manager loaded for {wsi_name}...")

        # Process nuclei data
        counter = 0
        currentlen = tile_coordinates_list[0][0]
        for i, nucleus in enumerate(combined_list):
            # Centroid data is changed to fit onto the original image.
            x = int(round(nucleus["centroid"][0], 0))
            y = int(round(nucleus["centroid"][1], 0))

            if i >= currentlen:
                counter += 1
                currentlen += tile_coordinates_list[counter][0]

            # Format is: {WSI-name}_tile_{width}_{height}.tiff_celldict.pickle --> {WSI-name}_tile_{x}_{y}.tiff_celldict.pickle
            x += int(tile_coordinates_list[counter][1][0])
            y += int(tile_coordinates_list[counter][1][1])

            cell_type = nucleus["type"]
            type_prob = nucleus["type_prob"]
            name = f"{wsi_name}_{x}_{y}"
            nucmore_samples.add(x=nucleus, y=(cell_type, type_prob), key=name)

        # Save the combined results
        nucmore_samples.save_samples()
        print(f"Processed {wsi_name} with {len(file_list)} tiles")
    return None


if __name__ == "__main__":
    split_wsi_csv_name = "umcg_wsi_list.csv"
    pickle_save_dir = Path(
        "/home/ubuntu/NucMore-Projects/fabio/nucmor/results/umcg_may24_cvpp_results/NUSP_input"
    )
    cellvit_pickle_dir = pickle_save_dir / "../CVpp-output"
    concat_split_pickle_files(pickle_save_dir, cellvit_pickle_dir, split_wsi_csv_name)
    print("Done!")
