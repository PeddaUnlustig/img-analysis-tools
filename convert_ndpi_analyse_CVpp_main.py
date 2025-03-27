import os
import csv
import subprocess
from pathlib import Path
import pickle
from collections import defaultdict
from compressed_sampler import SampleManager
import slideio
import tifffile
from tqdm import tqdm

#########################################################################
########### Functions to convert NDPI to smaller TIFFs ##################


def ndpi_to_tiff_tiles(ndpi_path, tiff_outpath="", max_pixel=4294967295):
    # Read in the NDPI and convert it to a np.array
    slide = slideio.open_slide(str(ndpi_path), driver="NDPI")
    print("NDPI loaded:", ndpi_path)
    scene = slide.get_scene(0)
    print("Resolution: ", scene.resolution)
    print("Image size: ", scene.size)
    print(
        "dtype: ",
        scene.get_channel_data_type(0),
        scene.get_channel_data_type(1),
        scene.get_channel_data_type(2),
    )
    ndpi_block = scene.read_block()

    # Determine the number of reductions to fit the export
    height, width = ndpi_block.shape[0], ndpi_block.shape[1]
    new_size = height * width * ndpi_block.shape[2]
    denominator = 1
    while new_size > max_pixel:
        height = height / 2
        width = width / 2
        denominator = denominator * 2
        new_size = height * width * ndpi_block.shape[2]
    print(f"Saving NDPI in {denominator ** 2} tiles.")

    # Path declaration
    if tiff_outpath == "":
        tiff_outpath = str(Path(ndpi_path).parent)
    ndpi_prefix = Path(ndpi_path).stem
    output_path_template = str(tiff_outpath) + "/" + ndpi_prefix + "_tile_{}_{}.tiff"
    print("Start saving tiles at: ", output_path_template)

    # Execution of saving
    tile_height, tile_width = int(ndpi_block.shape[0] / denominator), int(
        ndpi_block.shape[1] / denominator
    )

    for n_image in tqdm(range(denominator**2)):
        current_height = (n_image // denominator) * tile_height
        current_width = (n_image % denominator) * tile_width
        tile = ndpi_block[i : i + tile_height, j : j + tile_width, :]
        # Save using tifffile with JPEG compression and tiling
        tifffile.imwrite(
            output_path_template.format(current_width, current_height),
            tile,
            tile=(256, 256),  # Define tile size (adjust as needed)
            compression="zlib",  # Use zlib compression (OpenSlide compatible)
            photometric="rgb",  # RGB color space
            resolution=(scene.resolution[0], scene.resolution[1]),
        )
        print(f"Saved tile {n_image + 1} with start coordinates ({current_width}, {current_height})")

    print("Converted ndpi:", ndpi_prefix)
    return None


def convert_bone_dataset(parent_dir: str, output_parent_dir: str):
    parent_dir, output_parent_dir = Path(parent_dir), Path(output_parent_dir)
    for root, dirs, files in os.walk(parent_dir):
        # Get the relative path from parent_dir
        relative_path = os.path.relpath(root, parent_dir)

        # Create corresponding output directory
        output_dir = output_parent_dir / relative_path
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each .ndpi file in current directory
        for file in files:
            if file.lower().endswith(".ndpi"):
                # Get full input filepath
                input_filepath = Path(root) / file
                ndpi_to_tiff_tiles(input_filepath, output_dir)
    return None


################################################################################################
################### Run CellVit++ on the whole directory #######################################


def create_tiff_csv_and_process(
    input_root,
    output_root,
    cvpp_dir,
    WSI_magnification=40.0,
    WSI_resolution=0.25,
    csv_name = "tiff_list.csv"
):
    """
    Create CSV of TIFF files, run CellViT detection, and concatenate pickle files
    """
    input_root = Path(input_root)
    output_root = Path(output_root)

    # !! Change when using specific list!
    csv_path = output_root / csv_name

    # Step 1: Create CSV with TIFF file paths
    csv_path = output_root / "tiff_filelist.csv"
    tiff_files = []

    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(".tiff"):
                print(root)
                tiff_files.append(str(Path(root) / file))

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["path", "magnification", "slide_mpp"])  # Header
        for tiff_path in tiff_files:
            writer.writerow([tiff_path, WSI_magnification, WSI_resolution])
    print(f"Created csv file with {len(tiff_files)} tiles.")
    print("Executing CellVit++")

    # Step 2: Execute Linux command
    cellvit_outdir = output_root / "CVpp-output"
    cellvit_outdir.mkdir(parents=True, exist_ok=True)

    command = [
        "python3",
        str(cvpp_dir) + "/cellvit/detect_cells.py",
        "--model",
        str(cvpp_dir) + "/checkpoints/CellViT-256-x40-AMP.pth",
        "--outdir",
        str(cellvit_outdir),
        "process_dataset",
        "--filelist",
        str(csv_path),
        "--wsi_extension",
        "tiff",
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing CellViT command: {e}")
        return

    print("CellVit++ successfully executed - converting pickle files...")
    return None


def concat_pickle_files(pickle_save_dir, cellvit_outdir):
    os.chdir(pickle_save_dir)
    # Step 3: Concatenate pickle files by WSI-name
    pickle_files = defaultdict(list)

    # Collect all pickle files
    for root, _, files in os.walk(cellvit_outdir):
        for file in files:
            if file.endswith("_celldict.pickle"):
                # Extract WSI-name from filename format: {WSI-name}_tile_{x}_{y}.tiff_celldict.pickle
                parts = file.split("_tile_")
                if len(parts) == 2:
                    wsi_name = parts[0]
                    pickle_files[wsi_name].append(Path(root) / file)

    # Process each WSI group
    print(f"Processing {len(pickle_files)} WSIs for output.")
    for wsi_name, file_list in pickle_files.items():
        combined_list = list()
        tile_coordinates_list = list()

        # Read and combine all pickles for this WSI
        for pickle_path in file_list:
            with open(pickle_path, "rb") as f:
                cell_dict = pickle.load(f)
                combined_list.extend(cell_dict)
                celldict_len = len(cell_dict)
            # Storing the coordinates data to use them later.
            xy_coord = (
                pickle_path.stem.split("_tile_")[1]
                .replace(".tiff_celldict", "")
                .split("_")
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

            ## Format is: {WSI-name}_tile_{width}_{height}.tiff_celldict.pickle --> {WSI-name}_tile_{x}_{y}.tiff_celldict.pickle
            x += int(tile_coordinates_list[counter][1][1])
            y += int(tile_coordinates_list[counter][1][0])

            cell_type = nucleus["type"]
            type_prob = nucleus["type_prob"]
            name = f"{wsi_name}_{x}_{y}"
            nucmore_samples.add(x=nucleus, y=(cell_type, type_prob), key=name)

        # Save the combined results
        nucmore_samples.save_samples()
        print(f"Processed {wsi_name} with {len(file_list)} tiles")
    return None


if __name__ == "__main__":
    # Execute the Conversion of bone NDPI files into smaller TIFF tiles.
    conv_mouse_input = (
        "/home/ubuntu/NucMore-Projects/fabio/nucmor/data/mouse_ndpi/umcg_may24/"
    )
    conv_mouse_output = (
        "/home/ubuntu/NucMore-Projects/fabio/nucmor/data/mouse_ndpi/tiff_tiles/"
    )
    convert_bone_dataset(conv_mouse_input, conv_mouse_output)

    # Execute CVpp onto the whole parent directory with every file that has extension .tiff
    cvpp_input_directory = "/home/ubuntu/NucMore-Projects/fabio/nucmor/data/mouse_ndpi/tiff_tiles/"  # = conv_mouse_output
    cvpp_output_directory = (
        "/home/ubuntu/NucMore-Projects/fabio/nucmor/results/umcg_may24_cvpp_results/"
    )
    cvpp_directory = "/home/ubuntu/NucMore-Projects/fabio/CellViT-plus-plus"
    create_tiff_csv_and_process(
        cvpp_input_directory, cvpp_output_directory, cvpp_directory
    )

    # Concat pickle files into one
    cvpp_directory = "/home/ubuntu/NucMore-Projects/fabio/temp/"
    concat_pickle_files(cvpp_output_directory, cvpp_directory)
