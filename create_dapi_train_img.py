from sampler import SampleManager
import numpy as np
from pathlib import Path
import os
import time
from scipy.ndimage import label
from scipy.ndimage import center_of_mass
from sklearn.model_selection import KFold
from tqdm import tqdm
from PIL import Image
import pandas as pd



############ Step 1: Create png and npy ##############

def _scale_to_255(png_img):
    scaled_img = png_img - png_img.min()
    scaled_img = scaled_img * (255/scaled_img.max())
    scaled_img = scaled_img.astype("uint8")
    return scaled_img

def _save_pngs(png_list: list, output_path, scale_to_255: bool = False):
    os.makedirs(output_path.parent / "train/images", exist_ok=True)
    for i in tqdm(range(len(png_list))):
        image_array = png_list[i]

        if scale_to_255:
            image_array = _scale_to_255(image_array)

        png_img = Image.fromarray(image_array, mode = "RGB")

        output_file = output_path.parent / "train/images" /(output_path.stem + f"_{i}.png")
        png_img.save(output_file)
    print(f"Saved pngs of {output_path.name} in folder {output_path.parent}/train/images")
    return None

def _create_inst_type_map(bitmap_list: list, output_path, threeD_y: bool = False):
    
    os.makedirs(output_path.parent / "train/labels", exist_ok=True)
    for i in tqdm(range(len(bitmap_list))):

        if len(bitmap_list[i].shape) == 3:
            bitmap = bitmap_list[i][:,:,-1]
        elif len(bitmap_list[i].shape) == 2:
            bitmap = bitmap_list[i]
        else:
            raise ValueError(f"Something is wrong with the bitmap! - image: {output_path.name}")
        
        # Create a simple type-map
        type_map = np.where(bitmap > 0, 1, bitmap)

        # Create instant-map
        inst_map, n_cells = label(type_map)
        print(f"Found {n_cells} cells.")


        # Create new type_map with random values of 1 or 2 per blob
        new_type_map = np.zeros_like(type_map)
        for j in range(1, n_cells + 1):
            # Randomly assign 1 or 2 to each blob
            random_label = np.random.choice([1, 2])
            new_type_map[inst_map == j] = random_label

        # Create np.array in needed shape --> inst and type map as items. 
        new_type_map = new_type_map.astype(np.uint32)
        print(f"Type map containing {np.unique(new_type_map)}")

        # check if there are actually 2 different classes in the type_map
        if len(np.unique(new_type_map)) != 3: 
            # skip this npy file
            print("Not enough cells - skipping.")
            continue

        inst_map = inst_map.astype(np.uint32)

        data = {'inst_map': inst_map, 'type_map': new_type_map}

        output_file = output_path.parent / "train/labels" / (output_path.stem + f"_{i}.npy")
        np.save(output_file, np.array(data, dtype=object))
    
    print(f"Saved all individual bitmaps of file {output_path.name} in folder {output_path.parent}/train/labels")

    return None

def create_training_input(pickle_path, output_path, threeD_y: bool = False, scale_png_to_255: bool = False):
    data = SampleManager(pickle_path)
    file_name = pickle_path.stem
    output_path = output_path / file_name
    
    print("Creating inst_maps and type_maps.")
    _create_inst_type_map(data.sample_ys, output_path, threeD_y)
    
    print("Saving images as pngs.")
    _save_pngs(data.sample_xs, output_path, scale_png_to_255)

    return None

############# Step 2: Create test img ############
# Optional --> Done in Jupyter Notebook for better visibility


############ Step 3: Create fold ##############
def fold_definition(input_dir, output_dir, n_splits:int = 5):
    os.makedirs(output_dir, exist_ok=True)
    # Get list of .npy files
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    npy_files = [os.path.splitext(f)[0] for f in npy_files]  # Remove .npy extension
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Create folds
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(npy_files)):
        # Get train and validation file names for this fold
        train_files = [npy_files[i] for i in train_idx]
        val_files = [npy_files[i] for i in val_idx]
        
        # Create fold directory
        fold_dir = os.path.join(output_dir, f'fold_{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Create and save train.csv
        train_df = pd.DataFrame(train_files)
        train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False, header = False)
        
        # Create and save val.csv
        val_df = pd.DataFrame(val_files)
        val_df.to_csv(os.path.join(fold_dir, 'val.csv'), index=False, header=False)
        
        print(f'Created fold_{fold_idx} with {len(train_files)} training and {len(val_files)} validation files')


    return None 


########################### Step 4: Create configs #########################################
# This is done manually


####################### Step 5: Create csv for detection training ##############################
def _get_centroid_from_npy(npy_path, outpath):
    # read npy
    raw_npy = np.read(npy_path, allow_pickle = True)
    data_npy = raw_npy.item()
    inst_map = data_npy["inst_map"]

    unique_labels = np.unique(inst_map)
    unique_labels = unique_labels[unique_labels != 0]

    # list definitions
    centroids_x = list()
    centroids_y = list()
    labels = list()
    for label in unique_labels:
        # Create a binary mask for the current label
        binary_mask = inst_map == label
        # Calculate the centroid (row, col) = (y, x)
        centroid = center_of_mass(binary_mask)
        # Append the centroid coordinates and label to lists
        random_label = np.random.choice([0, 1])
        labels.append(random_label)
        centroids_x.append(centroid[1])  # x-coordinate
        centroids_y.append(centroid[0])  # y-coordinate

    # trandform to dataframe
    df = pd.DataFrame({
        centroids_x, 
        centroids_y,
        labels,
    })
    

    return None

def convert_npy_dir_to_csv(npy_dir_path, outpath):
    npy_paths_list = list()
    for root, _, files in os.walk(npy_dir_path):
        for file in files:
            if file.lower().endswith(".npy"):  # Batch 1
                npy_paths_list.append(str(Path(root) / file))

    return None


######################################## Execute ######################################################

if __name__ == "__main__":
    dapi_path = Path("/home/ubuntu/NucMore-Projects/fabio/nucmor/data/dapi_samples")
    output_path = Path("/home/ubuntu/NucMore-Projects/fabio/CellViT-plus-plus/retrain_CVpp/DAPI/Ver01/")

    # # execute the code for the different image-pickles
    create_training_input(pickle_path=dapi_path/"aug20.pickle", output_path = output_path, threeD_y=False, scale_png_to_255 = True)
    create_training_input(pickle_path=dapi_path/"samples-garik_progeria_at.pickle", output_path=output_path, scale_png_to_255=True)
    create_training_input(pickle_path=dapi_path/"samples-garik_mouse_neurons_may21.pickle", output_path=output_path, scale_png_to_255=True)
    create_training_input(pickle_path=dapi_path/"samples_rev1.pickle", output_path=output_path, scale_png_to_255=True)
    create_training_input(pickle_path=dapi_path/"samples-rev2.pickle", output_path=output_path, scale_png_to_255=True, threeD_y=True)

    # create folds
    fold_definition(output_path/"train"/"labels", output_path/"splits/")


    print("Done!")