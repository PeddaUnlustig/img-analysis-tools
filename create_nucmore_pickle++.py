import pickle
import numpy as np
from pathlib import Path
import os
from skimage import transform, util
from compressed_sampler import SampleManager


# Function definition for rescaling etc.
def _center_in_128(resized_nucleus: np.array, h: int, w: int):
    """
    !! Write DocString
    """
    start_x = (128 - w) // 2
    start_y = (128 - h) // 2

    centered_array = np.zeros((128, 128))

    centered_array[start_y : start_y + h, start_x : start_x + w] = resized_nucleus
    return centered_array


def resize_nucleus(nucleus_dict: dict, STD_SIZE=80, scale=True):
    """
    !! Write DocString
    """
    image = np.array(nucleus_dict["bbox_bitmap"])
    h = len(image)
    w = len(image[0])
    if w > h:
        h = STD_SIZE * h // w
        w = STD_SIZE
    else:
        w = STD_SIZE * w // h
        h = STD_SIZE

    resized_nucleus = transform.resize(
        image=image,
        output_shape=(h, w),
        anti_aliasing=True,
        mode="constant",
        cval=0,
        preserve_range=True,
    )
    resized_nucleus = util.img_as_ubyte(resized_nucleus)
    if scale:
        resized_nucleus = _center_in_128(resized_nucleus, h, w)

    return resized_nucleus


if __name__ == "__main__":
   
    print("Started...")
    original_img_path = "/home/ubuntu/NucMore-Projects/fabio/nucmor/data/KTB"
    processed_data_path = "/home/ubuntu/NucMore-Projects/fabio/nucmor/results/H_and_E/CVpp_SAM_025_KTB/cellvit_output"

    wsi_filelist = [f for f in sorted(Path(processed_data_path).glob(f"**/*.pickle"))]
    print(len(wsi_filelist))
    print(wsi_filelist[0])

    save_dir = "/home/ubuntu/NucMore-Projects/fabio/nucmor/results/H_and_E/CVpp_SAM_025_KTB/no_resize"
    os.chdir(save_dir)

    for i, wsi_path in enumerate(wsi_filelist):
        wsi_path = Path(wsi_path)
        wsi_name = wsi_path.stem
        wsi_name = wsi_name.removesuffix(".svs_celldict")

        nucmore_samples = SampleManager(filename=f"{wsi_name}_results")
        print("Sample Manager loaded...")

        with open(wsi_path, "rb") as file:
            wsi_dict = pickle.load(file)

        for nucleus in wsi_dict:
            # !! Turned of the scaling - keep data raw - postprocessing can still be done afterwards.
            # rc_nucleus = resize_nucleus(nucleus, scale = False)
            x = int(round(nucleus["centroid"][0], 0))
            y = int(round(nucleus["centroid"][1], 0))
            cell_type = nucleus["type"]
            type_prob = nucleus["type_prob"]
            name = f"{wsi_name}_{x}_{y}"
            nucmore_samples.add(x=nucleus, y=(cell_type, type_prob), key=name)
        nucmore_samples.save_samples()
        print("Image ", i, " done.")
    print("Analysis done.")
