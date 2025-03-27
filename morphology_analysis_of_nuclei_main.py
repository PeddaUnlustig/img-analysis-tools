import pandas as pd
from pathlib import Path
from compressed_sampler import SampleManager
import morphology_scan_nuclei_functions as snmf
from tqdm import tqdm


def create_morph_df(parent_dir, csv_name: str = None, outdir=None):
    pickle_filelist = [f for f in sorted(Path(parent_dir).glob(f"**/*_results"))]

    print("Loading pickles: ")
    slide_list = list()
    total_nuclei = 0
    for WSI_path in tqdm(pickle_filelist):
        sampler = SampleManager(filename=WSI_path)
        sampler.load_samples()
        total_nuclei += len(sampler.sample_xs)
        morph_results_list = snmf.scan_slides(sampler)
        slide_list.extend(morph_results_list)
    print("Total analysed nuclei: ", total_nuclei)

    morph_df = pd.DataFrame(
        slide_list,
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
    if csv_name == None:
        csv_name = "morph.csv"
    elif type(csv_name) == str:
        if csv_name.split(".")[-1] != "csv":
            csv_name = str(csv_name + ".csv")
    else:
        raise ValueError("CSV name is not a string.")

    if outdir == None:
        outdir = parent_dir

    print("Saving CSV as: ", csv_name, " - in directory: ", outdir)

    morph_df.to_csv(outdir + csv_name, index=False, header=True)
    print("Morph df saved.")
    return None


if __name__ == "__main__":
    datapath = "../nucmor/results/H_and_E/"
    outdir = "../nucmor/morph_analysis/"
    # csv_name = "bone_ndpi_morph_minsize-10.csv"
    # create_morph_df(datapath, csv_name)
    print("Creating 3 morph dfs")
    create_morph_df(
        datapath + "CVpp_SAM_05_KTB/no_resize/",
        "CVpp_SAM_05_morph_min-size_10.csv",
        outdir,
    )
    print("First one done")
    create_morph_df(
        datapath + "CVpp_SAM_025_KTB/no_resize/",
        "CVpp_SAM_025_morph_min-size_10.csv",
        outdir,
    )
    print("Second one done")
    create_morph_df(
        datapath + "CVpp_256_05_KTB/", "CVpp_256_05_morph_min-size_10.csv", outdir
    )
    print("Third one done!")
    print("ALLLLLL DONNNNE!")
